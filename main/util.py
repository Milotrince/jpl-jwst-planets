import numpy as np
import configparser
import webbpsf
import operator

nc = None

def recenter(img, new_center):
    length = min(map(lambda i, j: min(i - j, j), img.shape, new_center))
    start = tuple(map(lambda a, c: c - length, img.shape, new_center))
    end = tuple(map(lambda a: a + length*2, start))
    slices = tuple(map(slice, start, end))
    return img[slices]


def reshape_centered(img, bounding):
    if np.all(img.shape > bounding):
        # crop
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]
    else:
        # add empty outside
        result = np.zeros(bounding)
        start = tuple(map(lambda a, b: b//2-a//2, img.shape, bounding))
        end = tuple(map(operator.add, start, img.shape))
        slices = tuple(map(slice, start, end))
        result[slices] = img
        return result


if __name__ == "__main__":
    from scipy import misc
    import matplotlib.pyplot as plt
    
    img = misc.ascent()
    print(img.shape)
    reshaped = reshape_centered(img, (1000,1000))
    plt.imshow(reshaped)
    plt.show()


def convert_rtheta_to_xy(r, theta):
    pixels = round(nc.fov_arcsec / nc.pixelscale)
    x = pixels/2 - (r/nc.pixelscale) * np.sin(np.deg2rad(theta)) 
    y = pixels/2 + (r/nc.pixelscale) * np.cos(np.deg2rad(theta))
    return (x, y)


def get_test_psf(nc, offset_r=0, offset_theta=0, oversample=1):
    nc.options['source_offset_r'] = offset_r
    nc.options['source_offset_theta'] = offset_theta
    psf = nc.calc_psf(fov_arcsec=nc.fov_arcsec, oversample=oversample)
    # print('getting psf', 'filter', nc.filter, 'pixels', nc.pixelscale, 'fov_arcsec', nc.fov_arcsec)
    # psf.info()
    return psf[0].data


def convert_to_arcsec(px):
    """
    Convert unit from pixels to arcseconds
    """
    nc = get_nircam_with_options()
    pixels = round(nc.fov_arcsec / nc.pixelscale)
    return (px-pixels/2) * nc.pixelscale


def convert_to_px(arcsec):
    """
    Convert unit from arcseconds to pixels
    """
    return arcsec / nc.pixelscale


def convert_xy_to_rtheta(x, y):
    """
    Convert rectangular to polar units
    """
    if x == None or y == None:
        return (None, None)
    r = np.sqrt(x**2 + y**2)
    theta = np.rad2deg( np.arctan2(y, x) ) - 90
    if theta < 0:
        theta += 360
    return (r, theta)


def get_nircam_with_options():
    """
    Returns webbpsf.NIRCam with parameters in options.cfg
    """
    global nc
    if not nc:
        config = configparser.ConfigParser()
        config.read('options.cfg')

        nc = webbpsf.NIRCam()
        nc.filter = config['NIRCam']['filter']
        nc.image_mask = config['NIRCam']['image_mask']
        nc.pupil_mask = config['NIRCam']['pupil_mask']
        nc.fov_arcsec = config['NIRCam'].getfloat('fov_arcsec')
        nc.pixelscale = config['NIRCam'].getfloat('pixelscale')

    return nc

def get_constants():
    """
    Returns dictionary of default values in options.cfg
    """
    config = configparser.ConfigParser()
    config.read('options.cfg')
    return config['constants']