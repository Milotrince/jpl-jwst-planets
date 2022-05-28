import numpy as np
import configparser
import webbpsf
import operator
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel

nc = None


def get_existing_simulated_image(filepath):
    """
    Return image and star center from .fits file.
    """
    im = fits.open(filepath)

    starCoords = SkyCoord([float(im[0].header['TARG_RA'])], [float(im[0].header['TARG_DEC'])], unit='deg')

    wcs = WCS(im[('SCI',1)].header, naxis=2)
    # Geoff: star position is -1,-1 relative to XREF_SCI,YREF_SCI 
    starPosition = skycoord_to_pixel(starCoords,wcs)
    x_center = starPosition[0][0] + 1
    y_center = starPosition[1][0] + 1

    return im[1].data, (x_center, y_center)


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


def convert_rtheta_to_xy(r, theta):
    pixels = round(nc.fov_arcsec / nc.pixelscale)
    x = pixels/2 - (r/nc.pixelscale) * np.sin(np.deg2rad(theta)) 
    y = pixels/2 + (r/nc.pixelscale) * np.cos(np.deg2rad(theta))
    return (x, y)


def generate_psf(nc, offset_r=0, offset_theta=0, oversample=1):
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
    while theta < 0:
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