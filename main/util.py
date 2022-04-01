import numpy as np
import configparser
import webbpsf

nc = None

def convert_rtheta_to_xy(r, theta):
	pixels = round(nc.fov_arcsec / nc.pixelscale)
	x = pixels/2 - (r/nc.pixelscale) * np.sin(np.deg2rad(theta)) 
	y = pixels/2 + (r/nc.pixelscale) * np.cos(np.deg2rad(theta))
	return (x, y)


def get_test_psf(nc, offset_r=0, offset_theta=0):
	nc.options['source_offset_r'] = offset_r
	nc.options['source_offset_theta'] = offset_theta
	psf = nc.calc_psf(fov_arcsec=nc.fov_arcsec, oversample=1)
	# print('getting psf', 'filter', nc.filter, 'pixels', nc.pixelscale, 'fov_arcsec', nc.fov_arcsec)
	# psf.info()
	return psf[0].data


def convert_to_arcsec(px):
    """
    Convert unit from pixels to arcseconds
    """
    nc = get_nircam_with_options()
    return px * nc.pixelscale


def convert_xy_to_rtheta(x, y):
    nc = get_nircam_with_options()
    pixels = round(nc.fov_arcsec / nc.pixelscale)
    x = x - pixels/2 
    y = y - pixels/2
    # each pixel corresponds to __ arcseconds
    r = np.sqrt(x**2 + y**2) * nc.pixelscale
    theta = np.rad2deg( np.arctan2(y, x) ) - 90
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

def get_defaults():
	"""
	Returns dictionary of default values in options.cfg
	"""
	config = configparser.ConfigParser()
	config.read('options.cfg')
	return config['default']