import argparse
import webbpsf
import math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal, ndimage
from scipy.ndimage.filters import gaussian_filter
from astropy.io import fits
from PyAstronomy import pyasl # https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/quadextreme.html
from PyAstronomy.pyaC.pyaErrors.pyaValErrs import PyAValError 
from matplotlib.colors import LogNorm

from simulate_data import get_test_data
from util import reshape_centered, get_nircam_with_options, generate_psf, convert_xy_to_rtheta, get_constants

# Example to run:
# python3 ./main/extract_planets.py --fits ./data/reduction_data_F356W.fits

nc = get_nircam_with_options()

constants = get_constants()
star_flux = constants.getfloat('star_flux')
sigma = constants.getfloat('blur_sigma')
roll = constants.getfloat('roll')
photometry_aperture_radius = constants.getfloat('photometry_aperture_radius')
photometry_multiplier = constants.getfloat('photometry_multiplier')
# Input image is in MJy/sr
# MJy/sr to Jy/arcsec2
# MJy/sr * 2.35045e-11 sr/arcsec2 * 1e6 Jy/MJy
MJy_per_sr_to_Jy_per_arcsec2 = 0.0000235045


nc_nomask = webbpsf.NIRCam()
nc_nomask.filter = nc.filter
nc_nomask.fov_arcsec = nc.fov_arcsec
nc_nomask.pixelscale = nc.pixelscale
psf = generate_psf(nc=nc_nomask)

def load_fits_data(filename):
    return fits.getdata(filename)

def create_circular_mask(shape, center=None, radius=None):
    w, h = shape
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def mask_image(img, mask, mask_value=0):
	masked = img.copy()
	masked[~mask] = mask_value
	return masked

def calc_photometry(found_x, found_y, _psf, data):
    """
    found_x, found_y is subpixel location of planet in data
    data is in MJy/sr
    _psf is in normalized flux/px, with image and pupil mask
    MJy_per_sr_to_Jy_per_arcsec2 = 0.0000235045
        MJy/sr * 2.35045e-11 sr/arcsec2 * 1e6 Jy/MJy
    pixelscale is arcsec/px
    photometry_aperture_radius is pixel radius of circle around found x,y point
        constant set in options.cfg
    """
    mask = create_circular_mask(data.shape, center=(found_x, found_y), radius=photometry_aperture_radius)
    data_masked = mask_image(data, mask)
    psf_masked = mask_image(_psf, mask)

    # plt.suptitle("Aperture Photometry")
    # plt.subplot(2,2,1)
    # plt.title("Reference PSF")
    # plt.imshow(_psf, norm=LogNorm())
    # plt.colorbar()
    # plt.subplot(2,2,2)
    # plt.title("Reference PSF (masked)")
    # plt.imshow(psf_masked, norm=LogNorm())
    # plt.colorbar()
    # plt.subplot(2,2,3)
    # plt.title("Data")
    # plt.imshow(data)
    # plt.colorbar()
    # plt.subplot(2,2,4)
    # plt.title("Data (masked)")
    # plt.imshow(data_masked, norm=LogNorm())
    # plt.colorbar()
    # plt.show()
    # print("****** data masked sum", data_masked.sum())
    # print("****** psf masked sum", psf_masked.sum())

    return data_masked.sum() * MJy_per_sr_to_Jy_per_arcsec2 * (nc.pixelscale**2) / psf_masked.sum() * photometry_multiplier

 
def get_cross_corr(img, template):
    return signal.correlate2d(img, template, boundary='symm', mode='same')

def get_subtract_psf(psf, data_value_at_maxcorr):
    subtract_psf = ndimage.rotate(psf, roll, reshape=True)
    subtract_psf = gaussian_filter(subtract_psf, sigma=sigma)
    subtract_psf = (subtract_psf * data_value_at_maxcorr / np.max(subtract_psf)) # adjust brightness
    return subtract_psf


def get_subpixel(data, x_of_max, y_of_max):
    radius = 5

    y = data[y_of_max][x_of_max-radius:x_of_max+radius+1]
    x = np.arange(len(y))
    dp = (3,3)
    try:
        epos, mi = pyasl.quadExtreme(x, y, mode="max", dp=dp)

        subpixel_y = y_of_max - radius + epos

        y = data.transpose()[x_of_max][y_of_max-radius:y_of_max+radius+1]
        x = np.arange(len(y))
        epos, mi = pyasl.quadExtreme(x, y, mode="max", dp=dp)

        subpixel_x = x_of_max - radius + epos
    except PyAValError:
        subpixel_x = x_of_max
        subpixel_y = y_of_max
    return (subpixel_x, subpixel_y)


def extract_planets(input_data, star_location=None, debug=False):
    """
    Returns data, fnd_list where
        data is input_data after being processed and
        fnd_list is array of location/brightness of found items
    """
    if star_location == None:
        star_location = input_data.shape/2

    fnd_list = []
    step_number = 0 # for debug

    rolled_psf = ndimage.rotate(psf, roll, reshape=True)

    data = input_data.copy()
    if data.size > rolled_psf.size:
        # print('data size before', data.shape)
        data = reshape_centered(data, rolled_psf.shape)
        # print('data size after ', data.shape)
    elif data.size < rolled_psf.size:
        rolled_psf = reshape_centered(rolled_psf, data.shape)

    def get_corresponding_pixel_before_crop(x, y):
        # print('**********')
        # print('in ', x, y)
        d_w, d_h = input_data.shape
        psf_w, psf_h = rolled_psf.shape
        if d_w != psf_w:
            x = (d_w - psf_w)/2 + x
        if d_h != psf_h:
            y = (d_h - psf_h)/2 + y
        # print('out', x, y)
        # print('**********')
        return x, y

    def perform_step_at(data, found_x, found_y):
        # get psf at location
        # These arcsec values are based from center of image, not true center of star
        # x_arcsec = (found_x - (data.shape[0]-1)/2.0) * nc.pixelscale
        # y_arcsec = (found_y - (data.shape[1]-1)/2.0) * nc.pixelscale

        x_arcsec = (found_x - (data.shape[0]-1)/2.0) * nc.pixelscale
        y_arcsec = (found_y - (data.shape[1]+1)/2.0) * nc.pixelscale

        r, theta = convert_xy_to_rtheta(x_arcsec, y_arcsec)
        if debug:
            print('subpixel x',found_x,'y',found_y)
            # print('arcsec x',x_arcsec,'arcsec y',y_arcsec)
            # print('r',r,'theta',theta)

        psf_with_mask = generate_psf(nc, offset_r=r, offset_theta=theta)
        # psf_nomask = generate_psf(nc_nomask, offset_r=r, offset_theta=theta)

        # planet_flux = calc_photometry(found_x, found_y, psf_nomask, data)
        planet_flux = calc_photometry(found_x, found_y, psf_with_mask, data)

        # subtraction step
        subtract_psf = get_subtract_psf(psf_with_mask, data_value_at_maxcorr) 
        data = data - subtract_psf


        plt.subplot(2,2,3)
        plt.title(f"Subtrahend")
        plt.imshow(subtract_psf)
        plt.colorbar()

        fnd_list.append({
            'planet flux': planet_flux,
            'brightness contrast': planet_flux / star_flux,
            'location': get_corresponding_pixel_before_crop(found_x,found_y),
            'relative location': (found_x,found_y),
            'subtracted': subtract_psf})
        return data


    crosscorrelation = get_cross_corr(data, rolled_psf)

    # threshold_mult = constants.getfloat('threshold_mult')
    # threshold = np.max(crosscorrelation) * threshold_mult
    threshold = constants.getfloat('brightness_threshold')

    while np.max(data) > max(threshold, 0):
        step_number += 1

        y_of_max, x_of_max = np.unravel_index(np.argmax(crosscorrelation), crosscorrelation.shape)
        if debug:
            print(f'===== in loop, step {step_number}=====')
            print('x',x_of_max,'y',y_of_max)

        data_value_at_maxcorr = data[y_of_max, x_of_max]
        subpixel_x, subpixel_y = get_subpixel(data, x_of_max, y_of_max)

        distances = [math.dist([subpixel_x, subpixel_y], found['location']) for found in fnd_list]

        if debug:
            if len(distances) > 0: 
                print('min distance:', min(distances))

            plt.suptitle(f"Step {step_number}")
            plt.subplot(2,2,1)
            plt.title(f"Data")
            plt.imshow(data)
            plt.colorbar()

        data = perform_step_at(data, subpixel_x, subpixel_y)

        # EXPERIMENTAL
        # pixel_distance_same_planet_threshold = constants.getfloat('pixel_distance_same_planet_threshold')
        # if len(distances) < 1 or min(distances) > pixel_distance_same_planet_threshold:
        #     data = perform_step_at(data, subpixel_x, subpixel_y)
            
        # elif min(distances) < pixel_distance_same_planet_threshold:
        #     # Found two spots very close to each other. Average the two then subtract
        #     min_index = distances.index(min(distances))
        #     other = fnd_list.pop(min_index)
        #     other_x, other_y = other['relative location']
        #     avg_x = np.mean([other_x, subpixel_x])
        #     avg_y = np.mean([other_y, subpixel_y])
        #     # undo the previous subtraction
        #     data += other['subtracted']/2
        #     data = perform_step_at(data, avg_x, avg_y)


        if debug:
            plt.subplot(2,2,2)
            # plt.title(f"crosscorr; max={np.max(crosscorrelation)}")
            plt.title(f"Cross-correlation")
            plt.plot(x_of_max, y_of_max, 'rx')
            plt.imshow(crosscorrelation)
            plt.colorbar()

            plt.subplot(2,2,4)
            plt.title(f"Result")
            plt.imshow(data)
            plt.colorbar()

            # print('star location:', star_location)
            # print('subpixel:', subpixel_x, subpixel_y)
            # print('max crossc: ', np.max(crosscorrelation))
            print('max in data with subtractions: ', np.max(data))
            print('threshold                    : ', threshold)
            plt.show()
        crosscorrelation = get_cross_corr(data, rolled_psf)

    return data, fnd_list



if __name__ == '__main__':

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    numpoints = 3

    parser = argparse.ArgumentParser(description='Extract exoplanet astrometry and photometry.')
    parser.add_argument('--fits', metavar='f', help='load fits file data')
    parser.add_argument('--filter', help='filter used for fits file')
    args = parser.parse_args()

    if args.fits:
        input_data = load_fits_data(args.fits)
        if not args.filter:
            print('Error: When specifying the fits data, you should also have the --filter flag.')
            exit(0)
        nc.filter = args.filter
        width, height = input_data.shape
        nc.fov_arcsec = width * nc.pixelscale
        # TODO: properly change nc settings so psfs will be the right size
    else:
        input_data, expected = get_test_data(numpoints=numpoints)

    processed_data, found = extract_planets(input_data, debug=True)

    plt.title(f'{args.fits} Result')
    for f in found:
        x, y = f['location']
        plt.plot(x, y, 'ro', fillstyle='none', markersize=14)
    plt.imshow(processed_data)
    plt.colorbar()
    plt.show()

    df = pd.DataFrame(found)
    print(df)