import argparse
from audioop import cross
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
from util import reshape_centered, get_nircam_with_options, get_test_psf, convert_to_arcsec, convert_xy_to_rtheta, get_constants

# Example to run:
# python3 ./main/extract_planets.py --fits ./data/reduction_data_F356W.fits

nc = get_nircam_with_options()
nc.image_mask = None
nc.pupil_mask = None

constants = get_constants()
star_flux = constants.getfloat('star_flux')
conversion_constant = constants.getfloat('conversion_constant')
sigma = constants.getfloat('blur_sigma')
roll = constants.getfloat("roll")

psf = get_test_psf(nc)
max_psf_value = np.max(psf)

def load_fits_data(filename):
    return fits.getdata(filename)

def calc_brightness_contrast(planet_brightness):
    return (planet_brightness * conversion_constant / max_psf_value * nc.pixelscale) / star_flux
 
def get_cross_corr(img, template):
    return signal.correlate2d(img, template, boundary='symm', mode='same')

def get_subtract_psf(r, theta, max_data_value, shape):
    subtract_psf = get_test_psf(nc, offset_r=r, offset_theta=theta)
    subtract_psf = ndimage.rotate(subtract_psf, roll, reshape=True)
    subtract_psf = gaussian_filter(subtract_psf, sigma=sigma)
    subtract_psf = (subtract_psf * max_data_value / np.max(subtract_psf)) # adjust brightness
    subtract_psf = reshape_centered(subtract_psf, shape)
    return subtract_psf


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
    # if data.size > rolled_psf.size:
    #     data = reshape_centered(data, rolled_psf.shape)
    # elif data.size < rolled_psf.size:
    #     rolled_psf = reshape_centered(rolled_psf, data.shape)

    def get_corresponding_pixel_before_crop(x, y):
        print('**********')
        print('in ', x, y)
        d_w, d_h = input_data.shape
        psf_w, psf_h = rolled_psf.shape
        # if d_w != psf_w:
        #     x = (d_w - psf_w)/2 + x
        # if d_h != psf_h:
        #     y = (d_h - psf_h)/2 + y
        print('out', x, y)
        print('**********')
        return x, y

    crosscorrelation = get_cross_corr(data, rolled_psf)

    threshold_mult = constants.getfloat('threshold_mult')
    threshold = np.max(crosscorrelation) * threshold_mult

    while np.max(crosscorrelation) > max(threshold, 0):
        step_number += 1

        y_of_max, x_of_max = np.unravel_index(np.argmax(crosscorrelation), crosscorrelation.shape)
        if debug:
            print(f'===== in loop, step {step_number}=====')
            print('x',x_of_max,'y',y_of_max)

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

        max_data_value = data[y_of_max, x_of_max]

        distances = [math.dist([subpixel_x, subpixel_y], found['location']) for found in fnd_list]

        if debug:
            if len(distances) > 0: 
                print('min distance:', min(distances))


        # start = tuple(map(lambda m, s: max(m-5, 0), (y_of_max, x_of_max), data.shape))
        # end = tuple(map(lambda m, s: min(m+5, s), (y_of_max, x_of_max), data.shape))
        # slices = tuple(map(slice, start, end))

        if debug:
            plt.subplot(2,2,1)
            plt.title(f"step {step_number}: data ")
            plt.imshow(data)
            plt.colorbar()

        pixel_distance_same_planet_threshold = constants.getfloat('pixel_distance_same_planet_threshold')
        if len(distances) < 1 or min(distances) > pixel_distance_same_planet_threshold:
            # get psf at location
            # These arcsec values are based from center of image, not true center of star
            x_arcsec = (subpixel_x - data.shape[0]/2) * nc.pixelscale
            y_arcsec = (subpixel_y - data.shape[1]/2) * nc.pixelscale
            r, theta = convert_xy_to_rtheta(x_arcsec, y_arcsec)
            if debug:
                print('subpixel x',subpixel_x,'y',subpixel_y)
                print('arcsec x',x_arcsec,'arcsec y',y_arcsec)
                print('r',r,'theta',theta)
            # subtraction step
            subtract_psf = get_subtract_psf(r, theta, max_data_value, data.shape) 
            data = data - subtract_psf

            fnd_list.append({
                'brightness contrast': calc_brightness_contrast(max_data_value),
                'location': get_corresponding_pixel_before_crop(subpixel_x,subpixel_y),
                'relative location': (subpixel_x,subpixel_y),
                'subtracted': subtract_psf})
            
        elif min(distances) < pixel_distance_same_planet_threshold:
            # Found two spots very close to each other. Average the two then subtract
            min_index = distances.index(min(distances))
            other = fnd_list.pop(min_index)
            other_x, other_y = other['relative location']
            avg_x = np.mean([other_x, subpixel_x])
            avg_y = np.mean([other_y, subpixel_y])
            brightness_contrast = calc_brightness_contrast(max_data_value)
            avg_bc = np.mean([other['brightness contrast'], brightness_contrast])

            x_arcsec = (subpixel_x - data.shape[0]/2) * nc.pixelscale
            y_arcsec = (subpixel_y - data.shape[1]/2) * nc.pixelscale
            r, theta = convert_xy_to_rtheta(x_arcsec, y_arcsec)

            # undo the previous subtraction
            data += other['subtracted']/2
            # do subtraction again
            subtract_psf = get_subtract_psf(r, theta, max_data_value, data.shape) 
            data = data - subtract_psf

            fnd_list.append({
                'brightness contrast': avg_bc,
                'location': get_corresponding_pixel_before_crop(avg_x,avg_y),
                'relative location': (avg_x,avg_y),
                'subtracted': subtract_psf})


        if debug:
            plt.subplot(2,2,2)
            plt.title(f"crosscorr; max={np.max(crosscorrelation)}")
            plt.plot(x_of_max, y_of_max, 'rx')
            plt.imshow(crosscorrelation)
            plt.colorbar()

            plt.subplot(2,2,3)
            plt.title(f"subtrahend")
            plt.imshow(subtract_psf)
            # plt.imshow(subtract_psf, norm=LogNorm())
            # plt.colorbar()

            plt.subplot(2,2,4)
            plt.title(f"result")
            plt.imshow(data)
            plt.colorbar()

            print('---- debug ----')
            print('star location:', star_location)
            print('subpixel:', subpixel_x, subpixel_y)
            print('psf max bright: ', np.max(psf))
            print('brightness: ', max_data_value)
            print('max crossc: ', np.max(crosscorrelation))
            print('threshold : ', threshold)
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