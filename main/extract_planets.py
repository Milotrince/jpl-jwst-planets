import argparse
import math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal
from astropy.io import fits
from PyAstronomy import pyasl # https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/quadextreme.html


from simulate_data import get_test_data
from util import get_nircam_with_options, get_test_psf, convert_xy_to_rtheta

# Example to run:
# python3 ./main/extract_planets.py --fits ./data/reduction_data_F356W.fits

nc = get_nircam_with_options()



def load_fits_data(filename):
    return fits.getdata(filename)


def extract_planets(input_data, threshold, pixel_distance_same_planet_threshold=2, debug=False):
    """
    Returns data, fnd_list where
        data is input_data after being processed and
        fnd_list is array of location/brightness of found items
    """
    data = input_data.copy()

    fnd_list = []
    step_number = 0 # for debug
    while np.max(data) > max(threshold, 0):
        step_number += 1

        y_of_max, x_of_max= np.unravel_index(np.argmax(data), data.shape)
        if debug:
            print(f'===== in loop, step {step_number}=====')
            print('x',x_of_max,'y',y_of_max)

        radius = 5

        y = data[y_of_max][x_of_max-radius:x_of_max+radius+1]
        x = np.arange(len(y))
        epos, mi, xb, yb, p = pyasl.quadExtreme(x, y, mode="max", dp=(2,2), fullOutput=True)
 
        subpixel_y = y_of_max - radius + epos

        y = data.transpose()[x_of_max][y_of_max-radius:y_of_max+radius+1]
        x = np.arange(len(y))
        epos, mi, xb, yb, p = pyasl.quadExtreme(x, y, mode="max", dp=(2,2), fullOutput=True)

        subpixel_x = x_of_max - radius + epos

        brightness = input_data[y_of_max, x_of_max]

        distances = [math.dist([subpixel_x, subpixel_y], found['location']) for found in fnd_list]

        if debug:
            if len(distances) > 0: 
                print('min distance:', min(distances))

        if len(distances) < 1 or min(distances) > pixel_distance_same_planet_threshold:

            # get psf at location
            r, theta = convert_xy_to_rtheta(subpixel_x, subpixel_y)
            if debug:
                print('subpixel x',subpixel_x,'y',subpixel_y)
                print('r',r,'theta',theta)
                # test_x, test_y = convert_rtheta_to_xy(r, theta)
                # print('reconverted x',test_x,'reconverted y',test_y)
            psf = get_test_psf(nc, offset_r=r, offset_theta=theta)
            psf = (psf * brightness / np.max(psf)) # adjust brightness
            # subtraction step
            data = data - psf

            fnd_list.append(dict(brightness=brightness, location=(subpixel_x,subpixel_y), psf=psf))
        elif min(distances) < pixel_distance_same_planet_threshold:
            # Found two spots very close to each other. Average the two then subtract
            min_index = distances.index(min(distances))
            other = fnd_list.pop(min_index)
            other_x, other_y = other['location']
            avg_x = np.mean([other_x, subpixel_x])
            avg_y = np.mean([other_y, subpixel_y])
            avg_b = np.mean([other['brightness'], brightness])

            # undo the previous subtraction
            data += other['psf']
            # get psf at location
            r, theta = convert_xy_to_rtheta(avg_x, avg_y)
            psf = get_test_psf(nc, offset_r=r, offset_theta=theta)
            psf = (psf * avg_b / np.max(psf)) # adjust brightness
            # subtraction step
            data -= psf
            fnd_list.append(dict(brightness=avg_b, location=(avg_x,avg_y), psf=psf))


        if debug:
            print('---- debug ----')
            print('psf max bright: ', np.max(psf))
            print('brightness: ', brightness)
            print('threshold: ', threshold)
            print('max      : ', np.max(data))

            plt.title(f'Step {step_number}')
            plt.imshow(data)
            plt.colorbar()
            plt.show()

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

    threshold_mult = 0.4
    threshold = np.max(input_data) * threshold_mult

    processed_data, found = extract_planets(input_data, threshold, debug=True)

    plt.title(f'{args.fits} Result')
    for f in found:
        x, y = f['location']
        plt.plot(x, y, 'ro', fillstyle='none', markersize=14)
    plt.imshow(processed_data)
    plt.colorbar()
    plt.show()

    df = pd.DataFrame(found)
    print(df)