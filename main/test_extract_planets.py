import argparse
import math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from simulate_data import get_test_data
from extract_planets import extract_planets
from util import *


def save_testdata(input_data, expected):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f'in/{timestamp}_test_data.npy', input_data)
    np.save(f'in/{timestamp}_test_expected.npy', expected)

    print(f"saved test data to in/{timestamp}")


def load_testdata(timestamp):
    data = np.load(f'in/{timestamp}_test_data.npy', allow_pickle=True)
    expected = np.load(f'in/{timestamp}_test_expected.npy', allow_pickle=True)
    return data, expected


def calc_error(actual, expected):
    """
    returns percent error
    """
    if actual == None or expected == None:
        return None
    return abs((actual - expected) / expected * 100)


def analyze_planet_error(expected, input_data, found, processed_data, xy_error, debug=False):

    fig, axes = plt.subplots(1, 2)
    axes[0].set_title('Input Data')
    axes[1].set_title('Result')

    result_dict = {
        'expected x': [],
        'expected y': [],
        'found x': [],
        'found y': [],
        'expected brightness': [],
        'found brightness': [],
    }

    unused_from_fnd_list = found.copy()

    for exp in expected:
        exp_x = convert_to_px(exp['delta_X'])
        exp_y = convert_to_px(exp['delta_Y'])
        exp['location'] = (exp_x, exp_y)
        # exp_x, exp_y = exp['location']
        distances = [math.dist(exp['location'], f['location']) for f in found]
        min_dist = min(distances)

        axes[0].plot(exp_x, exp_y, 'yo', fillstyle='none', markersize=14)
        result_dict['expected x'].append(exp_x)
        result_dict['expected y'].append(exp_y)
        result_dict['expected brightness'].append(exp['brightness'])

        if min_dist < xy_error: 
            min_index = distances.index(min_dist)

            fnd = found[min_index]
            fnd_x, fnd_y = fnd['location']
            if fnd in unused_from_fnd_list:
                unused_from_fnd_list.remove(fnd)

            axes[1].plot(fnd_x, fnd_y, 'go', fillstyle='none', markersize=14)
            result_dict['found x'].append(fnd_x)
            result_dict['found y'].append(fnd_y)
            result_dict['found brightness'].append(fnd['brightness'])
        else:
            result_dict['found x'].append(None)
            result_dict['found y'].append(None)
            result_dict['found brightness'].append(None)

    for fnd in unused_from_fnd_list:
        fnd_x, fnd_y = fnd['location']

        axes[1].plot(fnd_x, fnd_y, 'ro', fillstyle='none', markersize=14)

        result_dict['found x'].append(fnd_x)
        result_dict['found y'].append(fnd_y)
        result_dict['found brightness'].append(fnd['brightness'])
        result_dict['expected x'].append(None)
        result_dict['expected y'].append(None)
        result_dict['expected brightness'].append(None)

    df = pd.DataFrame(result_dict)
    df['expected r, theta'] = df.apply(lambda row: convert_xy_to_rtheta(row['expected x'], row['expected y']), axis=1)
    df['expected r'] = df.apply(lambda row: convert_xy_to_rtheta(row['expected x'], row['expected y'])[0], axis=1)
    df['expected theta'] = df.apply(lambda row: convert_xy_to_rtheta(row['expected x'], row['expected y'])[1], axis=1)
    df['found r, theta'] = df.apply(lambda row: convert_xy_to_rtheta(row['found x'], row['found y']), axis=1)
    # These values are in pixels. Convert to arcsec
    df[['expected x', 'expected y', 'found x', 'found y']] = df[['expected x', 'expected y', 'found x', 'found y']].apply(convert_to_arcsec)
    def df_distance(ex, ey, fx, fy):
        if ex and ey and fx and fy:
            return math.dist([ex, ey], [fx, fy])
        else:
            return None
    df['distance'] = df.apply(lambda row: df_distance( row['expected x'], row['expected y'], row['found x'], row['found y'] ), axis=1)
    df['brightness % error'] = df.apply(lambda row: calc_error(row['found brightness'], row['expected brightness']), axis=1)

    found_count = int(df.count()['distance'])
    found_text = f"{found_count}/{len(expected)}"
    fp_count = len(df.index) - int(df.count()['expected x'])

    nc = get_nircam_with_options()
    columns = ['distance', 'brightness % error', 'expected x', 'expected y', 'found x', 'found y', 'expected brightness', 'found brightness', 'expected r', 'expected theta', 'found r, theta']
    logtext = f"""
filter = {nc.filter}
fov_arcsec = {nc.fov_arcsec}
pixelscale = {nc.pixelscale}

--------- FOUND: {found_text}, FALSE POSITIVES: {fp_count} ----------\n{df[columns].to_string(max_rows=None, max_cols=None)}
    """
    if debug:
        print(logtext)

    im = axes[0].imshow(input_data, cmap="viridis")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.06)
    im = axes[1].imshow(processed_data, cmap="RdBu")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.06, norm=mcolors.TwoSlopeNorm(0))


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # save plot
    plt.suptitle(f'Planet Extraction Test\n{nc.filter}, fov_arcsec {nc.fov_arcsec}, pixelscale {nc.pixelscale}\nFound: {found_text}, False Positives: {fp_count}')
    plt.savefig(f'./outs/testextract_{timestamp}.png')
    if debug:
        plt.show()

    with open(f'./outs/testextract_{timestamp}.txt', 'w') as f:
        f.write(logtext)

    return df



if __name__ == '__main__':

    numpoints = 3

    parser = argparse.ArgumentParser(description='Test extraction of planet astrometry and photometry.')
    parser.add_argument('--test', metavar='t', help='test data date (loads from /in/ folder)')
    args = parser.parse_args()

    if args.test:
        print("Loading data...")
        input_data, expected = load_testdata(args.test)
    else:
        print("Generating test data...")
        input_data, expected = get_test_data(numpoints=numpoints, debug=False)
        save_testdata(input_data, expected)

    threshold_mult = 0.6
    threshold = np.max(input_data) * threshold_mult

    acceptable_position_error = 2 # in pixel distance

    print("Extracting planets...")
    processed_data, found = extract_planets(input_data, threshold, debug=False)

    analyze_planet_error(expected, input_data, found, processed_data, xy_error=acceptable_position_error)
