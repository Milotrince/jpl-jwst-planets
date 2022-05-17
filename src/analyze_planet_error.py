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


def analyze_planet_error(expected, input_data, found, processed_data, star_location=None, acceptable_pixel_error=5, debug=False):
    if star_location == None:
        star_location = input_data.shape/2

    nc = get_nircam_with_options()

    fig, axes = plt.subplots(1, 2)
    axes[0].set_title('Input Data')
    axes[1].set_title('Result')

    result_dict = {
        'expected delta x': [],
        'expected delta y': [],
        'expected r': [],
        'expected theta': [],
        'found delta x': [],
        'found delta y': [],
        'expected brightness contrast': [],
        'found brightness contrast': [],
        'found planet flux': []
    }

    unused_from_fnd_list = found.copy()
    
    for exp in expected:
        exp_x_px = convert_to_px(exp['delta_Y']) + star_location[0]
        exp_y_px = convert_to_px(exp['delta_X']) + star_location[1]
        exp['location'] = (exp_x_px, exp_y_px)
        
        distances = [math.dist(exp['location'], f['location']) for f in found]
        min_dist = min(distances) if len(distances) > 0 else 10

        if debug:
            print('~~~~ exp px', exp['location'])
            print('---- fnd loc', [f['location'] for f in found])
            print('==== distances', distances)

        axes[0].plot(exp_y_px, exp_x_px, 'yo', fillstyle='none', markersize=14)
        result_dict['expected delta x'].append(exp['delta_X'])
        result_dict['expected delta y'].append(exp['delta_Y'])
        result_dict['expected r'].append(exp['r'])
        result_dict['expected theta'].append(exp['theta'])
        result_dict['expected brightness contrast'].append(exp['brightness'])

        if min_dist < acceptable_pixel_error: 
            min_index = distances.index(min_dist)

            fnd = found[min_index]
            if fnd in unused_from_fnd_list:
                unused_from_fnd_list.remove(fnd)

            fnd_px_x, fnd_px_y = fnd['location']
            plot_x, plot_y = fnd['relative location']
            axes[1].plot(plot_x, plot_y, 'go', fillstyle='none', markersize=14)

            # result_dict['found delta x'].append((star_location[0] - fnd_px_x) * nc.pixelscale)
            # result_dict['found delta y'].append((star_location[1] - fnd_px_y) * nc.pixelscale)
            result_dict['found delta x'].append((fnd_px_y - star_location[1]) * nc.pixelscale)
            result_dict['found delta y'].append((fnd_px_x - star_location[0]) * nc.pixelscale)
            result_dict['found brightness contrast'].append(fnd['brightness contrast'])
            result_dict['found planet flux'].append(fnd['planet flux'])
        else:
            result_dict['found delta x'].append(None)
            result_dict['found delta y'].append(None)
            result_dict['found brightness contrast'].append(None)
            result_dict['found planet flux'].append(None)

    for fnd in unused_from_fnd_list:
        fnd_px_x, fnd_px_y = fnd['location']
        axes[1].plot(fnd_px_x, fnd_px_y, 'go', fillstyle='none', markersize=14)

        result_dict['found delta x'].append((fnd_px_y - star_location[1]) * nc.pixelscale)
        result_dict['found delta y'].append((fnd_px_x - star_location[0]) * nc.pixelscale)
        result_dict['found brightness contrast'].append(fnd['brightness contrast'])
        result_dict['found planet flux'].append(fnd['planet flux'])

        result_dict['expected delta x'].append(None)
        result_dict['expected delta y'].append(None)
        result_dict['expected brightness contrast'].append(None)
        result_dict['expected r'].append(None)
        result_dict['expected theta'].append(None)


    df = pd.DataFrame(result_dict)
    df['found r,theta'] = df.apply(lambda row: convert_xy_to_rtheta(row['found delta x'], row['found delta y']), axis=1)
    df['found r'] = df.apply(lambda row: row['found r,theta'][0], axis=1)
    df['found theta'] = df.apply(lambda row: row['found r,theta'][1], axis=1)

    def df_distance(ex, ey, fx, fy):
        if ex and ey and fx and fy:
            return math.dist([ex, ey], [fx, fy])
        else:
            return None
    df['distance'] = df.apply(lambda row: df_distance( row['expected delta x'], row['expected delta y'], row['found delta x'], row['found delta y'] ), axis=1)
    df['brightness % error'] = df.apply(lambda row: calc_error(row['found brightness contrast'], row['expected brightness contrast']), axis=1)

    found_count = int(df.count()['distance'])
    found_text = f"{found_count}/{len(expected)}"
    fp_count = len(df.index) - int(df.count()['expected delta x'])

    nc = get_nircam_with_options()
    columns = ['distance', 'found planet flux', 'brightness % error', 'expected brightness contrast', 'found brightness contrast', 'expected delta x', 'expected delta y', 'found delta x', 'found delta y', 'expected r', 'expected theta', 'found r', 'found theta']
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
    figurefilepath = f'./outs/testextract_{timestamp}.png'
    print(f'figure saved to {figurefilepath}')
    plt.savefig(figurefilepath)
    if debug:
        plt.show()

    logfilepath = f'./outs/testextract_{timestamp}.txt'
    with open(logfilepath, 'w') as f:
        f.write(logtext)
    print(f'log saved to {logfilepath}')

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

    print("Extracting planets...")
    processed_data, found = extract_planets(input_data, debug=False)

    analyze_planet_error(expected, input_data, found, processed_data)
