import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import webbpsf
from astropy.io import fits

from extract_planets import extract_planets
from test_extract_planets import analyze_planet_error
from simulate_data import make_image_with_planets

def get_simulated_image(planet_id):
    return fits.getdata(f'./data/{planet_id}-jw01412-c0000_t000_nircam_f444w-maskrnd-sub320a430r_i2d.fits')

def print_summary_for_subregion(title, df):
    found_count = int(df.count()['distance'])
    found_text = f"{found_count}/{len(expected)}"
    fp_count = len(df.index) - int(df.count()['expected x'])
    cols = ['distance', 'brightness % error']
    partial_df = df[cols]

    print()
    print(f'===== {title} =====')
    print(f'found: ', found_text)
    print(f'false positives: ', fp_count)
    print('------------------')
    means = partial_df.mean()
    stds = partial_df.std(ddof=0)
    for col in cols:
        print(f'{col} mean: ', means[col])
        print(f'{col} std : ', stds[col])
    print('==================')


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description='Test extraction of planet astrometry and photometry.')
    parser.add_argument('--amount', metavar='n', help='amount of images to generate and test', default=None, type=int)
    parser.add_argument('--file', metavar='f', help='input json file', default=None)
    parser.add_argument('--simulate', metavar='s', help='whether or not to generate (simplified) simulated data on the fly', default=None)

    args = parser.parse_args()

    print(f"Generating data from {args.file}...")

    f = open(args.file, 'r')
    jsondata = json.load(f)

    nc = webbpsf.NIRCam()
    nc.filter = jsondata['filter']
    nc.image_mask = jsondata['image_mask']
    nc.pupil_mask = jsondata['pupil_mask']
    nc.fov_arcsec = jsondata['fov_arcsec']
    nc.pixelscale = jsondata['pixelscale']

    df = pd.DataFrame()
    count = 0
    for star_id, planet_dict in jsondata["data"].items():
        if args.amount and args.amount == count:
            break
        count += 1

        print(f"Generating test image for star {star_id}...")
        expected = list(planet_dict.values())
        if args.simulate:
            input_data = make_image_with_planets(expected)
        else:
            input_data = get_simulated_image(star_id)

        threshold_mult = 0.6
        threshold = np.max(input_data) * threshold_mult

        acceptable_position_error = 2 # in pixel distance

        print(f"Extracting planets for star {star_id}...")
        processed_data, found = extract_planets(input_data, threshold, debug=True)

        new_df = analyze_planet_error(expected, input_data, found, processed_data, xy_error=acceptable_position_error, debug=False)
        planet_keys = list(planet_dict.keys())
        new_df['planet id'] = planet_keys + [None]*(len(new_df)-len(planet_keys)) # the None is to account when false positives are found
        df = pd.concat([df, new_df], axis=0)
        # print(new_df)

    print('==== full output =====')
    print(df.to_string(max_rows=None, max_cols=None))

    found_count = int(df.count()['distance'])
    found_text = f"{found_count}/{len(expected)}"
    fp_count = len(df.index) - int(df.count()['expected x'])

    df_r = df[['expected r']]
    print_summary_for_subregion("summary for subregion r < 0.5", df[df_r < 0.5])
    print_summary_for_subregion("summary for subregion 0.5 <= r < 1", df[(df_r >= 0.5) & (df_r < 1)])
    print_summary_for_subregion("summary for subregion r >= 1", df[df_r >= 1])

    filename = f"output_{timestamp}.xlsx"
    print('mc test result output:', filename)
    df.to_excel(filename) 

