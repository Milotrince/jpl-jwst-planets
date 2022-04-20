import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import webbpsf


from extract_planets import extract_planets
from test_extract_planets import analyze_planet_error

from simulate_data import make_image_with_planets

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description='Test extraction of planet astrometry and photometry.')
    parser.add_argument('--amount', metavar='n', help='amount of images to generate and test', default=None, type=int)
    parser.add_argument('--file', metavar='n', help='input json file', default=None)

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
        input_data = make_image_with_planets(expected)

        threshold_mult = 0.6
        threshold = np.max(input_data) * threshold_mult

        acceptable_position_error = 2 # in pixel distance

        print(f"Extracting planets for star {star_id}...")
        processed_data, found = extract_planets(input_data, threshold, debug=False)

        new_df = analyze_planet_error(expected, input_data, found, processed_data, xy_error=acceptable_position_error, debug=False)
        planet_keys = list(planet_dict.keys())
        new_df['planet id'] = planet_keys + [None]*(len(new_df)-len(planet_keys)) # the None is to account when false positives are found
        df = pd.concat([df, new_df], axis=0)
        # print(new_df)

    print('==== full output =====')
    print(df.to_string(max_rows=None, max_cols=None))

    cols = ['distance', 'brightness % error']
    partial_df = df[cols]

    found_count = int(df.count()['distance'])
    found_text = f"{found_count}/{len(expected)}"
    fp_count = len(df.index) - int(df.count()['expected x'])

    print()
    print('==== summary =====')
    print(f'found: ', found_text)
    print(f'false positives: ', fp_count)
    print('------------------')
    means = partial_df.mean()
    stds = partial_df.std(ddof=0)
    for col in cols:
        print(f'{col} mean: ', means[col])
        print(f'{col} std : ', stds[col])
    print('==================')

    # partial_df.hist()
    # plt.show()

    filename = f"output_{timestamp}.xlsx"
    print('mc test result output:', filename)
    df.to_excel(filename) 

