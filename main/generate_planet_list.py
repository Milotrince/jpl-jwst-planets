import argparse
import json
import numpy as np
from datetime import datetime

import random
from util import convert_rtheta_to_xy, get_test_psf, get_nircam_with_options, convert_to_arcsec


def get_random_planets(nc, amount=3, psf=None, r_min_max=(0,3), brightness_min_max=(-7, -4)):
    if type(psf) == type(None):
        psf = get_test_psf(nc, 0, 0)
    r_min, r_max = r_min_max

    planets = []
    for i in range(amount):
        # r = (random() * 0.5 + 0.2) * nc.fov_arcsec/2 # in arcsec
        r = random.uniform(r_min, r_max)

        theta = random.random() * 360
        x, y = convert_rtheta_to_xy(r, theta)
        x_arcsec = convert_to_arcsec(x)
        y_arcsec = convert_to_arcsec(y)

# Geoff:
# It should have at least three columns - delta_X, delta_Y, and brightness -
# the X,Y position of the planet relative to the star and the brightness of the planet relative to the star.
        # because planets are much less bright
        # brightness = np.max(psf) * (random.random() * 0.5 + 1 ) * 0.1
        b_min, b_max = brightness_min_max
        brightness = (random.random() * 10) * 10 ** random.randint(b_min, b_max)

        # planets.append(dict(brightness=brightness, delta_X=x_arcsec, delta_Y=y_arcsec, r=r, theta=theta))
        planets.append(dict(brightness=brightness, delta_X=x_arcsec, delta_Y=y_arcsec, r=r, theta=theta, location=(x,y)))
    return planets

def get_simulated_planets_list(nc, amount = 100, max_planets_per_star = 5):
    psf = get_test_psf(nc, 0, 0)
    stars = {}
    for i in range(amount):
        num_planets = random.randint(0, max_planets_per_star)
        star_id = f'{i:03d}'
        planets = get_random_planets(nc, num_planets, psf)
        planet_ids = [f'{star_id}_{n}' for n in range(num_planets)]
        stars[star_id] = dict(zip(planet_ids, planets))
    return stars


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description='Test extraction of planet astrometry and photometry.')
    parser.add_argument('--amount', metavar='n', help='amount of images to generate and test', default=None)
    args = parser.parse_args()

    amount = int(args.amount) if args.amount else 50

    nc = get_nircam_with_options()

    data = dict(
        filter = nc.filter,
        image_mask = nc.image_mask,
        pupil_mask = nc.pupil_mask,
        fov_arcsec = nc.fov_arcsec,
        pixelscale = nc.pixelscale,
        data = get_simulated_planets_list(nc, amount=amount, max_planets_per_star=5),
    )

    fname = f'simulated_planets_{timestamp}.json'
    f = open(fname, 'w')
    json.dump(data, f)
    f.close()
    print(f'Data written to {fname}')
