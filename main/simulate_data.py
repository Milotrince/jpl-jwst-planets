import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime

from util import get_nircam_with_options, get_test_psf, convert_rtheta_to_xy
from generate_planet_list import get_random_planets


nc = get_nircam_with_options()

def make_image_with_planets(planets_data, debug=False):
    psf_data = get_test_psf(nc, offset_r=0.01, offset_theta=1) # star at not exact spot
    psf_data -= get_test_psf(nc, offset_r=0, offset_theta=0) # block star
    for i, planet in enumerate(planets_data):
        if debug:
            print(f'planet {i} inserted at', 'r', planet['r'], 'theta', planet['theta'], 'x,y', planet['location'])
        p = get_test_psf(nc, offset_r=planet['r'], offset_theta=planet['theta'])
        # p = p * planet['brightness']
        p = p * planet['brightness'] / np.max(p)
        psf_data += p
    return psf_data



def get_test_data(numpoints=2, debug=False):
    """
    return (data, dict of inserted information)
    """
    planets = get_random_planets(nc, numpoints)
    psf_data = make_image_with_planets(planets, debug)

    if debug:
        plt.imshow(psf_data)
        plt.colorbar()
        plt.show()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'./outs/testdata_{len(planets)}_{timestamp}.png')

    return (psf_data, planets)


if __name__ == '__main__':
    get_test_data(numpoints=3, debug=True)