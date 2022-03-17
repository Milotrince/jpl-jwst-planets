import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from astropy.io import fits
from PyAstronomy import pyasl
# https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/quadextreme.html
from datetime import datetime
import webbpsf
from generate_test_data import get_test_data


nc = webbpsf.NIRCam()
nc.filter = 'F444W'
# nc.filter = 'F356W'
nc.image_mask = 'MASK430R'
nc.pupil_mask = 'CIRCLYOT'


def convert_xy_to_rtheta(x, y, pixels=100):
    middle = pixels/2
    x = x - middle
    y = y - middle
    r = np.sqrt(x**2 + y**2) / 15.6
    theta = np.rad2deg( np.arctan2(y, x) ) - 90
    return (r, theta)


def get_psf(offset_r=0, offset_theta=0, fov_pixels=100):
    global nc
    nc.options['source_offset_r'] = offset_r
    nc.options['source_offset_theta'] = offset_theta
    psf = nc.calc_psf( fov_pixels=fov_pixels, oversample=1,)
    # psf.info()
    return psf[0].data



if __name__ == '__main__':

    input_data, expected = get_test_data(numpoints=2)
    data = input_data.copy()


    found_points = []
    threshold_mult = 0.5
    threshold = np.max(data) * threshold_mult
    while np.max(data) > threshold:
        print('inloop\n========')
        y, x = np.unravel_index(np.argmax(data), data.shape)
        print('x',x,'y',y)
        found_points.append((x,y))

        r, theta = convert_xy_to_rtheta(x, y)
        print('r',r,'t',theta)
        psf = get_psf(offset_r=r, offset_theta=theta)
        data = data - psf
        # threshold = np.max(data) * threshold_mult

        plt.imshow(data)
        plt.colorbar()
        plt.show()

    print('threshold: ', threshold)
    print('========')


    # fig, axes = plt.subplots(1, 2)

    # plot original data
    plt.subplot(121)
    plt.title('Input Data')
    plt.legend('expected')
    plt.set_cmap('viridis')
    print('expected:\t', len(expected))
    for e in expected:
        x, y = e['location']
        print('x', x, 'y', y)
        plt.plot(x, y, 'gx', fillstyle='none')
    plt.imshow(input_data)
    plt.colorbar()

    # plot found data
    plt.subplot(122)
    plt.title('Result')
    plt.legend('found')
    plt.set_cmap('RdBu')

    print('found:\t', len(found_points))
    for (x, y) in found_points:
        print('x', x, 'y', y)
        plt.plot(x, y, 'mo', fillstyle='none')

    plt.imshow(data)
    plt.colorbar()

    # save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'./outs/testdata_{len(expected)}_{timestamp}.png')
    plt.show()

