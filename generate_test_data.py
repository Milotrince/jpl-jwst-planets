from re import I
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime

import webbpsf

# R_TO_PIXELS = 16

def get_test_psf(offset_r=0, offset_theta=0, fov_pixels=100):
	nc = webbpsf.NIRCam()
	nc.filter = 'F444W'
	# nc.filter = 'F356W'
	nc.image_mask='MASK430R'
	nc.pupil_mask='CIRCLYOT'
	nc.options['source_offset_r'] = offset_r
	nc.options['source_offset_theta'] = offset_theta
	psf = nc.calc_psf( fov_pixels=fov_pixels, oversample=1,)
	# psf.info()
	return psf[0].data

# def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    # return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

# def generate_point(r=2, size=10): 
	# x = np.linspace(-r, r, num=size)
	# y = np.linspace(-r, r, num=size)
	# x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
	# z = gaus2d(x, y)
	# return z


def get_test_data(numpoints=5, debug=False):
	"""
	return (data, dict of inserted information)
	"""
	psf_data = get_test_psf() # star
	# psf_data = np.zeros((100,100))

	inserted = []

	for i in range(numpoints):
		# brightness_multiplier = random.randrange(2,3)
		# size = random.randrange(3,8)
		# p = generate_point(size=size, r=2) * brightness_multiplier
		# mu, sigma = 50, 15
		# x = int(np.random.normal(mu, sigma))
		# y = int(np.random.normal(mu, sigma))
		# psf_data[x:x+size,y:y+size] += p
		# inserted.append(dict(size=size, brightness=np.max(p), location=(y+size/2,x+size/2)))

		brightness_multiplier = 1#random.random()*0.5 + 1

		r = random.random() * 2
		theta = random.random() * 360
		p = get_test_psf(offset_r=r, offset_theta=theta) * brightness_multiplier
		y, x = np.unravel_index(np.argmax(p), p.shape)
		inserted.append(dict(brightness=np.max(p), location=(x,y)))
		psf_data += p

	if debug:
		plt.imshow(psf_data)
		plt.colorbar()
		plt.show()
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		plt.savefig(f'./outs/testdata_{len(inserted)}_{timestamp}.png')

	return (psf_data, inserted)


if __name__ == '__main__':
	get_test_data(debug=True)