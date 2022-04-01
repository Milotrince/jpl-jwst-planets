import numpy as np
import json
from random import random, randint
from util import convert_rtheta_to_xy, get_test_psf, get_nircam_with_options


def get_random_planets(nc, amount=3, psf=None):
	if type(psf) == type(None):
		psf = get_test_psf(nc, 0, 0)

	planets = []
	for i in range(amount):
		r = (random() * 0.5 + 0.2) * nc.fov_arcsec/2 # in arcsec
		theta = random() * 360
		x, y = convert_rtheta_to_xy(r, theta)

		# precision = 4
		# x = round(x, precision)
		# y = round(y, precision)
		# r = round(r, precision)
		# theta = round(theta, precision)

		# because planets are much less bright
		brightness = (random() * 0.5 + 1 ) * 0.1 
		planets.append(dict(brightness=np.max(psf) * brightness, location=(x,y), r=r, theta=theta))
		# planets.append(dict(brightness=brightness, location=(x,y), r=r, theta=theta))
	return planets

def get_simulated_planets_list(nc, amount = 100, max_planets_per_star = 5):
	psf = get_test_psf(nc, 0, 0)
	stars = []
	for i in range(amount):
		num_planets = randint(0, max_planets_per_star)
		stars.append(get_random_planets(nc, num_planets, psf))
	return stars


if __name__ == '__main__':
	nc = get_nircam_with_options()

	data = dict(
		filter = nc.filter,
		image_mask = nc.image_mask,
		pupil_mask = nc.pupil_mask,
		fov_arcsec = nc.fov_arcsec,
		pixelscale = nc.pixelscale,
		data = get_simulated_planets_list(nc, amount=100, max_planets_per_star=5),
	)

	f = open('simulated_planets.json', 'w')
	json.dump(data, f)
	f.close()
	print('Data written to simulated_planets.json')
