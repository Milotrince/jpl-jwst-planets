import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from astropy.io import fits
from PyAstronomy import pyasl
# https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/quadextreme.html


def get_maxes(data2d, threshold, dp=2, debug=False):
	output = []

	sizex, sizey = data2d.shape
	for row in range(sizex):
		y_i_offset = 0
		y = data2d[row]
		while np.max(y) > threshold:
			x = np.arange(len(y))
			epos, mi, xb, yb, p = pyasl.quadExtreme(x, y, mode="max", dp=(dp,dp), fullOutput=True)
			output.append(dict(
				row=row,
				epos=epos+y_i_offset,
				max_val = y[mi],
				mi=mi+y_i_offset,
				xb=xb,
				yb=yb ))
			if debug:
				print('=======')
				print("index of row: ", row)
				print("index of max: ", mi)
				print("value of max: ", y[mi])
				print("maximum found by parabolic fit: ", epos)
				print("xb: ", xb)
				print("yb: ", yb)
				plt.plot(x, y, 'bp')
				plt.plot(xb+x[mi], yb, 'rp')
				plt.show()
			y_i_offset = mi+dp
			y = data2d[row][y_i_offset:]
	return output

def group_maxes(maxes):
	groups = []
	current_group = []

	for (i, max_data) in enumerate(maxes):
		# if prev_max_data and prev_max_data['row']+1 == max_data['row']:
		if i+1 < len(maxes) and max_data['row']+1 == maxes[i+1]['row']:
			current_group.append(max_data)
		else:
			current_group.append(max_data)
			groups.append(current_group)
			current_group = []
	return groups

def get_data2d_from_grouped_maxes(grouped_maxes):
	data = np.zeros(corr.shape)
	planet_points = []
	for grouped in grouped_maxes:
		# print('len', len(grouped))
		max_in_group = grouped[0]
		# print('before', max_in_group['max_val'])
		for p in grouped:
			# print('row', p['row'])
			if p['max_val'] > max_in_group['max_val']:
				# print('new max is', p['max_val'])
				max_in_group = p
			data[p['row']][p['mi']+p['xb']] = p['yb']

		planet_points.append( (max_in_group['mi'], max_in_group['row']) )
		# print('after', max_in_group['max_val'])
		# print('=====')
	return (data, planet_points)



if __name__ == '__main__':
	reduction_data = fits.getdata('./data/reduction_data_F444W.fits')
	psf_data = fits.getdata('./outs/psf_test_100.fits')

	corr = signal.correlate2d(reduction_data, psf_data, boundary='symm', mode='same')

	maxes = get_maxes(corr, threshold=0.045, dp=3)
	if len(maxes) == 0:
		print("No maxes found.")
		exit(0)

	grouped_maxes = group_maxes(maxes)
	print('groups: ', len(grouped_maxes))

	(data2d, points) = get_data2d_from_grouped_maxes(grouped_maxes)
	for (x, y) in points:
		print('x', x, 'y', y)
		plt.plot(x, y, 'mo', fillstyle='none')
	plt.imshow(data2d)
	plt.colorbar()
	plt.show()


