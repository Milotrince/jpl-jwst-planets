from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, interpolate

# reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html

if __name__ == '__main__':
	data1 = fits.getdata('./data/reduction_data_F444W.fits')
	data2 = fits.getdata('./outs/psf_test_100.fits')

	corr = signal.correlate2d(data1, data2, boundary='symm', mode='same')
	print(corr.shape)
	y, x = np.unravel_index(np.argmax(corr), corr.shape)
	print(np.max(corr))

	# interp_x = interpolate.interp1d(x, y,, kind='cubic')
	# maximums = signal.argrelextrema(corr, np.greater)
	# print(maximums)

	# try to get x maximums, to identify all the planets
	# or subtract the found, and get maximum
	# make nircam psf with same maximum location; and then scale shift to match place

	# want more precise, subpixel location

	# can resample
	# look for "parabolic shape" in x and y
	# and then interpolate to get the subpixel value
	# https://stackoverflow.com/questions/13730913/how-to-perform-image-cross-correlation-with-subpixel-accuracy-with-scipy
	# https://stackoverflow.com/questions/70986860/sub-pixel-precision-template-matching-using-normalized-cross-correlation-normxc


	print(f'x {x}, y {y}')

	# fig, (ax_1, ax_2, ax_corr) = plt.subplots(3, 1, figsize=(6, 15))

	# ax_1.imshow(data1, cmap='gray')
	# ax_1.set_title('reduction data')

	# ax_2.imshow(data2, cmap='gray')
	# ax_2.set_title('psf (template)')

	# ax_corr.imshow(corr, cmap='gray')
	# ax_corr.set_title('Cross-correlation')

	# ax_1.plot(x, y, 'rx')
	# ax_2.plot(x, y, 'rx')
	# ax_corr.plot(x, y, 'rx')
	plt.subplot(131)
	plt.imshow(data1)
	plt.subplot(132)
	plt.imshow(data2)
	plt.subplot(133)
	plt.imshow(corr)

	plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
	cax = plt.axes([0.85, 0.1, 0.05, 0.8])
	plt.colorbar(cax=cax)

	plt.savefig('./outs/crosscorrelation.png')
	plt.show()

	# fig.savefig('./outs/crosscorrelation.png')
	# fig.show()

	# next step: extract astrometry/photometry