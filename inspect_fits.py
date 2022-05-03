from astropy.io import fits
# reference: https://learn.astropy.org/tutorials/FITS-images.html

# returns an astropy.io.fits.HDUlist containing PSF and header
# psf = fits.open('./outs/psf_test_100.fits')
# data = fits.open('./data/reduction_data_F444W.fits')
# data = fits.open('./data/reduction_data_F356W.fits')
data = fits.open('./data/000-jw01412-c0000_t000_nircam_f444w-maskrnd-sub320a430r_i2d.fits')

data.info()
print('=======')

# image_data = data[0].data
# print(type(image_data)) # numpy.ndarray

print(data[0].header)
print('--------')
print(data[1].header)
print('--------')
header = data[1].header
print(header['PA_V3'])

data.close()

# one-liner:
# image_data = fits.getdata('./outs/psf_test.fits')