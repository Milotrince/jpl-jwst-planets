from astropy.io import fits
# reference: https://learn.astropy.org/tutorials/FITS-images.html

# returns an astropy.io.fits.HDUlist containing PSF and header
psf = fits.open('./outs/psf_test_100.fits')
# psf = fits.open('./data/reduction_data_F444W.fits')

psf.info()

image_data = psf[0].data
print(type(image_data))

psf.close()

# one-liner:
# image_data = fits.getdata('./outs/psf_test.fits')