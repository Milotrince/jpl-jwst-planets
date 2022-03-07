import webbpsf
import os
os.environ['WEBBPSF_PATH'] = '/Users/trinity/Documents/GitHub/Work/jwst/webbpsf-data'

# reference: https://webbpsf.readthedocs.io/en/latest/usage.html#using-api

nc = webbpsf.NIRCam()
nc.filter = 'F444W'
# returns an astropy.io.fits.HDUlist containing PSF and header
# psf = nc.calc_psf(outfile='./outs/psf_test.fits')
psf = nc.calc_psf(outfile='./outs/psf_test_100.fits', fov_pixels=100, oversample=1)
psf.info()