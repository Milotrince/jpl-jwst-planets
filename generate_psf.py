import matplotlib.pyplot as plt
import webbpsf
from matplotlib.colors import LogNorm

# to ensure webbpsf path work
import os
from pathlib import Path
os.environ['WEBBPSF_PATH'] = os.path.join(Path(__file__).parent.parent, 'webbpsf-data')

# reference: https://webbpsf.readthedocs.io/en/latest/usage.html#using-api

nc = webbpsf.NIRCam()
# nc.filter = 'F444W'
nc.filter = 'F356W'
nc.image_mask='MASK430R'
nc.pupil_mask='CIRCLYOT'
# star is without offset 
pixels = 100
r = 1
theta = 90
nc.options['source_offset_r'] = r
nc.options['source_offset_theta'] = theta
# returns an astropy.io.fits.HDUlist containing PSF and header
# psf = nc.calc_psf(outfile='./outs/psf_test.fits')
psf = nc.calc_psf(outfile=f'./outs/psf_{nc.filter}_{pixels}_r{r}_theta{theta}.fits',
	fov_pixels=pixels,
	oversample=1,
	)
psf.info()


plt.imshow(psf[0].data)
# plt.imshow(psf[0].data, norm=LogNorm())
plt.colorbar()
plt.show()