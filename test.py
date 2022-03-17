import matplotlib.pyplot as plt
import webbpsf
# to ensure webbpsf path work
import os
from pathlib import Path
os.environ['WEBBPSF_PATH'] = os.path.join(Path(__file__).parent.parent, 'webbpsf-data')


pixelscale = 0.063
fov_arcsec = 10.017
#fov_arcsec = 9
# oversample = 4
oversample = 1
wavelength = 4.441e-6
pupil_diameter = 6.559 # (in meter) As used in WebbPSF
#6.603464
# webbPSF parameters
filt = 'F444W'
# Simulation webbpsf image
########################
nc = webbpsf.NIRCam()
#nc.filter=filt
# nc.pupilopd = opd
# nc.pupil = transmission
nc.jitter = None
nc.include_si_wfe = False
nc.image_mask='MASK430R'
nc.pupil_mask='CIRCLYOT'
nc.pixelscale=pixelscale
hdu_list = nc.calc_psf(fov_arcsec=fov_arcsec,oversample=oversample,monochromatic=wavelength,add_distortion=False)
psf_webbpsf = hdu_list[0].data

plt.imshow(psf_webbpsf)
plt.colorbar()
plt.show()

# webbpsf.display_psf(hdu_list)
# plt.savefig('outs/test.png')