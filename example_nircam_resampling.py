import matplotlib.pyplot as plt
import webbpsf
# to ensure webbpsf path work
import os
from pathlib import Path
os.environ['WEBBPSF_PATH'] = os.path.join(Path(__file__).parent.parent, 'webbpsf-data')

nc = webbpsf.NIRCam()
nc.filter='F444W'
nc.image_mask='MASK430R'
nc.pupil_mask='CIRCLYOT'
nc.options['source_offset_r'] = 0.20       # source is 200 mas from center of coronagraph
                                           # (note that this is MUCH larger than expected acq
                                           # offsets. This size displacement is just for show)
nc.options['source_offset_theta'] = 45     # at a position angle of 45 deg
psf = nc.calc_psf('coronagraphic.fits', oversample=4)   # create highly oversampled output image
psf.info()


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
webbpsf.display_psf('coronagraphic.fits', vmin=1e-10, vmax=1e-5,
    ext='OVERSAMP', title=f'NIRCam {nc.filter}+MASK430R, 4x oversampled', crosshairs=True)
plt.subplot(1,2,2)
webbpsf.display_psf('coronagraphic.fits', vmin=1e-10, vmax=1e-5,
    ext='DET_SAMP', title=f'NIRCam {nc.filter}+MASK430R, detector oversampled', crosshairs=True)

plt.savefig(f'outs/example_nircam_coron_resampling_{nc.filter}.png')