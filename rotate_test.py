from util import get_nircam_with_options, get_test_psf, get_defaults
from scipy import ndimage, misc
import matplotlib.pyplot as plt

roll = get_defaults().getfloat("roll")

# img = ndimage.rotate(misc.ascent(), roll, reshape=True) 
# plt.imshow(img)
# plt.show()

psf = get_test_psf(get_nircam_with_options())
print(psf.shape)
psf = ndimage.rotate(psf, roll, reshape=True)
print(psf.shape)
plt.imshow(psf)
plt.show()