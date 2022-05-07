from util import *
from scipy import ndimage, misc
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np

roll = get_defaults().getfloat("roll")

# img = ndimage.rotate(misc.ascent(), roll, reshape=True) 
# plt.imshow(img)
# plt.show()
nc = get_nircam_with_options()
nc.image_mask = None
nc.pupil_mask = None
psf = get_test_psf(nc)
print(psf.shape)
psf = ndimage.rotate(psf, roll, reshape=True)

blurred = gaussian_filter(psf, sigma=1)
new_center = (60,70)
recentered = recenter(blurred, new_center)
print(recentered.shape)

plt.subplot(1,3,1)
plt.imshow(psf)
plt.subplot(1,3,2)
plt.imshow(blurred)
plt.plot(new_center[0], new_center[1], 'yx')
plt.subplot(1,3,3)
plt.imshow(recentered)
plt.show()



# >>> fig = plt.figure()
# >>> plt.gray()  # show the filtered result in grayscale
# >>> ax1 = fig.add_subplot(121)  # left side
# >>> ax2 = fig.add_subplot(122)  # right side
# >>> ascent = misc.ascent()
# >>> result = gaussian_filter(ascent, sigma=5)
# >>> ax1.imshow(ascent)
# >>> ax2.imshow(result)