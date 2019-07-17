##################
#Fourier Analysis::
# FA is effectively the Decomposition of nd signal (Image)
# into sinusoid functions. The primay objective is to translate
# a signal from the time domain to the frequency one. This will
# reveal the periodic nature of the image. 
###################


import numpy as np
from scipy import misc
import cmath
import matplotlib.pyplot as plt
from skimage import data
# load data
pic1 = data.clock()
pic2 = data.camera()
# resize to (300, 300)
pic1 = misc.imresize(pic1, (300, 300))
pic2 = misc.imresize(pic2, (300, 300))
# compute Fourier transforms
pic1_fft = np.fft.rfft2(pic1)
pic2_fft = np.fft.rfft2(pic2)
# separate into magnitude and phase
pic1_mag = np.absolute(pic1_fft)
pic1_ph = np.angle(pic1_fft)
pic2_mag = np.absolute(pic2_fft)
pic2_ph = np.angle(pic2_fft)
# swap phases and reconstruct
cexp = np.vectorize(cmath.exp)
pic1_reconstructed = np.fft.irfft2(np.multiply(pic1_mag,
 cexp(1j * pic2_ph)))
pic2_reconstructed = np.fft.irfft2(np.multiply(pic2_mag,
 cexp(1j * pic1_ph)))
# display results
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8,5))
plt.gray()
ax[0, 0].imshow(pic1) 

ax[0, 0].axis('off')
ax[0, 0].set_title('Pic1, original')
ax[0, 1].imshow(np.fft.fftshift(np.log(pic1_mag)))
ax[0, 1].axis('off')
ax[0, 1].set_title('Pic1, magnitude')
4
ax[0, 2].imshow(np.fft.fftshift(pic1_ph))
ax[0, 2].axis('off')
ax[0, 2].set_title('Pic1, phase')
ax[0, 3].imshow(pic1_reconstructed)
ax[0, 3].axis('off')
ax[0, 3].set_title('Pic1, reconstructed')
ax[1, 0].imshow(pic2)
ax[1, 0].axis('off')
ax[1, 0].set_title('Pic2, original')
ax[1, 1].imshow(np.fft.fftshift(np.log(pic2_mag)))
ax[1, 1].axis('off')
ax[1, 1].set_title('Pic2, magnitude')
ax[1, 2].imshow(np.fft.fftshift(pic2_ph))
ax[1, 2].axis('off')
ax[1, 2].set_title('Pic2, phase')
ax[1, 3].imshow(pic2_reconstructed)
ax[1, 3].axis('off')
ax[1, 3].set_title('Pic2, reconstructed')
plt.show()