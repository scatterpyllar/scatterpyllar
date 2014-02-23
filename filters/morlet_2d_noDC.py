"""Makes a Morlet Wavelet by removing the DC offset from Gabor wavelet"""
from gabor_2d import gabor_2d
import numpy as np


def morlet_2d_noDC(shape, sigma, xi, theta, slant=None, offset=None):


    print "Morlet called with\n %s" % "\n".join(map(repr, [shape,
sigma, xi, theta, slant]))
    gabor = gabor_2d(shape, sigma, xi, theta, slant, offset)
    envelope = np.abs(gabor)
    K = gabor.sum() / envelope.sum()

    centered = gabor - K * envelope

    return centered


if __name__ == "__main__":
    
    mor = morlet_2d_noDC([512, 512], 12, np.pi / 16, np.pi / 6, 1. / 4.)

    import pylab as pl
    pl.figure()
    pl.subplot(1, 2, 1)
    pl.imshow(np.fft.fftshift(np.real(mor)))
    pl.axis('off')
    pl.gray()
    pl.subplot(1, 2, 2)
    pl.imshow(np.fft.fftshift(np.imag(mor)))
    pl.axis('off')

    
