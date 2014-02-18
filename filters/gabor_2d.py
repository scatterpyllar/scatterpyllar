"""Generates a 2d Gabor filter of given orientation and frequency"""

import numpy as np
from utils import rotation_matrix_2d


def gabor_2d(shape, sigma0, xi, theta, slant=None, offset=None):
    """
    Returns a Gabor filter following the specifications

    This function creates a 2D complex Gabor filter. All parameters
    are taken to be in integer grid space


    Parameters
    ----------
    shape : {tuple, list}
        shape=(2,)
        Indicates the shape of the output array

    sigma_0 : {float}
        Indicates the standard deviation of the filter envelope along
        the oscillation axis

    slant : {float}
        Indicates the standard deviation of the filter envelope along
        the axis orthogonal to the oscillation. It is given relatively
        with respect to sigma_0 (sigma_orthog = slant * sigma_0)

    xi : {float}
        The oscillation wave number

    theta : {float}
        The oscillation wave orientation 
        (0 is a downward pointing wave vector)

    offset : {tuple, list, ndarray}
        shape=(2,)
        Possible offset for the filter from the origin. Defaults to 0


    See also
    --------
    Morlet wavelets
"""

    if slant is None:
        slant = 1.
    if offset is None:
        offset = np.zeros([2, 1, 1])
    else:
        offset = np.asanyarray(offset).reshape(2, 1, 1)

    g = np.mgrid[-shape[0] / 2:shape[0] / 2, -shape[1] / 2:shape[1] / 2]
    g -= offset

    rot = rotation_matrix_2d(theta)
    invrot = np.linalg.inv(rot)
    rot_g = invrot.dot(g.reshape(2, -1))
    precision_matrix = np.diag([1., slant ** 2]) / (sigma0 ** 2)
    mahalanobis = (rot_g * (precision_matrix.dot(rot_g))).sum(0)

    raw_gabor = np.exp(-mahalanobis / 2 + 1j * xi * rot_g[0])

    gabor = np.fft.fftshift(raw_gabor.reshape(shape)) / (
        2 * np.pi * sigma0 * sigma0 / slant)

    return gabor


if __name__ == "__main__":
    """Print create a random phase texture at 30 degree orientation"""

    gab = gabor_2d([512, 512], 12, np.pi / 16, np.pi / 6., 1. / 4.)

    fft_env = np.abs(np.fft.fft2(gab))
    rng = np.random.RandomState(42)
    phase = np.exp(1j * rng.rand(*fft_env.shape) * 2 * np.pi)
    texture = np.real(np.fft.ifft2(fft_env * phase ))

    import pylab as pl
    pl.figure()
    pl.subplot(1, 3, 1)
    pl.imshow(np.real(np.fft.fftshift(gab)))
    pl.axis("off")
    pl.gray()
    pl.subplot(1, 3, 2)
    pl.imshow(np.imag(np.fft.fftshift(gab)))
    pl.axis("off")

    pl.subplot(1, 3, 3)
    pl.imshow(texture)
    pl.axis("off")

    pl.show()



