"""Creates a filter bank with all necessary information"""
import numpy as np
from morlet_2d_noDC import morlet_2d_noDC
from gabor_2d import gabor_2d


def morlet_filter_bank_2d(shape, Q=1, L=8, J=4, 
                          sigma_phi=.8,
                          sigma_psi=.8,
                          xi_psi=None,
                          slant_psi=None,
                          min_margin=None):
    """Creates a multiscale bank of filters

    Creates and indexes filters at several scales and orientations

    Parameters
    ----------

    shape : {tuple, list, ndarray}
        shape=(2,)
        Tuple indicating the shape of the filters to be generated

    Q : {integer}
        Number of scales per octave (constant-Q filter bank)

    J : {integer}
        Total number of scales

    L : {integer}
        Number of orientations

    sigma_phi : {float}
        standard deviation of low-pass mother wavelet

    sigma_psi : {float}
        standard deviation of the envelope of the high-pass psi_0

    xi_psi : {float}
        frequency peak of the band-pass mother wavelet

    slant_psi : {float}
        ratio between axes of elliptic envelope. Smaller means more
        orientation selective

    min_margin : {integer}
        Padding for convolution
    """

    # non-independent default values
    if xi_psi is None:
        xi_psi = .5 * np.pi * 2 ** (-1. / Q)
    if slant_psi is None:
        slant_psi = 4. / L
    if min_margin is None:
        min_margin = sigma_phi * 2 ** (float(J) / Q)

    max_resolution = int(J) / int(Q)

    # potentially do some padding here
    filter_shape = shape

    max_scale = 2 ** (float(J - 1) / Q - max_resolution)

    low_pass_spatial = np.real(gabor_2d(filter_shape, sigma_phi * max_scale,
                                0., 0., 1.))
    little_wood_paley = np.abs(np.fft.fft2(low_pass_spatial)) ** 2

    filters = dict(phi=low_pass_spatial, psi=dict(fil_list=[]),
                   j=list(), l=list(), J=J, L=L, Q=Q)

    angles = np.arange(L) * np.pi / L
    for j in range(J):
        filters['psi'][j] = dict()
        for l, angle in enumerate(angles):
            scale = 2 ** (float(j) / Q - max_resolution)

            band_pass_filter = morlet_2d_noDC(filter_shape,
                                              sigma_psi * scale,
                                              xi_psi / scale,
                                              angle,
                                              slant_psi)
            filters['psi'][j][l] = band_pass_filter
            little_wood_paley += np.abs(np.fft.fft2(band_pass_filter)) ** 2
            filters['j'].append(j)
            filters['l'].append(l)
            filters['psi']['fil_list'].append(band_pass_filter)

    little_wood_paley = np.fft.fftshift(little_wood_paley)
    lwp_max = little_wood_paley.max()

    for fil in filters['psi']['fil_list']:
        fil /= np.sqrt(lwp_max / 2)

    filters['littlewood_paley'] = little_wood_paley

    return filters


if __name__ == "__main__":
    Q, J, L = 1, 4, 8
    sigma_psi = 8
    sigma_phi = 8
    filters = morlet_filter_bank_2d([128, 128], Q=Q, J=J, L=L,
                                    sigma_psi=sigma_psi,
                                    sigma_phi=sigma_phi,
                                    xi_psi=np.pi / 16.)

    import pylab as pl
    pl.figure()
    pl.subplot(2 * J + 1, L, 1)
    pl.imshow(np.fft.fftshift(filters['phi']))
    pl.axis('off')
    pl.gray()
    pl.subplot(2 * J + 1, L, 2)
    pl.imshow(filters['littlewood_paley'])
    pl.axis('off')
    for j in range(J):
        for l in range(L):
            pl.subplot(2 * J + 1, L, 1 + (2 * j + 1) * L + l)
            pl.imshow(np.real(np.fft.fftshift(filters['psi'][j][l])))
            pl.axis('off')
            pl.subplot(2 * J + 1, L, 1 + (2 * j + 2) * L + l)
            pl.imshow(np.imag(np.fft.fftshift(filters['psi'][j][l])))
            pl.axis('off')

    pl.show()

