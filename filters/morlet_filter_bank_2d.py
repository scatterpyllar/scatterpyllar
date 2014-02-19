"""Creates a filter bank with all necessary information"""
import numpy as np
from morlet_2d_noDC import morlet_2d_noDC


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
    
