import numpy as np

from ..filters.utils import fft_convolve


def scattering_layer_0(imgs, filter_bank, subsample=0):

    low_pass = filter_bank['phi']

    low_passed_imgs = np.real(fft_convolve(imgs, low_pass))

    if subsample is not False:
        subsampling_factor = filter_bank['J'] + subsample
    else:
        subsampling_factor = 1

    low_passed_imgs = low_passed_imgs[..., ::subsampling_factor,
                                           ::subsampling_factor]

    all_filters = np.array(filter_bank['psi']['fil_list'])

    filter_moduli = np.abs(fft_convolve(imgs, all_filters))

    return low_passed_imgs, filter_moduli



    
