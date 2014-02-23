from filters.morlet_filter_bank_2d import morlet_filter_bank_2d
from core.scattering_simple import scattering_layer_0
import numpy as np
from scipy.misc import lena

if __name__ == "__main__":
    lena = lena() / 256.

    filters = morlet_filter_bank_2d(lena.shape, L=4, sigma_phi=16,
                                    sigma_psi=16)

    lenas = np.array([lena, lena[:, ::-1]])

    low, moduli = scattering_layer_0(lenas, filters)
