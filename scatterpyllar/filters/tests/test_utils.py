import numpy as np
from numpy.testing import assert_array_equal
from ..utils import pad


def test_pad_1D():
    a = np.arange(1, 5)
    b = np.zeros(8)
    b[1:5] = a

    a_padded = pad(a, 1, 3)

    assert_array_equal(a_padded, b)


def test_crop_1D():
    a = np.arange(2, 12)
    a_cropped = pad(a, -2, -4)
    b = np.arange(4, 8)

    assert_array_equal(a_cropped, b)


def test_pad_2D():
    a = np.arange(56).reshape(7, 8)
    b = np.zeros([10, 14])
    b[1:8, 5:13] = a

    a_padded = pad(a, [1, 5], [2, 1])

    assert_array_equal(b, a_padded)


def test_crop_2D():
    a = np.arange(56).reshape(7, 8)
    b = a[2:4, 1:6]

    a_cropped = pad(a, [-2, -1], [-3, -2])

    assert_array_equal(b, a_cropped)

