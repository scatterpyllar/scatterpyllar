"""Implements some generally useful utilities"""

import numpy as np


def rotation_matrix_2d(angle):
    """Returns a rotation matrix of the given angle


    Parameters
    ----------

    angle : {float}
        Angle in radians for counterclockwise rotation
"""

    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])

    return rot

