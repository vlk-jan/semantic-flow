from typing import Dict

import numpy as np


def q_2_rot(qx, qy, qz, qw) -> np.ndarray:
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
    Taken from: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
 
    Parameters
    ----------
    qx, qy, qz, qw : float
        The x, y, z, and w components of the quaternion.
 
    Returns
    -------
    np.ndarray
        A 3x3 element matrix representing the full 3D rotation matrix. 
        This rotation matrix converts a point in the local reference 
        frame to a point in the global reference frame.
    """
    # First row of the rotation matrix
    r00 = 2 * (qw * qw + qx * qx) - 1
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qx * qz + qw * qy)
     
    # Second row of the rotation matrix
    r10 = 2 * (qx * qy + qw * qz)
    r11 = 2 * (qw * qw + qy * qy) - 1
    r12 = 2 * (qy * qz - qw * qx)
     
    # Third row of the rotation matrix
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 2 * (qw * qw + qz * qz) - 1
     
    # 3x3 rotation matrix
    ret = np.array([[r00, r01, r02],
                    [r10, r11, r12],
                    [r20, r21, r22]])
                            
    return ret

def print_dict(d: Dict[any, any]) -> None:
    """
    Print a dictionary in a human-readable format.

    Parameters
    ----------
    d : dict
        Dictionary to be printed.
    """
    for key, value in d.items():
        print(key)
        print(value)
    print()
