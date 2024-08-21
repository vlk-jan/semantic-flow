from pathlib import Path
from typing import Dict, Union

import imageio
import numpy as np
import pandas as pd
import open3d as o3d


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

def animate_renders(render_dir: Union[Path, str]) -> None:
    """
    Animate the rendered images.

    Parameters
    ----------
    render_dir : Path | str
        Path to the rendered images.
    """
    render_dir = render_dir if isinstance(render_dir, Path) else Path(render_dir)

    with imageio.get_writer(render_dir.parent / (render_dir.stem + ".gif"), mode="I", loop=0, fps=10) as writer:
        for file in sorted(Path(render_dir).iterdir()):
            writer.append_data(imageio.imread(file))

def draw_pc(pc_xyzrgb: pd.DataFrame) -> None:
    """
    Plot colored point cloud.

    Parameters
    ----------
    pc_xyzrgb : pd.DataFrame
        Point cloud to plot.
    """
    pc_xyzrgb = np.array(pc_xyzrgb)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
    if pc_xyzrgb.shape[1] == 3:
        o3d.visualization.draw_geometries([pc])
        return 0
    if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
        pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
    else:
        pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
    o3d.visualization.draw_geometries([pc])
