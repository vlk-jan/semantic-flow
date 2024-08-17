import os
from pathlib import Path
from typing import Dict, List, Tuple
from time import sleep

import cv2
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import pyarrow.feather as feather


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

def get_camera_intrinsics(intrinsics: pd.Series) -> np.ndarray:
    """
    Construct camera intrinsics matrix.

    Parameters
    ----------
    intrinsics : pd.Series
        Series containing the intrinsics parameters for camera calibration.

    Returns
    -------
    np.ndarray
        A 3x3 element matrix representing the camera intrinsics matrix.
    """
    ret = np.zeros((3, 3))

    ret[0, 0] = intrinsics["fx_px"]
    ret[0, 2] = intrinsics["cx_px"]
    ret[1, 1] = intrinsics["fy_px"]
    ret[1, 2] = intrinsics["cy_px"]
    ret[2, 2] = 1

    return ret

def get_sensor_extrinsics(extrinsics: pd.Series) -> np.ndarray:
    """
    Calculate the transformation from sensor to ego vehicle.

    Parameters
    ----------
    extrinsics : pd.Series
        Series containing the extrinsics parameters for sensor calibration.

    Returns
    -------
    np.ndarray
        A 4x4 element matrix representing the sensor extrinsics matrix.
    """
    ret = np.zeros((4, 4))

    view_T_camera = np.array(
            [
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ]
        )

    ret[:3, :3] = q_2_rot(extrinsics["qx"], extrinsics["qy"], extrinsics["qz"], extrinsics["qw"]) @ view_T_camera
    ret[:3, 3] = extrinsics[["tx_m", "ty_m", "tz_m"]].values
    ret[3, 3] = 1

    return ret

def calculate_calibration(camera_intrinsics: Dict[str, np.ndarray], sensor_extrinsics: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate the calibration matrix.

    Parameters
    ----------
    camera_intrinsics : dict
        Dictionary containing the camera intrinsics matrices.
    sensor_extrinsics : dict
        Dictionary containing the sensor extrinsics matrices.

    Returns
    -------
    dict
        A dictionary containing the calibration matrices for each sensor.
    """
    calib = dict()

    for sensor, intrinsics in camera_intrinsics.items():
        calib[sensor] = np.dot(intrinsics, sensor_extrinsics[sensor])
        calib[sensor] = np.vstack((calib[sensor], np.array([0, 0, 0, 1])))

    for sensor, extrinsics in sensor_extrinsics.items():
        if sensor not in calib:
            calib[sensor] = extrinsics

    return calib

def load_calib(calib_path: str, debug: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load calibration files and compute transformation matrices.
    
    Parameters
    ----------
    calib_path : str
        Path to the calibration files.
    debug : bool
        Flag for printing debug information.

    Returns
    -------
    tuple
        A tuple containing the sensor extrinsics dictionary
        and dictionary of calibration matrices for each sensor.
    """
    intrinsics = feather.read_feather(calib_path + "intrinsics.feather")
    extrinsics = feather.read_feather(calib_path + "egovehicle_SE3_sensor.feather")

    camera_intrinsics = dict()
    for i in range(intrinsics.shape[0]):
        camera_intrinsics[intrinsics.iloc[i]["sensor_name"]] = get_camera_intrinsics(intrinsics.iloc[i])

    if debug:
        print("Camera intrinsics:\n-----------------")
        print_dict(camera_intrinsics)

    sensor_extrinsics = dict()
    for i in range(extrinsics.shape[0]):
        sensor_extrinsics[extrinsics.iloc[i]["sensor_name"]] = get_sensor_extrinsics(extrinsics.iloc[i])

    if debug:
        print("Sensor extrinsics:\n-----------------")
        print_dict(sensor_extrinsics)

    # calib = calculate_calibration(camera_intrinsics, sensor_extrinsics)

    # if debug:
    #     print("Calibration:\n------------")
    #     print_dict(calib)

    return sensor_extrinsics, camera_intrinsics

def load_imgs(sensor_path: str) -> Dict[str, List[Path]]:
    """
    Load images paths from the sensor path.

    Parameters
    ----------
    sensor_path : str
        Path to the sensor directory.

    Returns
    -------
    dict
        A dictionary containing the images paths for each sensor.
    """
    sensor_path += "cameras/"
    imgs = dict()

    for sensor in os.listdir(sensor_path):
        imgs[sensor] = sorted(Path(sensor_path + sensor).glob("*.jpg"))

    return imgs

def show_imgs(imgs: Dict[str, List[Path]], idx: int = 0) -> None:
    """
    Display images on the idx from the dictionary.

    Parameters
    ----------
    imgs : dict
        Dictionary containing the images paths.
    idx : int
        Index of the image to be displayed.
    """
    plt.figure(figsize=(20, 20))

    plt.subplot(2, 4, 1)
    plt.imshow(plt.imread(imgs["ring_front_left"][idx]))
    plt.title("ring_front_left")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(plt.imread(imgs["ring_front_right"][idx]))
    plt.title("ring_front_right")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(plt.imread(imgs["ring_side_left"][idx]))
    plt.title("ring_side_left")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(plt.imread(imgs["ring_side_right"][idx]))
    plt.title("ring_side_right")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(plt.imread(imgs["ring_rear_left"][idx]))
    plt.title("ring_rear_left")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(plt.imread(imgs["ring_rear_right"][idx]))
    plt.title("ring_rear_right")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(plt.imread(imgs["ring_front_center"][0]))
    plt.title("ring_front_center")
    plt.axis("off")

    plt.show()

def load_lidars(sensor_path: str) -> Dict[int, pd.DataFrame]:
    """
    Load point clouds paths from the sensor path.

    Parameters
    ----------
    sensor_path : str
        Path to the sensor directory.

    Returns
    -------
    dict
        A dictionary containing the point clouds paths.
    """
    pcd = dict()
    sensor_path += "lidar/"

    for file in sorted(os.listdir(sensor_path)):
        pcd[int(file[:-8])] = feather.read_feather(sensor_path + file)

    return pcd

def show_pcd(pcd: Dict[int, pd.DataFrame], idx: int = 0) -> None:
    """
    Display point clouds on the idx from the dictionary.

    Parameters
    ----------
    pcd : dict
        Dictionary containing the point clouds paths.
    idx : int
        Index of the point cloud to be displayed.
    """
    show_pcd = o3d.geometry.PointCloud()
    key = list(pcd.keys())[idx]
    show_pcd.points = o3d.utility.Vector3dVector(pcd[key][["x", "y", "z"]].values)

    o3d.visualization.draw_geometries([show_pcd])

def show_img(name: str, img: np.ndarray) -> None:
    """
    Show the image in a named window.

    Parameters
    ----------
    name: str
        Name of window.
    img: np.ndarray
        The image to be displayed.
    """
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)
    cv2.waitKey(0)

def get_timestamps(imgs: Dict[int, List[Path]], pcd: Dict[str, pd.DataFrame]) -> Dict[int, Dict[str, int]]:
    lidar_to_img_map = dict()
    timestamp_to_path = dict()

    for sensor in imgs.keys():
        img_timestamp_to_img_file_map = {int(e.stem): e for e in imgs[sensor]}
        timestamp_to_path[sensor] = img_timestamp_to_img_file_map

        lidar_to_img_map[sensor] = {
                lidar_timestamp: min(
                    img_timestamp_to_img_file_map.keys(),
                    key=lambda img_timestamp: abs(img_timestamp - lidar_timestamp),
                )
                for lidar_timestamp in pcd.keys()
            }

    ret = dict()
    for lidar_timestamp in pcd.keys():
        ret[lidar_timestamp] = {
                sensor: lidar_to_img_map[sensor][lidar_timestamp]
                for sensor in imgs.keys()
            }

    return ret, timestamp_to_path

def get_imgs_for_timestamp(
        timestamp_to_path: Dict[int, List[Path]], lidar_to_img_map: Dict[int, Dict[str, int]], timestamp: int, sensors: List[str]
    ) -> Dict[str, np.ndarray]:
    timestamped_imgs = dict()
    for sensor in sensors:
        timestamped_imgs[sensor] = cv2.imread(str(timestamp_to_path[sensor][lidar_to_img_map[timestamp][sensor]]))

    return timestamped_imgs
