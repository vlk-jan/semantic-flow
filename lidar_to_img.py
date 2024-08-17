#!/usr/bin/env python3

"""
Project a point cloud onto an image.
Created for the Argoverse2 sensor dataset.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_calib, load_imgs, show_imgs, show_img, load_lidars, show_pcd, get_timestamps, get_imgs_for_timestamp
# from segmentation import Segmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("calib_path", type=str, help="Path to the calibration files")
    parser.add_argument("sensor_path", type=str, help="Path to the sensor data")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="Enable debug mode")

    return parser.parse_args()

def color_pcd_distance(x: np.ndarray, max_distance: float = 10, cmap: str = "viridis") -> np.ndarray:
    """
    Color the point cloud based on the distance from the origin.

    Parameters
    ----------
    x : np.ndarray
        The x coordinate of the point cloud.
    max_distance : float
        The maximum distance.
    cmap : str
        The colormap to use.

    Returns
    -------
    np.ndarray
        The colored point cloud.
    """
    colors = x.copy()

    # Normalize to [0, 1]
    colors = colors / max_distance
    colors[colors > 1] = 1.0

    colormap = plt.get_cmap(cmap)
    colors = colormap(colors)[:, :3]
    return colors

def rescale_img(img: np.ndarray, reduction_factor: float) -> np.ndarray:
        """
        Rescale the image by a factor.

        Parameters
        ----------
        img : np.ndarray
            The image to be rescaled.
        reduction_factor : float
            The reduction factor.

        Returns
        -------
        np.ndarray
            The rescaled image.
        """
        new_shape = (
            int(np.ceil(img.shape[1] / reduction_factor)),
            int(np.ceil(img.shape[0] / reduction_factor)),
        )
        new_img = cv2.resize(img, new_shape).astype(np.float32)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)        
        return new_img

def project_lidar_to_img(
        pcd: pd.DataFrame,
        imgs: Dict[str, List[np.ndarray]],
        extrinsics: Dict[str, np.ndarray],
        intrinsics: Dict[str, np.ndarray],
        reduction_scale: int = 4,
        debug: bool = False
    ) -> Dict[str, np.ndarray]:
    """
    Project a point cloud onto an image.

    Parameters
    ----------
    pcd : pd.DataFrame
        Point cloud data.
    imgs : dict
        Dictionary containing the images.
    extrinsics : dict
        Dictionary containing the extrinsics.
    intrinsics : dict
        Dictionary containing the intrinsics.
    reduction_scale : int
        The rescale factor. Keep it as a power of 2.
    debug : bool
        Enable debug mode.

    Returns
    -------
    dict
        Dictionary containing images with the projected point cloud.
    """
    projected_imgs = dict()

    full_pcd = np.c_[pcd["x"], pcd["y"], pcd["z"], np.ones(pcd.shape[0])]
    camera_T_view = np.array(
            [
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
            ]
        )

    for sensor in imgs.keys():
        pc_into_img = np.linalg.pinv(extrinsics[sensor]) @ full_pcd.T
        pc_into_img = pc_into_img[:3].T
        pc_into_img = pc_into_img[pc_into_img[:, 0] >= 0]

        # Camera frame to pixels
        pixel_points_3d = pc_into_img @ camera_T_view @ intrinsics[sensor].T
        pixel_points_2d = np.round(pixel_points_3d[:, :2] / pixel_points_3d[:, 2:]).astype(int)

        # Filter out points that are outside the image
        mask = (
            (pixel_points_2d[:, 0] > 0)
            & (pixel_points_2d[:, 1] > 0)
            & (pixel_points_2d[:, 0] < imgs[sensor].shape[1])
            & (pixel_points_2d[:, 1] < imgs[sensor].shape[0])
        )
        pixel_points_2d = pixel_points_2d[mask] // reduction_scale
        colors = color_pcd_distance(pc_into_img[:, 0], max_distance=30)[mask]

        # Rescale the image and project the point cloud
        projected_rgb_image = rescale_img(imgs[sensor].copy(), reduction_scale)
        projected_rgb_image[pixel_points_2d[:, 1], pixel_points_2d[:, 0], :] = colors * 255

        projected_imgs[sensor] = projected_rgb_image / 255

        if debug:
            show_img(sensor, projected_imgs[sensor])
    
    return projected_imgs

def visualize(rgb_images: Dict[str, np.ndarray], save_path: Path = None) -> None:
    """
    Visualize point clouds projected onto images.

    Parameters
    ----------
    rgb_images : dict
        Dictionary containing the images.
    save_path : Path
        Path to save the image. If None, the image is displayed.
    """
    plt.figure(figsize=(20, 20))
    for plot_idx, rgb_image in enumerate(rgb_images.values()):
        plt.subplot(1, len(rgb_images), plot_idx + 1)
        plt.imshow(rgb_image)
        # Disable axis ticks
        plt.xticks([])
        plt.yticks([])
        # Set padding between subplots to 0.3
        plt.tight_layout(pad=1)
        # Get rid of white space
        plt.margins(0)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # Set the background to be black
    fig = plt.gcf()
    fig.set_facecolor("black")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.clf()
    else:
        plt.show()

def main(args: argparse.Namespace) -> None:
    np.set_printoptions(precision=3, suppress=True)

    sensors = ["ring_side_left", "ring_front_left", "ring_front_center", "ring_front_right", "ring_side_right"]

    # mobileSAM = Segmentation("/home/vlkjan6/Documents/diplomka/code/weights")

    extrinsics, intrinsics = load_calib(args.calib_path, args.debug)

    imgs = load_imgs(args.sensor_path)
    if args.debug:
        show_imgs(imgs)

    pcd = load_lidars(args.sensor_path)
    if args.debug:
        show_pcd(pcd)

    '''mobileSAM.load_image(imgs["ring_rear_right"][0])
    mobileSAM.segment_image()
    mobileSAM.plot_image()'''

    lidar_to_img_map, timestamp_to_path = get_timestamps(imgs, pcd)

    timestamp = sorted(pcd.keys())[0]
    timestamped_imgs = get_imgs_for_timestamp(timestamp_to_path, lidar_to_img_map, timestamp, sensors)
    projected_imgs = project_lidar_to_img(pcd[timestamp], timestamped_imgs, extrinsics, intrinsics, debug=args.debug)
    visualize(projected_imgs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
