import tqdm
import pickle
from pathlib import Path
from typing import Optional, Union, List, Dict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.feather as feather

from dataset import AVScene
from utils import draw_pc


class LidarToImg:
    def __init__(self, scene: Union[AVScene, Path, str], sensors: Optional[List[str]] = None):
        if isinstance(scene, (Path, str)):
            with open(scene, "rb") as f:
                scene = pickle.load(f)
        self.scene = scene
        self.sensors = sensors if sensors is not None else [sensor for sensor in self.scene.camera_intrinsics.keys() if "stereo" not in sensor]

    def run(self, save_dir: Union[Path, dir], visualize: bool = False) -> None:
        """
        Create and save the colored point clouds.

        Parameters
        ----------
        save_dir : Path
            The directory to save the colored point clouds.
        visualize : bool
            Flag to visualize the colored point clouds.
        """
        print("Calculating projections into images")
        self.project()

        print("Coloring point clouds")
        for timestamp in tqdm.tqdm(self.scene.pcd_timestamps):
            pcd = self.color_pcd(timestamp, save_dir)
            if visualize:
                draw_pc(pcd)

            save_dir = save_dir if isinstance(save_dir, Path) else Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            feather.write_feather(pcd, save_dir / f"{timestamp}.feather")

    def color_pcd(self, timestamp: int, save_dir: Union[Path, str]) -> pd.DataFrame:
        """
        Color the point cloud based on the projection to the image.

        Parameters
        ----------
        timestamp : int
            The timestamp of the point cloud.
        save_dir : Path
            The directory to save the colored point cloud.

        Returns
        -------
        pd.DataFrame
            The colored point cloud.
        """
        imgs = self.scene.get_imgs_for_timestamp(timestamp, self.sensors)
        pcd = feather.read_feather(self.scene.pcd[self.scene.pcd_timestamps.index(timestamp)])
        full_pcd = np.c_[pcd["x"], pcd["y"], pcd["z"], np.zeros(pcd.shape[0]), np.zeros(pcd.shape[0]), np.zeros(pcd.shape[0])]

        projected_points = self.projected_points[timestamp]

        for sensor in self.sensors:
            img = imgs[sensor]
            projected_points_2d = projected_points[sensor]

            mask = (
                (projected_points_2d[:, 0] > 0)
                & (projected_points_2d[:, 1] > 0)
                & (projected_points_2d[:, 0] < imgs[sensor].shape[1])
                & (projected_points_2d[:, 1] < imgs[sensor].shape[0])
            )
            indexes = projected_points_2d[mask]
            full_pcd[mask, 3:] = img[indexes[:, 1], indexes[:, 0]] / 255

        return pd.DataFrame(full_pcd, columns=["x", "y", "z", "r", "g", "b"])

    def project(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Project the LiDAR point cloud to the image for all pointcloud samples.

        Returns
        -------
        dict
            A dictionary containing the projected points for each camera for each timestamp.
        """
        self.projected_points = dict()
        for timestamp in tqdm.tqdm(self.scene.pcd_timestamps):
            self.projected_points[timestamp] = self._project_lidar_to_img(timestamp)

        return self.projected_points

    def _project_lidar_to_img(self, timestamp: int) -> Dict[str, np.ndarray]:
        """
        Project the LiDAR point cloud to the image.

        Parameters
        ----------
        timestamp : int
            The timestamp of point cloud to project.

        Returns
        -------
        dict
            A dictionary containing the projected points for each camera.
        """
        projected_points = dict()

        pcd = feather.read_feather(self.scene.pcd[self.scene.pcd_timestamps.index(timestamp)])
        full_pcd = np.c_[pcd["x"], pcd["y"], pcd["z"], np.ones(pcd.shape[0])]

        for sensor in self.sensors:
            pc_into_img = np.linalg.pinv(self.scene.sensor_extrinsics[sensor]) @ full_pcd.T
            pc_into_img = pc_into_img[:3].T
            pc_into_img[pc_into_img[:, 0] < 0] = np.nan

            # Camera frame to pixels
            pixel_points_3d = pc_into_img @ self.scene.camera_T_view @ self.scene.camera_intrinsics[sensor].T
            pixel_points_2d = pixel_points_3d[:, :2] / pixel_points_3d[:, 2:]
            pixel_points_2d[np.isnan(pixel_points_2d)] = 0
            pixel_points_2d = np.round(pixel_points_2d).astype(int)

            projected_points[sensor] = pixel_points_2d

        return projected_points

    def visualize(self, timestamp: int, save_path: Optional[Union[Path, str]] = None, reduction_scale: int = 4) -> None:
        """
        Visualize the projected point cloud on the image.

        Parameters
        ----------
        timestamp : int
            The timestamp of the point cloud.
        save_path : Path
            The path to save the visualization. If None, the visualization is displayed.
        reduction_scale : int
            The reduction scale for the image.
        """
        imgs = self.scene.get_imgs_for_timestamp(timestamp, self.sensors)
        pcd = feather.read_feather(self.scene.pcd[self.scene.pcd_timestamps.index(timestamp)])
        full_pcd = np.c_[pcd["x"], pcd["y"], pcd["z"], np.ones(pcd.shape[0])]

        projected_imgs = dict()

        # Project colored point cloud onto images
        for sensor in self.sensors:
            pc_into_img = np.linalg.pinv(self.scene.sensor_extrinsics[sensor]) @ full_pcd.T
            pc_into_img = pc_into_img[:3].T
            pc_into_img = pc_into_img[pc_into_img[:, 0] >= 0]

            # Camera frame to pixels
            pixel_points_3d = pc_into_img @ self.scene.camera_T_view @ self.scene.camera_intrinsics[sensor].T
            pixel_points_2d = np.round(pixel_points_3d[:, :2] / pixel_points_3d[:, 2:]).astype(int)

            # Filter out points that are outside the image
            mask = (
                (pixel_points_2d[:, 0] > 0)
                & (pixel_points_2d[:, 1] > 0)
                & (pixel_points_2d[:, 0] < imgs[sensor].shape[1])
                & (pixel_points_2d[:, 1] < imgs[sensor].shape[0])
            )
            pixel_points_2d = pixel_points_2d[mask] // reduction_scale
            colors = self._color_pcd_distance(pc_into_img[:, 0], max_distance=30)[mask]

            # Rescale the image and project the point cloud
            projected_rgb_image = self._rescale_img(imgs[sensor].copy(), reduction_scale)
            projected_rgb_image[pixel_points_2d[:, 1], pixel_points_2d[:, 0], :] = colors * 255

            projected_imgs[sensor] = projected_rgb_image / 255

        plt.figure(figsize=(20, 20))
        for plot_idx, rgb_image in enumerate(projected_imgs.values()):
            plt.subplot(1, len(projected_imgs), plot_idx + 1)
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
            save_path = save_path if isinstance(save_path, Path) else Path(save_path)
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close()
        else:
            plt.show()

    def _color_pcd_distance(self, x: np.ndarray, max_distance: float = 10, cmap: str = "viridis") -> np.ndarray:
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

    def _rescale_img(self, img: np.ndarray, reduction_factor: float) -> np.ndarray:
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
        return new_img

    def generate(self, save_dir: Union[Path, str]) -> None:
        """
        Generate the visualizations for all timestamps.

        Parameters
        ----------
        save_dir : Path
            The directory to save the visualizations.
        """
        save_dir = save_dir if isinstance(save_dir, Path) else Path(save_dir)
        save_dir = save_dir / self.scene.root_dir.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        for timestamp in tqdm.tqdm(self.scene.pcd_timestamps):
            self.visualize(timestamp, save_path=save_dir / f"{timestamp}.png")


if __name__ == "__main__":
    np.printoptions(precision=3, suppress=True)

    import argparse

    parser = argparse.ArgumentParser(description="Lidar to image class debugging")
    parser.add_argument("--scene_dir", type=str, default="./datasets/ff8e7fdb-1073-3592-ba5e-8111bc3ce48b.pkl", help="Path to the scene pickle file")
    parser.add_argument("--save_dir", type=str, default="./colored_pcds", help="Path to save the colored point clouds")
    parser.add_argument("--vis_3d", action="store_true", help="Flag to visualize the colored point cloud")
    parser.add_argument("--vis_img", action="store_true", help="Flag to visualize the projected point cloud")
    parser.add_argument("--vis_dir", type=str, help="Path to save the visualizations")

    args = parser.parse_args()

    sensors = ["ring_side_left", "ring_front_left", "ring_front_center", "ring_front_right", "ring_side_right"]

    lidar_to_img = LidarToImg(Path(args.scene_dir), None)
    lidar_to_img.run(Path(args.save_dir), args.vis_3d)

    if args.vis_img:
        lidar_to_img.visualize(0, Path(args.vis_dir))
