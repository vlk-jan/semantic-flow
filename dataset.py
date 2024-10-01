import pickle
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from utils import print_dict, q_2_rot


class AVScene:
    def __init__(self, root_dir: Union[Path, str], debug: bool = False):
        self.root_dir = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.view_T_camera = np.array(
            [
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],
            ]
        )
        self.camera_T_view = np.array(
            [
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
            ]
        )
        self._load_calib(self.root_dir / "calibration", debug)
        self._load_imgs(self.root_dir / "sensors", debug)
        self._load_lidar(self.root_dir / "sensors", debug)
        self._get_timestamps()

    def _load_calib(self, calib_path: Path, str, debug: bool = False) -> None:
        """
        Load calibration files of intrinsics and extrinsics.

        Parameters
        ----------
        calib_path : str
            Path to the calibration files.
        debug : bool
            Flag for printing debug information.
        """
        intrinsics = feather.read_feather(calib_path / "intrinsics.feather")
        extrinsics = feather.read_feather(calib_path / "egovehicle_SE3_sensor.feather")

        self.camera_intrinsics = dict()
        for i in range(intrinsics.shape[0]):
            self.camera_intrinsics[intrinsics.iloc[i]["sensor_name"]] = (
                self._get_camera_intrinsics(intrinsics.iloc[i])
            )
        if debug:
            print("Camera intrinsics:\n-----------------")
            print_dict(self.camera_intrinsics)

        self.sensor_extrinsics = dict()
        for i in range(extrinsics.shape[0]):
            self.sensor_extrinsics[extrinsics.iloc[i]["sensor_name"]] = (
                self._get_sensor_extrinsics(extrinsics.iloc[i])
            )
        if debug:
            print("Sensor extrinsics:\n-----------------")
            print_dict(self.sensor_extrinsics)

    def _get_camera_intrinsics(self, intrinsics: pd.Series) -> np.ndarray:
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

    def _get_sensor_extrinsics(self, extrinsics: pd.Series) -> np.ndarray:
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

        ret[:3, :3] = (
            q_2_rot(
                extrinsics["qx"], extrinsics["qy"], extrinsics["qz"], extrinsics["qw"]
            )
            @ self.view_T_camera
        )
        ret[:3, 3] = extrinsics[["tx_m", "ty_m", "tz_m"]].values
        ret[3, 3] = 1

        return ret

    def _load_imgs(self, sensor_path: Path, str, debug: bool = False) -> None:
        """
        Load image paths from the sensor path.

        Parameters
        ----------
        sensor_path : Path
            Path to the sensor directory.
        debug : bool
            Flag for printing debug information.
        """
        self.raw_imgs = dict()

        for sensor in (sensor_path / "cameras").iterdir():
            self.raw_imgs[sensor.stem] = sorted(
                Path(sensor_path / "cameras" / sensor.stem).glob("*.jpg")
            )

        if debug:
            print("Raw images:\n-----------")
            print_dict(self.raw_imgs)

    def _load_lidar(self, sensor_path: Path, debug: bool = False) -> None:
        """
        Load lidar data from the sensor path.

        Parameters
        ----------
        sensor_path : Path
            Path to the sensor directory.
        debug : bool
            Flag for printing debug information.
        """
        self.pcd = sorted(Path(sensor_path / "lidar").glob("*.feather"))

        if debug:
            print("Lidar data:\n-----------")
            print(self.pcd)

    def _get_timestamps(self) -> None:
        """
        Create a mapping between lidar and image timestamps.
        """
        lidar_to_img_map_sensor = dict()
        self.timestamp_to_path = dict()

        self.pcd_timestamps = [int(e.stem) for e in self.pcd]

        for sensor in self.raw_imgs.keys():
            img_timestamp_to_img_file_map = {
                int(e.stem): e for e in self.raw_imgs[sensor]
            }
            self.timestamp_to_path[sensor] = img_timestamp_to_img_file_map

            lidar_to_img_map_sensor[sensor] = {
                lidar_timestamp: min(
                    img_timestamp_to_img_file_map.keys(),
                    key=lambda img_timestamp: abs(img_timestamp - lidar_timestamp),
                )
                for lidar_timestamp in self.pcd_timestamps
            }

        self.lidar_to_img_map = dict()
        for lidar_timestamp in self.pcd_timestamps:
            self.lidar_to_img_map[lidar_timestamp] = {
                sensor: lidar_to_img_map_sensor[sensor][lidar_timestamp]
                for sensor in self.raw_imgs.keys()
            }

    def get_imgs_for_timestamp(
        self, timestamp: int, sensors: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Get images for a specific timestamp.

        Parameters
        ----------
        timestamp : int
            Timestamp to get the images.
        sensors : list
            List of sensors to get the images from.

        Returns
        -------
        dict
            A dictionary containing the images for each sensor.
        """
        return {
            sensor: cv2.cvtColor(
                cv2.imread(
                    str(
                        self.timestamp_to_path[sensor][
                            self.lidar_to_img_map[timestamp][sensor]
                        ]
                    )
                ),
                cv2.COLOR_BGR2RGB,
            )
            for sensor in sensors
        }

    def save(self, save_path: Union[Path, str]) -> None:
        """
        Save the dataset to a pickle file.

        Parameters
        ----------
        save_path : Path
            Path to save the dataset.
        """
        save_path = save_path if isinstance(save_path, Path) else Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)


if __name__ == "__main__":
    np.printoptions(precision=3, suppress=True)

    import argparse

    parser = argparse.ArgumentParser(description="Dataset class creation")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="../../dataset/train/ff8e7fdb-1073-3592-ba5e-8111bc3ce48b",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./datasets", help="Path to save the dataset"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Flag for extra debugging info"
    )

    args = parser.parse_args()

    dataset = AVScene(args.root_dir, debug=args.debug)
    dataset.save(Path(args.save_dir) / f"{Path(args.root_dir).stem}.pkl")
