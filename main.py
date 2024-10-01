import tqdm
import pickle
import argparse
from pathlib import Path

import numpy as np

from dataset import AVScene
from lidar_to_img import LidarToImg
from segmentation import Segmentation

from utils import animate_renders


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Flow")

    parser.add_argument(
        "--root_dir",
        type=str,
        default="../../dataset/train",
        help="Path to the dataset",
    )
    parser.add_argument("--dataset_dir", type=str, help="Path to the pickled dataset")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./weights/",
        help="Path to MobileSAM checkpoint",
    )
    parser.add_argument(
        "--seg_dir", type=str, default="./seg", help="Path to save the segmented images"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./colored_pcds",
        help="Path to save the colored point clouds",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Flag to render the point cloud projections into images",
    )
    parser.add_argument(
        "--render_dir",
        type=str,
        default="./render",
        help="Path to save the rendered images",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Flag for extra debugging info",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    sensors = [
        "ring_rear_left",
        "ring_side_left",
        "ring_front_left",
        "ring_front_center",
        "ring_front_right",
        "ring_side_right",
        "ring_rear_right",
    ]

    assert Path(
        args.root_dir
    ).exists(), f"Path {Path(args.root_dir).absolute()} does not exist"

    # Iterate over the all scenes
    for scene_dir in tqdm.tqdm(sorted(Path(args.root_dir).iterdir())):
        if not scene_dir.is_dir():  # check for random files
            print(f"{scene_dir} is not a directory")
            continue

        # Load the dataset
        if (
            args.dataset_dir is not None
            and (Path(args.dataset_dir) / scene_dir.stem).exists()
        ):  # dataset already exists
            with open((args.dataset_dir / scene_dir), "rb") as fp:
                dataset = pickle.load(fp)
        else:  # create the dataset
            dataset = AVScene(scene_dir, debug=args.debug)
            if args.dataset_dir is not None:
                dataset.save(args.dataset_dir + scene_dir.stem + ".pkl")

        lidar_to_img = LidarToImg(dataset, sensors)

        if not args.render:
            # Segment the images
            assert Path(
                args.checkpoint_dir
            ).exists(), f"Path {Path(args.checkpoint_dir).absolute()} does not exist"

            segmentation = Segmentation(args.checkpoint_dir)
            segmentation.segment_scene(scene_dir, sensors)
            return
            for sensor in sensors:
                (Path(args.seg_dir) / scene_dir.stem / sensor).mkdir(
                    parents=True, exist_ok=True
                )
            for timestamp in tqdm.tqdm(
                dataset.pcd_timestamps
            ):  # iterate over all timestamps
                loaded_images = dataset.get_imgs_for_timestamp(timestamp, sensors)
                for sensor, img in tqdm.tqdm(
                    loaded_images.items()
                ):  # iterate over all sensors TODO: parallelize?
                    segmentation.segment_image(img)
                    seg_img = segmentation.create_segmented_image()
                    np.save(
                        Path(args.seg_dir)
                        / scene_dir.stem
                        / sensor
                        / f"{timestamp}.npy",
                        seg_img,
                    )

            del segmentation
            lidar_to_img.run(args.save_dir, args.seg_dir + scene_dir.stem)
        else:
            # Project the point cloud to images
            lidar_to_img.generate(args.render_dir)
            animate_renders(args.render_dir + scene_dir.stem)


if __name__ == "__main__":
    args = parse_args()
    main(args)
