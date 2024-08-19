import tqdm
import pickle
import argparse

from dataset import AVScene
from lidar_to_img import LidarToImg
from segmentation import Segmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Flow")

    parser.add_argument("--root_dir", type=str, default="../../dataset/train", help="Path to the dataset")
    parser.add_argument("--dataset_dir", type=str, help="Path to the pickled dataset")
    parser.add_argument("--checkpoint_path", type=str, default="./weights/", help="Path to MobileSAM checkpoint")
    parser.add_argument("--debug", action="store_true", default=False, help="Flag for extra debugging info")

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    sensors = ["ring_side_left", "ring_front_left", "ring_front_center", "ring_front_right", "ring_side_right"]
    # Load the dataset
    if args.dataset_dir:
        with open(args.dataset_dir, "rb") as fp:
            dataset = pickle.load(fp)
    else:
        dataset = AVScene(args.root_dir, debug=args.debug)
        if args.dataset_dir:
            dataset.save(args.dataset_dir)
    # Segment the images
    segmentation = Segmentation(args.checkpoint_path)
    for timestamp in tqdm.tqdm(dataset.pcd_timestamps):
        loaded_images = dataset.get_imgs_for_timestamp(timestamp, sensors)
        for sensor, img in tqdm.tqdm(loaded_images.items()):
            segmentation.segment_image(img)
            segmentation.create_segmented_image()
            segmentation.plot_image(img)


if __name__ == "__main__":
    args = parse_args()
    main(args)
