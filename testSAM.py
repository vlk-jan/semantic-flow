import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator


def get_anns(anns: list) -> np.ndarray:
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    return img

def load_image(file: str) -> np.ndarray:
    return np.array(Image.open(file))

def plot_image(image: np.ndarray, masks: list) -> None:
    plt.figure(figsize=(20,20))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis('off')

    blank_image = 255*np.ones_like(image)

    anns = get_anns(masks)

    plt.subplot(1, 3, 2)
    plt.imshow(blank_image)
    plt.imshow(anns)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(anns)
    plt.axis('off')

    plt.show()

def main() -> None:
    image = load_image("/home/vlkjan6/Documents/diplomka/dataset/ff8e7fdb-1073-3592-ba5e-8111bc3ce48b/sensors/cameras/ring_front_center/315968518649927209.jpg")

    sam_checkpoint = "/home/vlkjan6/Documents/diplomka/code/weights/mobile_sam.pt"
    model_type = "vit_t"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    plot_image(image, masks)


if __name__ == "__main__":
    main()
