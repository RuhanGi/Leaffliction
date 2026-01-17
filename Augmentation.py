import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

from modules.config import DISPLAY, on_key, RED, RESET
from modules.augments import transform, AvailableTransforms


def vis(data):
    keys = list(data.keys())
    arr2d = list(data.values())
    if not arr2d or not arr2d[0]:
        return

    cols = len(arr2d)
    rows = min(len(arr2d[0]), DISPLAY)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols*2, rows*2),
        num="Data Augmentation"
    )
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for j in range(cols):
        axes[0][j].set_title(keys[j], fontsize=12, fontweight='bold', pad=5)
        for i in range(rows):
            axes[i][j].imshow(arr2d[j][i])
            axes[i][j].axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def cved(path_list):
    """Loads images from a list of paths"""
    processed = []
    valid_paths = []
    for path in path_list:
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        if img is not None:
            processed.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            valid_paths.append(path)
    return processed, valid_paths


def save_files(og_paths, data):
    for change, imgs in data.items():
        if change == "Original":
            continue
        suffix = "_" + change.replace(" ", "_")
        for i, img_rgb in enumerate(imgs):
            if i < len(og_paths):
                root, ext = os.path.splitext(og_paths[i])
                new_path = f"{root}{suffix}{ext}"
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(new_path, img_bgr)


def generate_balanced_dataset(imgs, paths, target_count):
    """
    Randomly generates images until total count = target_count.
    """
    current_count = len(imgs)
    needed = target_count - current_count
    if needed <= 0:
        return

    for i in range(needed):
        rand_idx = random.randint(0, len(imgs) - 1)
        func, name = random.choice(AvailableTransforms)
        aug_img = func(imgs[rand_idx])

        directory = os.path.dirname(paths[rand_idx])
        filename = os.path.basename(paths[rand_idx])
        base_name = filename.split('_aug_')[0].split('.')[0]
        ext = os.path.splitext(filename)[1]
        unique_id = current_count + i
        save_name = f"{base_name}_aug_{unique_id}_{name}{ext}"
        save_path = os.path.join(directory, save_name)

        img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
        imgs.append(aug_img)
        paths.append(save_path)

    print(f"\nSuccessfully generated {needed} images.")


def main():
    parser = argparse.ArgumentParser(description="Augments images.")
    parser.add_argument('imgs', nargs='*', help='List of images to augment')
    parser.add_argument('-src', help='Source directory containing images')
    parser.add_argument('--save', action='store_true', help='Save changes')
    parser.add_argument('-count', type=int, help='Target count of images')
    args = parser.parse_args()

    assert bool(args.imgs) != bool(args.src), "either pass imgs or src"
    if args.src:
        assert os.path.isdir(args.src), "src directory not valid"
        args.imgs = [os.path.join(args.src, f) for f in os.listdir(args.src)]
        args.imgs = [f for f in args.imgs if os.path.isfile(f)]

    if not args.count and not args.save:
        args.imgs = args.imgs[:DISPLAY]
    loaded_imgs, valid_paths = cved(args.imgs)
    assert bool(loaded_imgs), "No valid images found"
    if args.count:
        generate_balanced_dataset(loaded_imgs, valid_paths, args.count)
    else:
        data = transform(loaded_imgs)
        vis(data)
        if args.save:
            save_files(valid_paths, data)
            print("Grid augmentations saved.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        exit(1)
