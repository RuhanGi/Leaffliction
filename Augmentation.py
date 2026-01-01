import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
import os

from modules.config import DISPLAY, on_key, RED, RESET
from modules.augments import transform


def graph(dir, all_files):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), num=(dir + " Distribution"))
    ax[0].pie()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


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


def cved(imgs):
    processed = []
    for path in imgs:
        assert os.path.isfile(path), f"improper file: '{path}'"
        img = cv2.imread(path)
        assert img is not None, f"improper image '{path}'"
        processed.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return processed


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


def main():
    parser = argparse.ArgumentParser(
        description="Augments an image and displays it"
    )
    parser.add_argument('imgs', nargs='+', help='image to augment')
    parser.add_argument('--save', action='store_true', help='Save changes')
    args = parser.parse_args()

    if not args.save:
        args.imgs = args.imgs[:DISPLAY]
    imgs = cved(args.imgs)
    data = transform(imgs)
    vis(data)
    if args.save:
        save_files(args.imgs, data)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        exit(1)
