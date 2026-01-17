import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
import os

from modules.config import DISPLAY, on_key, RED, RESET
from modules.transforms import transform


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


def vis_histogram_analysis(img_rgb):
    """
    Replicates the complex histogram from Leaffliction Figure IV.7.
    Plots RGB, HSV, and LAB channels all on one graph.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    channels = [
        (img_rgb, 0, 'Red', 'red'),
        (img_rgb, 1, 'Green', 'green'),
        (img_rgb, 2, 'Blue', 'blue'),
        (img_hsv, 0, 'Hue', 'purple'),
        (img_hsv, 1, 'Saturation', 'cyan'),
        (img_hsv, 2, 'Value', 'orange'),
        (img_lab, 0, 'Lightness', 'black'),
        (img_lab, 1, 'Green-Magenta', 'magenta'),
        (img_lab, 2, 'Blue-Yellow', 'yellow')
    ]
    fig, (ax_img, ax_hist) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(14, 6),
        gridspec_kw={'width_ratios': [1, 2]}
    )
    ax_img.imshow(img_rgb)
    ax_img.set_title("Original Image")
    ax_img.axis('off')

    for (src_img, channel_idx, label, color) in channels:
        hist = cv2.calcHist([src_img], [channel_idx], None, [256], [0, 256])
        total_pixels = src_img.shape[0] * src_img.shape[1]
        hist_proportion = (hist / total_pixels) * 100
        ax_hist.plot(
            hist_proportion, color=color,
            label=label, linewidth=1.2, alpha=0.7
        )

    ax_hist.set_title("Pixel Intensity Distribution (All Channels)")
    ax_hist.set_xlabel("Pixel Intensity (0-255)")
    ax_hist.set_ylabel("Proportion of pixels (%)")
    ax_hist.set_xlim([0, 256])
    ax_hist.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    ax_hist.grid(True, linestyle='--', alpha=0.3)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()


def cved(path_list):
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


# def save_files(og_paths, data, dst):
#     os.makedirs(dst, exist_ok=True)
#     for change, imgs in data.items():
#         suffix = "_" + change.replace(" ", "_")
#         for i, img_rgb in enumerate(imgs):
#             if i < len(og_paths):
#                 filename = os.path.basename(og_paths[i])
#                 name, ext = os.path.splitext(filename)
#                 new_path = os.path.join(dst, f"{name}{suffix}{ext}")
#                 img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
#                 cv2.imwrite(new_path, img_bgr)


def save_files(og_paths, data, dst, src_root):
    """
    Saves files while preserving the subdirectory structure from src_root.
    """
    for change, imgs in data.items():
        if change == "Original":
            continue

        suffix = "_" + change.replace(" ", "_")
        for i, img_rgb in enumerate(imgs):
            if i >= len(og_paths):
                break
            original_full_path = og_paths[i]
            if src_root:
                rel_path = os.path.relpath(original_full_path, src_root)
            else:
                rel_path = os.path.basename(original_full_path)
            dest_dir = os.path.join(dst, os.path.dirname(rel_path))
            filename = os.path.basename(rel_path)
            name, ext = os.path.splitext(filename)
            os.makedirs(dest_dir, exist_ok=True)
            new_path = os.path.join(dest_dir, f"{name}{suffix}{ext}")
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_path, img_bgr)


def main():
    parser = argparse.ArgumentParser(description="Image Transformations")
    parser.add_argument('imgs', nargs='*', help='Image files')
    parser.add_argument('-src', help='Source directory (Recursive)')
    parser.add_argument('-dst', help='Destination directory')
    parser.add_argument('-tsf', help='"mask", "blur", "roi"')
    args = parser.parse_args()

    assert bool(args.imgs) != bool(args.src), "either pass imgs or src"
    if args.src:
        assert os.path.isdir(args.src), "src directory not valid"
        for root, _, files in os.walk(args.src):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    args.imgs.append(os.path.join(root, file))

    if not args.dst:
        args.imgs = args.imgs[:DISPLAY]
    imgs, valid_paths = cved(args.imgs)
    data = transform(imgs, selection=args.tsf)
    vis(data)
    if imgs and not args.dst:
        vis_histogram_analysis(imgs[0])

    if args.dst:
        save_files(valid_paths, data, args.dst, args.src)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        exit(1)
