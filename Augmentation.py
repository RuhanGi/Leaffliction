import matplotlib.pyplot as plt
import argparse
import math
import cv2

from modules.config import on_key, RED, RESET


def graph(dir, all_files):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), num=(dir + " Distribution"))
    ax[0].pie()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def see(imgs):
    n = len(imgs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if n > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i < n:
            cv_img = cv2.imread(imgs[i])
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.axis('off')
            # ax.set_title(f"Image {i+1}")
        else:
            ax.axis('off')

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()


def vis(arr2d):
    if len(arr2d) == 0 or len(arr2d[0]) == 0:
        return

    cols = len(arr2d)
    rows = len(arr2d[0])
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    for i in range(rows):
        for j in range(cols):
            axes[i][j].imshow(arr2d[j][i])
            axes[i][j].axis('off')
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()


def transform(imgs):
    arr2d = [imgs]
    # 1 = Horizontal, 0 = Vertical, -1 = Both
    arr2d.append([cv2.flip(img, 0) for img in imgs])
    # specific constant for 90 degrees
    arr2d.append([cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in imgs])
    return arr2d


def cved(imgs):
    # ! DO ERROR HANDLING
    cv_imgs = [cv2.imread(img) for img in imgs]
    imgs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in cv_imgs]
    return imgs_rgb


def main():
    parser = argparse.ArgumentParser(
        description="Augments an image and displays it"
    )
    parser.add_argument('imgs', nargs='+', help='image to augment')
    args = parser.parse_args()

    # see(args.imgs[:8])
    imgs = cved(args.imgs[:3])
    arr2d = transform(imgs)
    vis(arr2d)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        exit(1)
