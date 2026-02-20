import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import cv2
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from modules.config import DISPLAY, on_key, RED, RESET

IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_learnings():
    if not os.path.exists("leaf_model.keras") or not os.path.exists("classes.json"):
        raise FileNotFoundError("Model or classes.json not found in current directory.")
    
    model = tf.keras.models.load_model("leaf_model.keras")
    with open("classes.json", 'r') as f:
        class_names = json.load(f)
    return model, class_names


def vis_predictions(imgs_rgb, filenames, predictions, confidences):
    count = min(len(imgs_rgb), DISPLAY)
    if count == 0:
        return

    cols = min(count, 4)
    rows = math.ceil(count / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), num="Predictions")
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        if i < count:
            ax.imshow(imgs_rgb[i])
            ax.set_title(f"{predictions[i]}\n({confidences[i]:.1f}%)", fontsize=10, fontweight='bold', pad=5)
            ax.set_xlabel(filenames[i], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def predict_images(img_paths, model, class_names):
    imgs_rgb = []
    filenames = []
    arrays = []

    for path in img_paths:
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT))
            imgs_rgb.append(rgb)
            filenames.append(os.path.basename(path))
            arrays.append(resized)

    if not arrays:
        raise ValueError("No valid images found to predict.")

    batch = np.array(arrays)
    preds = model.predict(batch, verbose=0)
    
    predictions = []
    confidences = []
    for p in preds:
        idx = np.argmax(p)
        predictions.append(class_names[idx])
        confidences.append(p[idx] * 100)

    vis_predictions(imgs_rgb, filenames, predictions, confidences)


def evaluate_directory(src, model, class_names):
    ds = tf.keras.utils.image_dataset_from_directory(
        src,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        shuffle=False
    )
    
    print(f"\nEvaluating directory against {len(class_names)} known classes...")
    loss, acc = model.evaluate(ds, verbose=1)
    print(f"\nOverall Accuracy on '{src}': {acc*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Predict leaf disease.")
    parser.add_argument('imgs', nargs='*', help='Image files to predict')
    parser.add_argument('-src', help='Directory of subfolders for accuracy evaluation')
    args = parser.parse_args()

    assert bool(args.imgs) != bool(args.src), "Provide either images OR a -src directory"

    model, class_names = load_learnings()

    if args.src:
        assert os.path.isdir(args.src), "-src directory not valid"
        evaluate_directory(args.src, model, class_names)
    else:
        predict_images(args.imgs, model, class_names)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        exit(1)
