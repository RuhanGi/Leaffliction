import argparse
import logging
import json
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from modules.config import DISPLAY, on_key, CYAN, GREEN, RED, RESET, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
from modules.transforms import apply_mask


def load_learnings():
    assert os.path.exists("leaf_model.keras"), "Model not found in current directory."
    assert os.path.exists("classes.json"), "classes.json not found in current directory."
    
    model = tf.keras.models.load_model("leaf_model.keras")
    with open("classes.json", 'r') as f:
        class_names = json.load(f)
    return model, class_names


def evaluate_directory(src, model, class_names):
    assert os.path.isdir(src), "-src directory not valid"
    
    print(CYAN + "\nEXTRACTING IMAGES:" + RESET)
    ds = tf.keras.utils.image_dataset_from_directory(
        src,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    print(CYAN + "\nEVALUATING MODEL:" + RESET)
    loss, acc = model.evaluate(ds, verbose=1)
    print(f"\n'{src}':\t Accuracy = {GREEN}[{acc*100:.2f}%]\t{RESET} Loss = {RED}[{loss:.4f}]{RESET}")


def vis_predictions(imgs_rgb, imgs_masked, filenames, predictions, confidences):
    count = len(imgs_rgb)
    if count == 0: 
        return

    fig, axes = plt.subplots(
        nrows=3, 
        ncols=count, 
        figsize=(count * 4, 10), 
        constrained_layout=True,
        num="Leaffliction Predictions"
    )

    if count == 1:
        axes = np.expand_dims(axes, axis=1)

    for c in range(count):
        axes[0, c].imshow(imgs_rgb[c])
        axes[0, c].set_title(filenames[c], fontsize=9)
        axes[0, c].axis('off')

        axes[1, c].imshow(imgs_masked[c])
        axes[1, c].set_title("Masked", fontsize=9)
        axes[1, c].axis('off')

        pred_str = f"PREDICTION:\n\n{predictions[c]}\n({confidences[c]:.1f}%)"
        axes[2, c].text(0.5, 0.5, pred_str, ha='center', va='center', 
                        color='darkgreen', fontsize=11, fontweight='bold')
        axes[2, c].axis('off')

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def predict_images(img_paths, model, class_names):
    imgs_rgb = []
    imgs_masked = []
    filenames = []
    arrays = []

    for path in img_paths:
        assert os.path.isfile(path), "Improper Arguments Passed!"

        img = cv2.imread(path)
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            masked = apply_mask(rgb)
            resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT))

            imgs_rgb.append(rgb)
            imgs_masked.append(masked)
            filenames.append(os.path.basename(path))
            arrays.append(resized)

    if not arrays:
        raise ValueError("No images found.")

    batch = np.array(arrays, dtype=np.float32)
    preds = model.predict(batch, verbose=0)

    predictions = []
    confidences = []
    for p in preds:
        idx = np.argmax(p)
        predictions.append(class_names[idx])
        confidences.append(p[idx] * 100)

    vis_predictions(imgs_rgb, imgs_masked, filenames, predictions, confidences)


def main():
    parser = argparse.ArgumentParser(description="Predict leaf disease.")
    parser.add_argument('imgs', nargs='*', help='Image files to predict')
    parser.add_argument('-src', help='Directory of subfolders for accuracy evaluation')
    args = parser.parse_args()

    assert bool(args.imgs) != bool(args.src), "Provide either images OR a -src directory"

    model, class_names = load_learnings()

    if args.src:
        evaluate_directory(args.src, model, class_names)
    else:
        predict_images(args.imgs, model, class_names)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        exit(1)
