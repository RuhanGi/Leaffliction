#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
import json
import zipfile
import matplotlib.pyplot as plt

# --- MASKING HELPER (Copied from transformations) ---
def get_plant_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Standard green mask range
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def apply_mask(img):
    mask = get_plant_mask(img)
    return cv2.bitwise_and(img, img, mask=mask)

# --- PREDICTION LOGIC ---
def load_learnings(zip_path="learnings.zip"):
    """Extracts model and class names from the zip file"""
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found. Run train.py first.")
        sys.exit(1)
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("temp_model")
        
    model = tf.keras.models.load_model("temp_model/leaf_model.h5")
    with open("temp_model/classes.json", 'r') as f:
        class_names = json.load(f)
        
    # Cleanup temp folder (optional, or keep for speed)
    import shutil
    shutil.rmtree("temp_model")
    
    return model, class_names

def predict_and_display(image_path, model, class_names):
    # 1. Load Original Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Could not read image.")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Transform (Apply Mask)
    # The model expects the image to look like the training data (black background)
    transformed_img = apply_mask(img_rgb)

    # 3. Prepare for Model (Resize & Batch Dimension)
    img_resized = cv2.resize(transformed_img, (256, 256))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # 4. Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print(f"Prediction: {predicted_class} ({confidence:.2f}%)")

    # 5. Display (Original vs Transformed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(img_rgb)
    ax1.set_title("Original Image")
    ax1.axis("off")
    
    ax2.imshow(transformed_img)
    ax2.set_title(f"Prediction: {predicted_class}\nConf: {confidence:.2f}%")
    ax2.axis("off")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Predict leaf disease")
    parser.add_argument('image', help='Path to an image file')
    args = parser.parse_args()

    # Load resources
    model, class_names = load_learnings()
    
    # Run prediction
    predict_and_display(args.image, model, class_names)

if __name__ == "__main__":
    main()