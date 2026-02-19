#!/usr/bin/env python3
import os
import sys
import argparse
import json
import pathlib
import random

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 15

def get_images_and_labels(data_dir):
    """
    Manually scans a nested directory structure: Root/Plant/Disease/Image
    Returns:
        file_paths: List of strings (path to images)
        labels: List of strings (e.g., "Apple_Black_rot")
    """
    data_root = pathlib.Path(data_dir)
    all_image_paths = []
    all_labels = []
    
    # Supported extensions
    exts = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG'}
    
    # 1. Check if structure is nested (Root/Plant/Disease) or flat (Root/Disease)
    # We look for images 2 levels deep vs 1 level deep
    is_nested = False
    for ext in exts:
        if list(data_root.glob(f"*/*/{ext}")): # Checks Root/*/*/Image
            is_nested = True
            break
    
    print(f"[Data Discovery] Detected {'NESTED (Plant/Disease)' if is_nested else 'FLAT (Disease)'} structure.")

    # 2. Collect Paths
    if is_nested:
        # Scan structure: masked/Apple/Healthy/img.jpg
        for item in data_root.glob('*/*'):
            if item.is_dir():
                plant_name = item.parent.name # e.g., Apple
                disease_name = item.name      # e.g., Healthy
                class_label = f"{plant_name}_{disease_name}" # e.g., Apple_Healthy
                
                # Grab all images in this folder
                for ext in exts:
                    for img_path in item.glob(ext):
                        all_image_paths.append(str(img_path))
                        all_labels.append(class_label)
    else:
        # Scan structure: masked/Apple_Healthy/img.jpg (Standard Keras format)
        for item in data_root.glob('*'):
            if item.is_dir():
                class_label = item.name
                for ext in exts:
                    for img_path in item.glob(ext):
                        all_image_paths.append(str(img_path))
                        all_labels.append(class_label)

    return all_image_paths, all_labels

def preprocess_image(path, label):
    """
    TF Function to load and resize images
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0 # Normalize here manually
    return img, label

def create_dataset(paths, labels, class_to_index):
    """
    Converts lists of paths/labels into a TensorFlow Dataset
    """
    # Convert string labels to integer indices
    label_indices = [class_to_index[l] for l in labels]
    
    # Create dataset from tensor slices
    ds = tf.data.Dataset.from_tensor_slices((paths, label_indices))
    
    # Map the loading function (Parallelized)
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

def create_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        # Note: Rescaling is done in preprocess_image, so we don't need it here
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def save_learnings(model, class_names):
    print("\n[Saving] Saving model and classes...")
    model.save("leaf_model.keras")
    with open("classes.json", 'w') as f:
        json.dump(class_names, f)
    print("[Success] Saved to leaf_model.keras and classes.json")

def main():
    parser = argparse.ArgumentParser(description="Train model on leaf dataset")
    parser.add_argument('dir', help='Directory containing the class subfolders')
    args = parser.parse_args()
    
    if not os.path.isdir(args.dir):
        print("Error: Directory not found.")
        sys.exit(1)

    print(f"Loading data from: {args.dir}")

    # 1. Manual Data Discovery
    file_paths, file_labels = get_images_and_labels(args.dir)
    
    if not file_paths:
        print("Error: No images found. Check directory structure.")
        sys.exit(1)

    # 2. Determine Classes
    unique_labels = sorted(list(set(file_labels)))
    class_to_index = {label: i for i, label in enumerate(unique_labels)}
    print(f"Found {len(file_paths)} images belonging to {len(unique_labels)} classes:")
    print(unique_labels)

    # 3. Split Data (80% Train, 20% Val)
    # Stratify ensures we get a mix of all classes in both sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, file_labels, test_size=0.2, random_state=123, stratify=file_labels
    )

    print(f"Training on {len(train_paths)} files, Validating on {len(val_paths)} files.")

    # 4. Build TF Datasets
    train_ds = create_dataset(train_paths, train_labels, class_to_index)
    val_ds = create_dataset(val_paths, val_labels, class_to_index)

    # 5. Batch & Prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # 6. Build & Train Model
    model = create_model(len(unique_labels))
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
    )

    print("\n[Training] Starting...")
    try:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[early_stopping]
        )
    except KeyboardInterrupt:
        print("\n[Notice] Interrupted. Saving current state...")

    # 7. Evaluate
    print("\n[Evaluation] Validating accuracy...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=2)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")

    if val_acc < 0.90:
        print("Warning: Accuracy is below 90%.")

    save_learnings(model, unique_labels)

if __name__ == "__main__":
    main()