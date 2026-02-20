#!/usr/bin/env python3
import os
import sys
import argparse
import json

# Silence Windows TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- CONFIGURATION ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20

def create_model(num_classes):
    """Builds the Convolutional Neural Network."""
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # Rescaling (0-255 to 0.0-1.0)
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), # Prevents overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def save_learnings(model, class_names):
    """Saves the trained model and class labels locally."""
    print("\n[Saving] Packaging learnings...")
    model.save("leaf_model.keras")
    
    with open("classes.json", 'w') as f:
        json.dump(class_names, f)
    
    print("Successfully saved 'leaf_model.keras' and 'classes.json'")


def main():
    parser = argparse.ArgumentParser(description="Train model on split leaf dataset")
    parser.add_argument('dir', help='Directory containing "train" and "val" subfolders')
    args = parser.parse_args()
    
    data_dir = args.dir
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"ERROR: Cannot find '{train_dir}' or '{val_dir}'.")
        print("Please ensure your directory has the pre-split train and val folders.")
        sys.exit(1)

    print(f"📂 Loading datasets from {data_dir}...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Found {len(class_names)} Classes: {class_names}")

    # Optimize data pipeline for speed
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build and Train
    model = create_model(len(class_names))
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=4, 
        restore_best_weights=True, 
        verbose=1
    )

    print("\n🚀 Starting Training...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[early_stopping]
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Evaluating current state...")

    val_loss, val_acc = model.evaluate(val_ds, verbose=2)
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")

    save_learnings(model, class_names)


if __name__ == "__main__":
    main()
