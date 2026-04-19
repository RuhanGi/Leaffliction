import argparse
import logging
import json
import os

from modules.config import CYAN, GREEN, RED, RESET
from modules.config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, EPOCHS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def getData(dir):
    """Extracts the dataset efficiently from the passed directory."""
    assert os.path.exists(dir), f"Cannot find '{dir}"
    train_dir = os.path.join(dir, 'train')
    val_dir = os.path.join(dir, 'val')

    assert os.path.exists(train_dir), f"Cannot find '{train_dir}'"
    assert os.path.exists(val_dir), f"Cannot find '{val_dir}'"
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
    assert class_names == val_ds.class_names, "Mismatched classes"
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def create_model(num_classes):
    """Builds the Convolutional Neural Network."""
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        layers.Rescaling(1./255),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def save_learnings(model, class_names):
    """Saves the trained model and class labels locally."""
    model.save("leaf_model.keras")

    with open("classes.json", 'w') as f:
        json.dump(class_names, f)

    print("Successfully saved 'leaf_model.keras' and 'classes.json'")


def main():
    parser = argparse.ArgumentParser(description="Train Model on Dataset")
    parser.add_argument('dir', help='Dataset directory for train and val')
    args = parser.parse_args()

    print(CYAN + "\nEXTRACTING DATA:" + RESET)
    train_ds, val_ds, class_names = getData(args.dir)
    model = create_model(len(class_names))
    early = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True
    )

    print(CYAN + "\nSTARTING TRAINING:" + RESET)
    try:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[early]
        )
    except KeyboardInterrupt:
        print(RED + "\nInterrupted! Evaluating current state..." + RESET)

    val_loss, val_acc = model.evaluate(val_ds, verbose=1)
    print(GREEN + f"Final Validation Accuracy: {val_acc*100:.2f}%" + RESET)

    print(CYAN + "\nSAVING MODEL:" + RESET)
    save_learnings(model, class_names)


if __name__ == "__main__":
    import tensorflow as tf
    from keras import models, layers, callbacks
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        exit(1)
