import matplotlib.pyplot as plt
import os
import random
import shutil


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

DISPLAY = 6

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20


def split_dataset(src_dir, ratio):
    files = [
        f for f in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, f))
    ]
    random.shuffle(files)

    split_idx = int(len(files) * ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    train_dir = os.path.join(src_dir, 'train')
    val_dir = os.path.join(src_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for f in train_files:
        shutil.move(os.path.join(src_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.move(os.path.join(src_dir, f), os.path.join(val_dir, f))

    print(f"Split completed: {len(train_files)} train, {len(val_files)} val.")


def on_key(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)


def parse(dir):
    sdirs = [s for s in os.listdir(dir) if os.path.isdir(os.path.join(dir, s))]

    all_files = {}
    for s in sdirs:
        path = os.path.join(dir, s)
        content = os.listdir(path)
        files = [f for f in content if os.path.isfile(os.path.join(path, f))]
        all_files[s] = files

    return all_files
