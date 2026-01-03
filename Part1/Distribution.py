#!/usr/bin/env python3

import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt


def is_image(filename):
    ext = filename.lower().split('.')[-1]
    return ext in {"jpg", "jpeg", "png", "bmp", "gif", "tiff"}


def collect_distribution(root_dir):
    data = defaultdict(int)

    for current_root, dirs, files in os.walk(root_dir):
        # skip the root directory itself
        if current_root == root_dir:
            continue

        label = os.path.basename(current_root)
        for f in files:
            if is_image(f):
                data[label] += 1

    return data


def plot_charts(data, plant_name):
    labels = list(data.keys())
    values = list(data.values())

    # Bar chart
    plt.figure()
    plt.bar(labels, values)
    plt.title(f"Image Distribution for {plant_name}")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()

    # Pie chart
    plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title(f"Image Distribution for {plant_name}")
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: ./Distribution.py <directory>")
        sys.exit(1)

    root_dir = sys.argv[1]

    if not os.path.isdir(root_dir):
        print("Error: provided path is not a directory")
        sys.exit(1)

    plant_name = os.path.basename(os.path.abspath(root_dir))
    distribution = collect_distribution(root_dir)

    if not distribution:
        print("No images found in subdirectories")
        sys.exit(1)

    for label, count in distribution.items():
        print(f"{label}: {count} images")

    plot_charts(distribution, plant_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")