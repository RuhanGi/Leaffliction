import matplotlib.pyplot as plt
import os


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

DISPLAY = 6


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
