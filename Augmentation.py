import matplotlib.pyplot as plt
import argparse

from modules.config import on_key, RED, RESET


def graph(dir, all_files):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), num=(dir + " Distribution"))
    ax[0].pie()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Augments an image and displays it"
    )
    parser.add_argument('img', help='image to augment')
    args = parser.parse_args()



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        exit(1)
