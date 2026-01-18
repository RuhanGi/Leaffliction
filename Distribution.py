import matplotlib.pyplot as plt
import argparse

from modules.config import parse, on_key, RED, RESET


def is_image(filename):
    """Safety check: ensures we only count actual images."""
    return filename.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    )


def pie_chart(all_files, counts, ax, dir):
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=all_files.keys(),
        autopct=lambda p: f'{p:.1f}%',
        textprops={'weight': 'bold'},
        labeldistance=None
    )
    for i, (text, autotext) in enumerate(zip(all_files.keys(), autotexts)):
        autotext.set_text(f"{text}\n{autotext.get_text()}")
        autotext.set_color("white")
    ax.set_title(f"{dir} Percentages", fontsize=16, fontweight='bold')
    return [wedge.get_facecolor() for wedge in wedges]


def bar_chart(all_files, counts, ax, colors):
    bars = ax.bar(
        all_files.keys(),
        counts,
        color=colors
    )
    ax.bar_label(bars, padding=-15, weight="bold", color="white")
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')


def analyse(dir, all_files):
    clean_files = {}
    for category, files in all_files.items():
        valid = [f for f in files if is_image(f)]
        if valid:
            clean_files[category] = valid
    all_files = clean_files

    print(f"[Analysis] Distribution for '{dir}':")
    total = 0
    for category, files in all_files.items():
        count = len(files)
        total += count
        print(f"  - {category}: {count} images")
    print(f"  = Total: {total} images\n")

    counts = [len(files) for files in all_files.values()]
    assert bool(len(all_files)) and bool(sum(counts)), "No file found!"

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), num=(dir + " Distribution"))
    colors = pie_chart(all_files, counts, ax[0], dir)
    bar_chart(all_files, counts, ax[1], colors)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analysis of Dataset: dir/subdirs/images.jpg"
    )
    parser.add_argument('dir', help='directory of analysis')
    args = parser.parse_args()

    all_files = parse(args.dir)
    analyse(args.dir, all_files)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        exit(1)
