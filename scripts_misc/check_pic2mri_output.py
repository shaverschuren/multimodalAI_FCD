import argparse
import math
import os
import sys
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

#!/usr/bin/env python3
"""
check_pic2mri_output.py

Scan a dataset directory for subfolders named RESPxxxx containing
pic2mri_output/scene/scene_screenshot.png, plot all found screenshots
in a grid with titles (RESPxxxx) and save the composed image.

Usage:
    python check_pic2mri_output.py /path/to/dataset_fs /path/to/output.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def find_screenshots(dataset_dir, pattern="RESP*"):
    # Search for RESP*/pic2mri_output/scene/scene_screenshot.png
    glob_pattern = os.path.join(dataset_dir, pattern, "pic2mri_output", "scene", "scene_screenshot.png")
    files = sorted(glob(glob_pattern))
    results = []
    for f in files:
        resp_dir = os.path.dirname(os.path.dirname(os.path.dirname(f)))
        resp_name = os.path.basename(resp_dir)
        results.append((resp_name, f))
    return results


def load_image(path):
    try:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))
    except Exception:
        return None


def plot_grid(images, titles, out_path, cols=None, dpi=150, highlights=None):
    n = len(images)
    if n == 0:
        raise ValueError("No images to plot.")
    if cols is None:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    figsize = (cols * 3, rows * 3)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # flatten axes for easy indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax in axes:
        ax.axis("off")

    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        if highlights and i < len(highlights) and highlights[i]:
            # draw a green box around the image (use axes coordinates so it surrounds the image area)
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             fill=False, edgecolor="lime", linewidth=3)
            ax.add_patch(rect)

    # hide any unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Collect and plot scene screenshots from RESP folders.")
    parser.add_argument("--dataset_dir", default="..\\data\\dataset_fs", help="Path to dataset_fs directory")
    parser.add_argument("--output_image", default="..\\data\\tmp\\pic2mri_summary.png", help="Path to save the output image (e.g. /tmp/all_screens.png)")
    parser.add_argument("--pattern", default="RESP*", help="Folder name pattern (default: RESP*)")
    parser.add_argument("--cols", type=int, default=None, help="Number of columns in the grid (default: auto)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    out_path = args.output_image

    if not os.path.isdir(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(2)

    entries = find_screenshots(dataset_dir, pattern=args.pattern)
    if not entries:
        print("No screenshots found.", file=sys.stderr)
        sys.exit(1)

    images = []
    titles = []
    highlights = []
    for name, path in tqdm(entries, desc="Loading images"):
        img = load_image(path)
        if img is None:
            tqdm.write(f"Failed to load image: {path}", file=sys.stderr)
            continue
        images.append(img)
        titles.append(name)
        # check for corresponding resection mask file
        resp_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
        mask_path = os.path.join(resp_dir, "pic2mri_output", "pic2mri_resection_mask_final.nii.gz")
        highlights.append(os.path.exists(mask_path))
        tqdm.write(f"Loaded image: {path}")

    if not images:
        print("No valid images loaded.", file=sys.stderr)
        sys.exit(1)

    plot_grid(images, titles, out_path, cols=args.cols, highlights=highlights)
    print(f"Saved composed image to: {out_path}")


if __name__ == "__main__":
    main()