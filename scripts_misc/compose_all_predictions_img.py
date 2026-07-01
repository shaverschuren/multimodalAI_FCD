"""
Compose prediction images from eeg/, mri/, and multimodal/ subfolders into a
single grid image saved in the root folder.

Grid layout:
    columns : EEG | MRI | Multimodal
    rows    : one per subject (sorted by subject ID)

Subjects that lack an image in a given modality get a blank placeholder.
"""

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path("L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\results\\runs")
OUTPUT = ROOT / "composed_predictions.png"

MODALITIES = ["eeg", "mri", "multimodal_late_fusion"]
SUBJECT_RE = re.compile(r"(RESP\d+)", re.IGNORECASE)

PAD = 6          # pixels between cells
HEADER_H = 120    # height of column-header bar
LABEL_W = 300    # width of row-label column
FONT_SIZE = 48
HEADER_FONT_SIZE = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_images(folder: Path) -> dict[str, Path]:
    """Return {subject_id: path} for all PNGs in *folder*."""
    result = {}
    for p in sorted(folder.glob("*.png")):
        m = SUBJECT_RE.search(p.stem)
        if m:
            result[m.group(1).upper()] = p
    return result


def load_or_blank(path: Path | None, size: tuple[int, int]) -> Image.Image:
    if path is not None:
        return Image.open(path).convert("RGB")
    img = Image.new("RGB", size, color=(220, 220, 220))
    d = ImageDraw.Draw(img)
    d.text((size[0] // 2 - 20, size[1] // 2 - 8), "n/a", fill=(120, 120, 120))
    return img


def make_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        ("arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf") if bold
        else ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    modal_images: dict[str, dict[str, Path]] = {}
    for m in MODALITIES:
        images: dict[str, Path] = {}
        for folder in ROOT.glob(f"{m}\\**\\per_*"):
            images.update(find_images(folder))
        modal_images[m] = images

    # Only keep subjects present in ALL modalities
    all_subjects = sorted(
        set.intersection(*(set(imgs.keys()) for imgs in modal_images.values()))
    )
    n_rows = len(all_subjects)
    n_cols = len(MODALITIES)

    # Determine cell size from first image found
    sample_path = next(
        p for imgs in modal_images.values() for p in imgs.values()
    )
    cell_w, cell_h = Image.open(sample_path).size

    canvas_w = LABEL_W + PAD + n_cols * (cell_w + PAD)
    canvas_h = HEADER_H + PAD + n_rows * (cell_h + PAD)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = make_font(FONT_SIZE)
    font_header = make_font(HEADER_FONT_SIZE, bold=True)
    font_small = make_font(FONT_SIZE)

    # Column headers
    for col, mod in enumerate(MODALITIES):
        x = LABEL_W + PAD + col * (cell_w + PAD) + cell_w // 2
        draw.text((x, PAD), mod.upper(), fill=(255, 255, 255), font=font_header, anchor="mt")

    # Rows
    for row, subject in enumerate(all_subjects):
        y_top = HEADER_H + PAD + row * (cell_h + PAD)

        # Row label
        draw.text(
            (LABEL_W // 2, y_top + cell_h // 2),
            subject,
            fill=(255, 255, 255),
            font=font_small,
            anchor="mm",
        )

        # Cells
        for col, mod in enumerate(MODALITIES):
            x_left = LABEL_W + PAD + col * (cell_w + PAD)
            img_path = modal_images[mod].get(subject)
            cell = load_or_blank(img_path, (cell_w, cell_h))
            canvas.paste(cell, (x_left, y_top))

    canvas.save(OUTPUT, optimize=True)
    print(f"Saved → {OUTPUT}  ({canvas_w}x{canvas_h} px, {n_rows} subjects)")


if __name__ == "__main__":
    main()
