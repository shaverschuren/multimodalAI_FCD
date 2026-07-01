import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from matplotlib.patches import Rectangle
from tqdm import tqdm

OVERWRITE_EXISTING = True

def load_intensity_and_normalize(img_path: Path):
    """Load anatomical image and robustly normalize to 0-1."""
    img = nib.as_closest_canonical(nib.load(str(img_path)))
    data = np.asanyarray(img.dataobj).astype(np.float32, copy=True)
    img = nib.Nifti1Image(data.copy(), img.affine, img.header.copy())
    p2, p98 = np.percentile(data, [2, 98])
    data = np.clip(data, p2, p98)
    dmin = float(data.min())
    dmax = float(data.max())
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)
    else:
        data = np.zeros_like(data, dtype=float)
    return data, img


def load_label_image(img_path: Path):
    """Load atlas labels as integer values (do not normalize categorical IDs)."""
    img = nib.as_closest_canonical(nib.load(str(img_path)))
    data = np.asanyarray(img.dataobj).astype(np.int32, copy=True)
    img = nib.Nifti1Image(data.copy(), img.affine, img.header.copy())
    return data, img


def pick_slice_index(t1_data: np.ndarray) -> int:
    """Use the middle axial slice."""
    return int(t1_data.shape[2] // 2)


def resample_atlas_to_t1(atlas_img: nib.spatialimages.SpatialImage, t1_img: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Resample atlas labels to T1 grid/affine using nearest-neighbor interpolation."""
    atlas_resampled_img = resample_from_to(atlas_img, t1_img, order=0, cval=0)
    return np.asarray(atlas_resampled_img.get_fdata(), dtype=np.int32)


def get_inplane_spacing(img: nib.spatialimages.SpatialImage) -> tuple[float, float]:
    """Return (x, y) voxel spacing in mm for the axial x-y plane."""
    voxel_sizes = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    sx = float(voxel_sizes[0]) if voxel_sizes[0] > 0 else 1.0
    sy = float(voxel_sizes[1]) if voxel_sizes[1] > 0 else 1.0
    return sx, sy


def apply_inplane_aspect(ax, inplane_spacing: tuple[float, float]):
    """Set axis aspect so anisotropic in-plane voxels are displayed with true scale."""
    sx, sy = inplane_spacing
    ax.set_aspect(sy / sx)


def find_t1_path(mri_preop_dir: Path, subject_id: str):
    """Find pre-op T1 in dataset_mri/pre_operative/<id>/<id>-preop-T1w*."""
    subject_mri_dir = mri_preop_dir / subject_id
    candidates = [
        subject_mri_dir / f"{subject_id}-preop-T1w.nii.gz",
        subject_mri_dir / f"{subject_id}-preop-T1w.nii",
        subject_mri_dir / f"{subject_id}-preop-T1w.mgz",
    ]
    for c in candidates:
        if c.exists():
            return c

    for pattern in [
        f"{subject_id}-preop-T1w*",
        "*preop-T1w.nii.gz",
        "*preop-T1w.nii",
    ]:
        hits = sorted(subject_mri_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def find_atlas_path(subject_dir: Path):
    """Only accept the 2009 Destrieux+aseg atlas variants."""
    candidates = [
        subject_dir / "mri" / "aparc.a2009s+aseg.mgz",
        subject_dir / "mri" / "aparc.a2009s+aseg.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            return c

    for pattern in ["**/*aparc.a2009s+aseg.nii.gz", "**/*aparc.a2009s+aseg.mgz"]:
        hits = sorted(subject_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def add_missing_atlas_box(ax, t1_slice: np.ndarray):
    h, w = t1_slice.T.shape
    ax.add_patch(
        Rectangle(
            (1, 1),
            max(w - 2, 1),
            max(h - 2, 1),
            linewidth=8,
            edgecolor="red",
            facecolor="none",
        )
    )


def as_display_image(slice_2d: np.ndarray) -> np.ndarray:
    """Convert x-y slice to imshow row-col layout."""
    return slice_2d.T


def plot_missing_t1(subject_id: str, shape: tuple[int, int] = (256, 256)):
    """Create a placeholder panel for missing T1 with red warning square."""
    fig, ax = plt.subplots(figsize=(6, 6))
    t1_slice = np.zeros(shape, dtype=float)
    ax.imshow(as_display_image(t1_slice), cmap="gray", origin="lower")
    apply_inplane_aspect(ax, (1.0, 1.0))
    add_missing_atlas_box(ax, t1_slice)
    ax.set_title(f"{subject_id} (T1 missing)", fontsize=10)
    ax.axis("off")
    return fig, t1_slice


def plot_alignment(
    t1_data: np.ndarray,
    atlas_data: np.ndarray | None,
    subject_id: str,
    inplane_spacing: tuple[float, float],
):
    fig, ax = plt.subplots(figsize=(6, 6))

    slice_idx = pick_slice_index(t1_data)
    t1_slice = t1_data[:, :, slice_idx]

    ax.imshow(as_display_image(t1_slice), cmap="gray", origin="lower")
    apply_inplane_aspect(ax, inplane_spacing)

    if atlas_data is None:
        add_missing_atlas_box(ax, t1_slice)
        ax.set_title(f"{subject_id} (atlas missing)", fontsize=10)
        atlas_slice = None
    else:
        atlas_slice = atlas_data[:, :, slice_idx]
        atlas_masked = np.ma.masked_where(atlas_slice == 0, atlas_slice)
        ax.imshow(
            as_display_image(atlas_masked),
            cmap="nipy_spectral",
            alpha=0.35,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(subject_id, fontsize=10)

    ax.axis("off")
    return fig, t1_slice, atlas_slice


def get_subject_ids(fs_dir: Path, selection_csv: Path | None):
    if selection_csv is not None and selection_csv.exists():
        df = pd.read_csv(selection_csv)
        if "Participant Id" in df.columns:
            ids = sorted(df["Participant Id"].dropna().astype(str).unique().tolist())
            if ids:
                return ids

    return sorted([p.name for p in fs_dir.iterdir() if p.is_dir()])


def parse_args():
    base_path = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Check T1 and FreeSurfer atlas alignment per subject."
    )
    parser.add_argument(
        "--fs-dir",
        type=Path,
        default=base_path / "data" / "dataset_fs",
        help="Path to FreeSurfer dataset root.",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=base_path / "data" / "selection" / "selected_summary.csv",
        help="Optional CSV containing Participant Id to filter subject list.",
    )
    parser.add_argument(
        "--mri-dir",
        type=Path,
        default=base_path / "data" / "dataset_mri" / "pre_operative",
        help="Path to pre-operative MRI dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_path / "data" / "data_availability" / "FS_alignment",
        help="Directory for per-subject and grid plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    fs_dir = args.fs_dir
    mri_dir = args.mri_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not fs_dir.exists():
        print(f"FreeSurfer directory not found: {fs_dir}")
        return
    if not mri_dir.exists():
        print(f"Pre-operative MRI directory not found: {mri_dir}")
        return

    subject_ids = get_subject_ids(fs_dir, args.selection_csv)
    if not subject_ids:
        print("No subjects found.")
        return

    grid_data = []

    for subject_id in tqdm(subject_ids, desc="Checking atlas alignment"):
        subject_dir = fs_dir / subject_id
        individual_path = output_dir / f"{subject_id}_alignment.png"

        if individual_path.exists() and not OVERWRITE_EXISTING:
            tqdm.write(f"{subject_id}: Alignment image already exists, skipping.")
            continue

        t1_path = find_t1_path(mri_dir, subject_id)
        if t1_path is None:
            tqdm.write(f"No T1 image found for {subject_id}")
            fig, t1_slice = plot_missing_t1(subject_id)
            individual_path = output_dir / f"{subject_id}_alignment.png"
            fig.savefig(individual_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            grid_data.append((subject_id, t1_slice, None, True, (1.0, 1.0)))
            continue

        atlas_path = find_atlas_path(subject_dir) if subject_dir.exists() else None

        try:
            t1_data, t1_img = load_intensity_and_normalize(t1_path)
            inplane_spacing = get_inplane_spacing(t1_img)
            atlas_data = None
            if atlas_path is not None:
                _, atlas_img = load_label_image(atlas_path)
                atlas_data = resample_atlas_to_t1(atlas_img, t1_img)

            fig, t1_slice, atlas_slice = plot_alignment(
                t1_data,
                atlas_data,
                subject_id,
                inplane_spacing,
            )
            individual_path = output_dir / f"{subject_id}_alignment.png"
            fig.savefig(individual_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            grid_data.append((subject_id, t1_slice, atlas_slice, False, inplane_spacing))

        except Exception as exc:
            tqdm.write(f"Error processing {subject_id}: {exc}")

    if not grid_data:
        tqdm.write("No subjects processed successfully.")
        return

    num_subjects = len(grid_data)
    num_cols = int(np.ceil(np.sqrt(num_subjects)))
    num_rows = int(np.ceil(num_subjects / num_cols))

    fig = plt.figure(figsize=(num_cols * 5, num_rows * 5))

    for idx, (subject_id, t1_slice, atlas_slice, t1_missing, inplane_spacing) in enumerate(grid_data):
        ax = fig.add_subplot(num_rows, num_cols, idx + 1)
        ax.imshow(as_display_image(t1_slice), cmap="gray", origin="lower")
        apply_inplane_aspect(ax, inplane_spacing)

        if t1_missing:
            add_missing_atlas_box(ax, t1_slice)
            ax.set_title(f"{subject_id} (T1 missing)", fontsize=8)
        elif atlas_slice is None:
            add_missing_atlas_box(ax, t1_slice)
            ax.set_title(f"{subject_id} (atlas missing)", fontsize=8)
        else:
            atlas_masked = np.ma.masked_where(atlas_slice == 0, atlas_slice)
            ax.imshow(
                as_display_image(atlas_masked),
                cmap="nipy_spectral",
                alpha=0.35,
                origin="lower",
                interpolation="nearest",
            )
            ax.set_title(subject_id, fontsize=8)

        ax.axis("off")

    grid_path = output_dir / "FS_alignment_grid.png"
    fig.savefig(grid_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid image saved to {grid_path}")


if __name__ == "__main__":
    main()
