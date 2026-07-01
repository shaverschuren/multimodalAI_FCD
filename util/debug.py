"""
util/debug.py
Utility functions for debugging and quality control plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def quality_control_plot(
    segments,
    patient_id="RESPxxxx",
    channel_names=[
    "Fp1", "Fp2", "F9", "F10", "F7", "F3", "Fz", "F4", "F8", 
    "T7", "C3", "Cz", "C4", "T8", 
    "P7", "P3", "Pz", "P4", "P8", 
    "O1", "O2"],
    n_samples=25,
    figsize=(18, 10),
    downsample=1,
    fseq=256,
    output_path=None
):
    """
    Plot a QC figure where:
      - First axis = average of max(100, n_segments) random segments
      - Remaining axes = individual sample segments
    """

    # Handle different input types (numpy vs tensor)
    if hasattr(segments, 'detach'):
        # PyTorch tensor - detach and move to CPU
        segments = segments.detach().cpu().numpy()

    if len(segments) == 0:
        print("No segments to plot")
        return

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_channels = len(channel_names)

    # Compute average segment
    # ------------------------------
    n_total = len(segments)
    n_avg = min(max(100, n_samples), n_total)

    # Randomly choose segments for averaging
    avg_idx = np.random.choice(n_total, n_avg, replace=False)
    avg_seg = segments[avg_idx].mean(axis=0)  # shape (C, T)

    # Choose individual samples
    # ------------------------------
    if n_samples < n_total:
        n_individual = min(n_samples, n_total)
        sample_idx = np.random.choice(n_total, n_individual, replace=False)
        segs = segments[sample_idx]
    else:
        n_individual = n_total
        segs = segments

    # Downsample for plotting (we lowpassed at 100Hz, so should be fine)
    sfreq_orig = fseq
    if downsample > 1:
        avg_seg = avg_seg[..., ::downsample]
        segs = segs[..., ::downsample]
        sfreq = sfreq_orig / downsample
    else:
        sfreq = sfreq_orig

    n_time = avg_seg.shape[-1]
    time = np.arange(n_time)

    # Vertical spacing
    spacing = 8
    offsets = np.arange(n_channels)[::-1] * spacing

    # Build side-by-side axes
    fig, axes = plt.subplots(
        1, n_individual + 1,
        figsize=figsize,
        sharey=True,
        gridspec_kw={'wspace': 0.0}
    )

    # Make iterable
    if n_individual + 1 == 1:
        axes = [axes]

    # Major ticks (1s)
    major_ticks = np.arange(0, n_time, int(sfreq) // 2)
    major_labels = [f"{t/sfreq:.1f}" for t in major_ticks]
    minor_step = int(0.2 * sfreq)
    minor_ticks = np.arange(0, n_time, minor_step)
    minor_labels = [f"{t/sfreq:.1f}" for t in minor_ticks]

    # Plot the individual segments
    # ------------------------------
    for col in range(n_individual):
        ax = axes[col + 1]
        seg = segs[col]

        for ch_idx in range(n_channels):
            ax.plot(time, -seg[ch_idx] + offsets[ch_idx],
                    color="black", linewidth=0.7)

        # Grid
        for t in major_ticks:
            ax.axvline(t, color="gray", alpha=0.5, linewidth=0.6)
        for t in minor_ticks:
            ax.axvline(t, color="gray", alpha=0.25, linewidth=0.4)

        # No y-ticks for individual plots
        ax.set_yticks([])
        ax.set_xlim(0, n_time)
        ax.set_ylim(-spacing, offsets.max() + spacing)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_labels)
        ax.set_xlabel("Time (s)")

        for y in offsets:
            ax.axhline(y, color="gray", alpha=0.25, linewidth=0.4)

    # Plot the averaged segment in first axis
    ax_avg = axes[0]
    for ch_idx in range(n_channels):
        ax_avg.plot(time, -avg_seg[ch_idx] + offsets[ch_idx], color="black", linewidth=0.8)
    # Channel labels
    for ch_idx, y in enumerate(offsets):
        ax_avg.text(-n_time*0.05, y, channel_names[ch_idx],
                va="center", ha="right", fontsize=7)
    # Grid lines
    for t in major_ticks:
        ax_avg.axvline(t, color="gray", alpha=0.5, linewidth=0.6)
    for t in minor_ticks:
        ax_avg.axvline(t, color="gray", alpha=0.25, linewidth=0.4)
    # Formatting
    ax_avg.set_facecolor((0.95, 1.0, 1.0))
    ax_avg.set_title(f"Average of:\n{n_avg} random windows", fontsize=8)
    ax_avg.set_xlim(0, n_time)
    ax_avg.set_ylim(-spacing, offsets.max() + spacing)
    ax_avg.set_xticks(major_ticks)
    ax_avg.set_xticklabels(major_labels)
    ax_avg.set_xlabel("Time (s)")
    # Set thicker border
    for spine in ax_avg.spines.values():
        spine.set_linewidth(1.5)
    # Zero-deviation lines
    for y in offsets:
        ax_avg.axhline(y, color="gray", alpha=0.25, linewidth=0.4)

    # Title
    fig.suptitle(
        f"{patient_id} — Avg + {n_individual} Individual Spike Segments",
        fontsize=14, fontweight='bold'
    )

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_mri_volume_slices(
    volume,
    title="MRI Volume",
    channel_names=None,
    slice_indices=None,
    figsize=(15, 10),
    output_path=None,
    cmap='gray',
    vmin=None,
    vmax=None
):
    """
    Plot multiple slices of a 3D or 4D MRI volume for visualization.
    
    Args:
        volume: numpy array of shape (C, D, H, W) or (D, H, W)
        title: Title for the figure
        channel_names: List of channel names (for multi-channel volumes)
        slice_indices: Dict with keys 'axial', 'sagittal', 'coronal' and lists of slice indices
                      If None, will show middle slices
        figsize: Figure size
        output_path: Path to save figure (if None, displays instead)
        cmap: Colormap to use
        vmin, vmax: Min/max values for colormap normalization
    """
    # Handle different input types
    if hasattr(volume, 'detach'):
        volume = volume.detach().cpu().numpy()
    
    # Handle 3D vs 4D volumes
    if volume.ndim == 3:
        volume = volume[np.newaxis, ...]  # Add channel dimension
        has_channels = False
    elif volume.ndim == 4:
        has_channels = True
    else:
        raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")
    
    C, D, H, W = volume.shape
    
    # Default channel names
    if channel_names is None:
        if C == 2:
            channel_names = ["T1", "FLAIR"]
        elif C == 3:
            channel_names = ["T1", "FLAIR", "Prior"]
        else:
            channel_names = [f"Ch{i}" for i in range(C)]
    
    # Default slice indices (show middle slices)
    if slice_indices is None:
        slice_indices = {
            'axial': [D // 4, D // 2, 3 * D // 4],
            'sagittal': [W // 4, W // 2, 3 * W // 4],
            'coronal': [H // 4, H // 2, 3 * H // 4],
        }
    
    # Create figure
    n_views = 3  # axial, sagittal, coronal
    n_slices_per_view = max(len(slice_indices['axial']), 
                            len(slice_indices['sagittal']), 
                            len(slice_indices['coronal']))
    
    fig, axes = plt.subplots(C * n_views, n_slices_per_view, figsize=figsize)
    
    # Make axes always 2D
    if C * n_views == 1:
        axes = axes.reshape(1, -1)
    elif n_slices_per_view == 1:
        axes = axes.reshape(-1, 1)
    
    for c in range(C):
        # Axial slices (D axis)
        row_offset = c * n_views
        for i, slice_idx in enumerate(slice_indices['axial']):
            ax = axes[row_offset, i]
            img = volume[c, slice_idx, :, :]
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            if i == 0:
                ax.set_ylabel(f"{channel_names[c]}\nAxial", fontsize=10, fontweight='bold')
            ax.set_title(f"D={slice_idx}", fontsize=8)
            ax.axis('off')
        
        # Sagittal slices (W axis)
        for i, slice_idx in enumerate(slice_indices['sagittal']):
            ax = axes[row_offset + 1, i]
            img = volume[c, :, :, slice_idx]
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            if i == 0:
                ax.set_ylabel(f"{channel_names[c]}\nSagittal", fontsize=10, fontweight='bold')
            ax.set_title(f"W={slice_idx}", fontsize=8)
            ax.axis('off')
        
        # Coronal slices (H axis)
        for i, slice_idx in enumerate(slice_indices['coronal']):
            ax = axes[row_offset + 2, i]
            img = volume[c, :, slice_idx, :]
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            if i == 0:
                ax.set_ylabel(f"{channel_names[c]}\nCoronal", fontsize=10, fontweight='bold')
            ax.set_title(f"H={slice_idx}", fontsize=8)
            ax.axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_mri_augmentation_comparison(
    volume_before,
    volume_after,
    channel_names=None,
    slice_indices=None,
    figsize=(20, 8),
    output_path=None,
    label_before=None,
    label_after=None
):
    """
    Plot side-by-side comparison of MRI volume before and after augmentation.
    
    Args:
        volume_before: Original volume (C, D, H, W)
        volume_after: Augmented volume (C, D, H, W)
        channel_names: List of channel names
        slice_indices: Dict with 'axial', 'sagittal', 'coronal' slice indices
        figsize: Figure size
        output_path: Path to save figure
        label_before: Optional ground truth label before augmentation (D, H, W)
        label_after: Optional ground truth label after augmentation (D, H, W)
    """
    # Handle different input types
    if hasattr(volume_before, 'detach'):
        volume_before = volume_before.detach().cpu().numpy()
    if hasattr(volume_after, 'detach'):
        volume_after = volume_after.detach().cpu().numpy()
    
    # Handle labels
    has_labels = label_before is not None and label_after is not None
    if has_labels:
        if hasattr(label_before, 'detach'):
            label_before = label_before.detach().cpu().numpy()
        if hasattr(label_after, 'detach'):
            label_after = label_after.detach().cpu().numpy()
    
    C = volume_before.shape[0]
    D, H, W = volume_before.shape[1:]
    
    # Default channel names
    if channel_names is None:
        if C == 2:
            channel_names = ["T1", "FLAIR"]
        elif C == 3:
            channel_names = ["T1", "FLAIR", "Prior"]
        else:
            channel_names = [f"Ch{i}" for i in range(C)]
    
    # Default slice indices (middle slices only for comparison)
    if slice_indices is None:
        slice_indices = {
            'axial': [D // 2],
            'sagittal': [W // 2],
            'coronal': [H // 2],
        }
    
    # Create figure: (C + label_row) × 3 views × 2 (before/after)
    n_views = 3
    n_rows = C + (1 if has_labels else 0)
    fig, axes = plt.subplots(n_rows, n_views * 2, figsize=figsize)
    
    # Make axes always 2D
    if C == 1:
        axes = axes.reshape(1, -1)
    
    for c in range(C):
        # Axial view
        slice_idx = slice_indices['axial'][0]
        # Before
        ax = axes[c, 0] if C > 1 else axes[0]
        img = volume_before[c, slice_idx, :, :]
        ax.imshow(img, cmap='gray', origin='lower')
        if c == 0:
            ax.set_title("Axial - Before", fontsize=10, fontweight='bold')
        ax.set_ylabel(f"{channel_names[c]}", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # After
        ax = axes[c, 1] if C > 1 else axes[1]
        img = volume_after[c, slice_idx, :, :]
        ax.imshow(img, cmap='gray', origin='lower')
        if c == 0:
            ax.set_title("Axial - After", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Sagittal view
        slice_idx = slice_indices['sagittal'][0]
        # Before
        ax = axes[c, 2] if C > 1 else axes[2]
        img = volume_before[c, :, :, slice_idx]
        ax.imshow(img, cmap='gray', origin='lower')
        if c == 0:
            ax.set_title("Sagittal - Before", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # After
        ax = axes[c, 3] if C > 1 else axes[3]
        img = volume_after[c, :, :, slice_idx]
        ax.imshow(img, cmap='gray', origin='lower')
        if c == 0:
            ax.set_title("Sagittal - After", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Coronal view
        slice_idx = slice_indices['coronal'][0]
        # Before
        ax = axes[c, 4] if C > 1 else axes[4]
        img = volume_before[c, :, slice_idx, :]
        ax.imshow(img, cmap='gray', origin='lower')
        if c == 0:
            ax.set_title("Coronal - Before", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # After
        ax = axes[c, 5] if C > 1 else axes[5]
        img = volume_after[c, :, slice_idx, :]
        ax.imshow(img, cmap='gray', origin='lower')
        if c == 0:
            ax.set_title("Coronal - After", fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Add label row if provided
    if has_labels:
        label_row = C
        
        # Axial view
        slice_idx = slice_indices['axial'][0]
        # Before
        ax = axes[label_row, 0] if n_rows > 1 else axes[0]
        img = label_before[slice_idx, :, :]
        ax.imshow(img, cmap='jet', origin='lower', vmin=0, vmax=1, interpolation='nearest')
        ax.set_ylabel("Ground Truth", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # After
        ax = axes[label_row, 1] if n_rows > 1 else axes[1]
        img = label_after[slice_idx, :, :]
        ax.imshow(img, cmap='jet', origin='lower', vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
        
        # Sagittal view
        slice_idx = slice_indices['sagittal'][0]
        # Before
        ax = axes[label_row, 2] if n_rows > 1 else axes[2]
        img = label_before[:, :, slice_idx]
        ax.imshow(img, cmap='jet', origin='lower', vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
        
        # After
        ax = axes[label_row, 3] if n_rows > 1 else axes[3]
        img = label_after[:, :, slice_idx]
        ax.imshow(img, cmap='jet', origin='lower', vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
        
        # Coronal view
        slice_idx = slice_indices['coronal'][0]
        # Before
        ax = axes[label_row, 4] if n_rows > 1 else axes[4]
        img = label_before[:, slice_idx, :]
        ax.imshow(img, cmap='jet', origin='lower', vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
        
        # After
        ax = axes[label_row, 5] if n_rows > 1 else axes[5]
        img = label_after[:, slice_idx, :]
        ax.imshow(img, cmap='jet', origin='lower', vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
    
    fig.suptitle("MRI Augmentation Comparison" + (" (with Ground Truth)" if has_labels else ""), fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()