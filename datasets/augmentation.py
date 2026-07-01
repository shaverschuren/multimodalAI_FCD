"""
datasets/augmentation.py

Augmentation pipelines for EEG spike localisation and multimodal MRI + Prior segmentation.
"""

import numpy as np
import torch
import scipy.signal as sps
from scipy.ndimage import gaussian_filter, map_coordinates, affine_transform


class MRIWithPriorAugment:
    """
    Augmentation pipeline for multimodal MRI + Prior segmentation.

    Handles 3D volumes with shape (C, D, H, W) where:
    - C=2: [T1, FLAIR] - MRI only (without prior)
    - C=3: [T1, FLAIR, PRIOR] - MRI with spatial prior
    - The first 2 channels are MRI images
    - The optional third channel is a spatial prior map (Gaussian blob)

    Spatial transforms are applied consistently to ALL channels (and y).
    Intensity transforms are applied ONLY to MRI channels (0,1).
    """

    def __init__(
        self,
        # Spatial augmentations
        p_rotation=0.3,
        rotation_angles=(-20, 20),          # degrees

        p_scaling=0.3,
        scaling_range=(0.9, 1.1),

        p_flip=0.5,
        flip_axes=(0, 1, 2),                # D, H, W

        p_elastic=0.3,
        elastic_alpha=10,                   # deformation magnitude
        elastic_sigma=3,                    # smoothness

        # Intensity augmentations (MRI only)
        p_intensity_scale=0.5,
        intensity_scale_range=(0.85, 1.15),

        p_intensity_shift=0.5,
        intensity_shift_range=(-0.1, 0.1),

        p_gamma=0.3,
        gamma_range=(0.7, 1.3),

        p_gaussian_noise=0.3,
        noise_std=0.02,

        p_mri_heavy_noise=0.3,
        heavy_noise_std=0.5,

        p_gaussian_blur=0.2,
        blur_sigma_range=(0.25, 0.5),

        # Contrast augmentation
        p_contrast=0.3,
        contrast_range=(0.75, 1.25),

        # Brightness augmentation
        p_brightness=0.3,
        brightness_range=(-0.1, 0.1),

        # Smooth multiplicative bias field (MRI only)
        p_bias_field=0.0,
        bias_field_strength_range=(0.05, 0.25),
        bias_field_sigma_range=(24.0, 48.0),

        # Channel dropout (Doing only prior now, so channel 2) -> Set to 0 by default.
        p_channel_dropout=0.0,
        channel_dropout_range=(2, 2),

        # Interpolation / padding defaults
        img_interp_order=1,
        prior_interp_order=1,
        label_interp_order=0,
        img_mode="constant",
        label_mode="constant",
        label_cval=0,
    ):
        # Spatial
        self.p_rotation = p_rotation
        self.rotation_angles = rotation_angles

        self.p_scaling = p_scaling
        self.scaling_range = scaling_range

        self.p_flip = p_flip
        self.flip_axes = flip_axes

        self.p_elastic = p_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        # Intensity (MRI only)
        self.p_intensity_scale = p_intensity_scale
        self.intensity_scale_range = intensity_scale_range

        self.p_intensity_shift = p_intensity_shift
        self.intensity_shift_range = intensity_shift_range

        self.p_gamma = p_gamma
        self.gamma_range = gamma_range

        self.p_gaussian_noise = p_gaussian_noise
        self.noise_std = noise_std

        self.p_mri_heavy_noise = p_mri_heavy_noise
        self.heavy_noise_std = heavy_noise_std
        self.p_gaussian_blur = p_gaussian_blur
        self.blur_sigma_range = blur_sigma_range

        self.p_contrast = p_contrast
        self.contrast_range = contrast_range

        self.p_brightness = p_brightness
        self.brightness_range = brightness_range

        self.p_bias_field = p_bias_field
        self.bias_field_strength_range = bias_field_strength_range
        self.bias_field_sigma_range = bias_field_sigma_range
        
        self.p_channel_dropout = p_channel_dropout
        self.channel_dropout_range = channel_dropout_range

        # Warp settings
        self.img_interp_order = int(img_interp_order)
        self.prior_interp_order = int(prior_interp_order)
        self.label_interp_order = int(label_interp_order)
        self.img_mode = img_mode
        self.label_mode = label_mode
        self.label_cval = label_cval

        # Cache for elastic base grid per (D,H,W)
        self._elastic_grid_cache = {}  # key: (D,H,W) -> coords float32 shape (3,D,H,W)

    # ---------------------------
    # Spatial aug helpers
    # ---------------------------

    @staticmethod
    def _rotation_matrix_xyz(angles_deg):
        """Forward rotation matrix R = Rz @ Ry @ Rx (right-handed)."""
        ax, ay, az = np.deg2rad(angles_deg)

        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        cz, sz = np.cos(az), np.sin(az)

        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx, cx]], dtype=np.float32)

        Ry = np.array([[cy, 0, sy],
                       [0, 1, 0],
                       [-sy, 0, cy]], dtype=np.float32)

        Rz = np.array([[cz, -sz, 0],
                       [sz, cz, 0],
                       [0, 0, 1]], dtype=np.float32)

        return (Rz @ Ry @ Rx).astype(np.float32)

    @staticmethod
    def _center_offset(matrix_3x3, shape_dhw):
        """Offset so transform is around volume center: offset = c - M @ c."""
        D, H, W = shape_dhw
        center = np.array([(D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0], dtype=np.float32)
        offset = center - matrix_3x3 @ center
        return offset.astype(np.float32)

    def random_flip(self, x, y=None):
        """Randomly flip along one or more axes. Cheap (no interpolation)."""
        axes_to_flip = [ax for ax in self.flip_axes if np.random.rand() < 0.5]
        if not axes_to_flip:
            return (x, y) if y is not None else x

        # x is (C,D,H,W) => flip axes +1
        x = np.flip(x, axis=[ax + 1 for ax in axes_to_flip]).copy()
        if y is not None:
            y = np.flip(y, axis=axes_to_flip).copy()
            return x, y
        return x

    def random_affine_rotate_scale(self, x, y=None):
        """
        Apply ONE combined affine warp for rotation (+ optional scaling).
        This replaces 3 sequential rotates + separate scaling warp.
        """
        do_rot = (np.random.rand() < self.p_rotation)
        do_scl = (np.random.rand() < self.p_scaling)
        if not (do_rot or do_scl):
            return (x, y) if y is not None else x

        angles = [0.0, 0.0, 0.0]
        if do_rot:
            angles = [
                np.random.uniform(*self.rotation_angles),
                np.random.uniform(*self.rotation_angles),
                np.random.uniform(*self.rotation_angles),
            ]

        scale = 1.0
        if do_scl:
            scale = float(np.random.uniform(*self.scaling_range))

        # We need matrix that maps output coords -> input coords (SciPy convention):
        # For a desired forward rotation R (input -> rotated), sampling uses inverse: R^T.
        # For zoom scale s (zoom in if s>1), sampling uses 1/s.
        R = self._rotation_matrix_xyz(angles)          # forward
        M = (R.T / scale).astype(np.float32)           # output->input
        offset = self._center_offset(M, x.shape[1:])   # D,H,W

        C = x.shape[0]
        x_warp = np.empty_like(x)

        # If you have a prior channel (C==3), use same order as images by default
        for c in range(C):
            order = self.img_interp_order if c < 2 else self.prior_interp_order
            x_warp[c] = affine_transform(
                x[c],
                matrix=M,
                offset=offset,
                output_shape=x[c].shape,
                order=order,
                mode=self.img_mode,
                cval=0.0,
                prefilter=(order > 1),
            )

        if y is not None:
            y_warp = affine_transform(
                y,
                matrix=M,
                offset=offset,
                output_shape=y.shape,
                order=self.label_interp_order,
                mode=self.label_mode,
                cval=self.label_cval,
                prefilter=False,
            )
            return x_warp, y_warp

        return x_warp

    def _get_elastic_base_grid(self, shape_dhw):
        """Cache base grid coords for elastic deformation: (3,D,H,W) float32."""
        key = tuple(shape_dhw)
        grid = self._elastic_grid_cache.get(key)
        if grid is not None:
            return grid

        D, H, W = shape_dhw
        d = np.arange(D, dtype=np.float32)
        h = np.arange(H, dtype=np.float32)
        w = np.arange(W, dtype=np.float32)
        d_coords, h_coords, w_coords = np.meshgrid(d, h, w, indexing="ij")
        grid = np.stack([d_coords, h_coords, w_coords], axis=0).astype(np.float32)  # (3,D,H,W)
        self._elastic_grid_cache[key] = grid
        return grid

    def elastic_deformation(self, x, y=None):
        """
        Elastic deformation using cached base grid and non-flattened coordinates.
        """
        C, D, H, W = x.shape

        # Displacement fields (float32)
        # Note: using constant here is OK; you can try mode="reflect" for slightly different boundary behavior.
        dx = gaussian_filter((np.random.rand(D, H, W).astype(np.float32) * 2 - 1),
                             self.elastic_sigma, mode="constant", cval=0) * self.elastic_alpha
        dy = gaussian_filter((np.random.rand(D, H, W).astype(np.float32) * 2 - 1),
                             self.elastic_sigma, mode="constant", cval=0) * self.elastic_alpha
        dz = gaussian_filter((np.random.rand(D, H, W).astype(np.float32) * 2 - 1),
                             self.elastic_sigma, mode="constant", cval=0) * self.elastic_alpha

        base = self._get_elastic_base_grid((D, H, W))
        coords = np.empty_like(base)
        coords[0] = base[0] + dx
        coords[1] = base[1] + dy
        coords[2] = base[2] + dz

        x_def = np.empty_like(x)
        for c in range(C):
            order = self.img_interp_order if c < 2 else self.prior_interp_order
            x_def[c] = map_coordinates(
                x[c],
                coords,
                order=order,
                mode=self.img_mode,
                cval=0.0,
                prefilter=(order > 1),
            )

        if y is not None:
            y_def = map_coordinates(
                y,
                coords,
                order=self.label_interp_order,
                mode=self.label_mode,
                cval=self.label_cval,
                prefilter=False,
            )
            return x_def, y_def

        return x_def

    # ---------------------------
    # Intensity augmentations (MRI only, channels 0 and 1)
    # Implemented to minimize full-volume copies.
    # ---------------------------

    def apply_intensity_ops_inplace(self, x):
        """
        Apply all intensity ops in-place on x (assumed numpy float array).
        Only affects MRI channels 0,1.
        """
        # intensity scaling
        if np.random.rand() < self.p_intensity_scale:
            scale = np.random.uniform(*self.intensity_scale_range)
            x[:2] *= scale

        # intensity shift
        if np.random.rand() < self.p_intensity_shift:
            shift = np.random.uniform(*self.intensity_shift_range)
            x[:2] += shift

        # gamma
        if np.random.rand() < self.p_gamma:
            gamma = np.random.uniform(*self.gamma_range)
            for c in range(2):
                chan = x[c]
                chan_min = float(chan.min())
                chan_max = float(chan.max())
                if chan_max > chan_min:
                    # normalize -> gamma -> denormalize
                    inv_range = 1.0 / (chan_max - chan_min)
                    chan_norm = (chan - chan_min) * inv_range
                    # ensure numeric stability
                    chan_norm = np.clip(chan_norm, 0.0, 1.0, out=chan_norm)
                    np.power(chan_norm, gamma, out=chan_norm)
                    x[c] = chan_norm * (chan_max - chan_min) + chan_min

        # gaussian noise
        if np.random.rand() < self.p_gaussian_noise:
            noise = np.random.normal(0.0, self.noise_std, size=x[:2].shape).astype(x.dtype, copy=False)
            x[:2] += noise

        # heavy gaussian noise on MRI channels to reduce over-reliance on MRI appearance.
        if np.random.rand() < self.p_mri_heavy_noise:
            heavy_noise = np.random.normal(0.0, self.heavy_noise_std, size=x[:2].shape).astype(x.dtype, copy=False)
            x[:2] += heavy_noise

        # gaussian blur
        if np.random.rand() < self.p_gaussian_blur:
            sigma = np.random.uniform(*self.blur_sigma_range)
            # gaussian_filter allocates output; write back channel-wise
            x[0] = gaussian_filter(x[0], sigma=sigma)
            x[1] = gaussian_filter(x[1], sigma=sigma)

        # contrast
        if np.random.rand() < self.p_contrast:
            factor = np.random.uniform(*self.contrast_range)
            for c in range(2):
                mean = x[c].mean()
                x[c] = (x[c] - mean) * factor + mean

        # brightness
        if np.random.rand() < self.p_brightness:
            b = np.random.uniform(*self.brightness_range)
            x[:2] += b

        # smooth multiplicative bias field (same field for both MRI channels)
        if np.random.rand() < self.p_bias_field:
            _, d, h, w = x.shape
            sigma = float(np.random.uniform(*self.bias_field_sigma_range))
            strength = float(np.random.uniform(*self.bias_field_strength_range))

            field = np.random.randn(d, h, w).astype(np.float32)
            field = gaussian_filter(field, sigma=sigma)
            max_abs = float(np.max(np.abs(field)))
            if max_abs > 1e-6:
                field /= max_abs
            bias = np.exp(strength * field).astype(x.dtype, copy=False)
            x[:2] *= bias[None, ...]

        return x

    # ---------------------------
    # Main pipeline
    # ---------------------------

    def __call__(self, x, y=None):
        """
        Args:
            x: [C, D, H, W] torch.Tensor or np.ndarray
            y: [D, H, W] torch.Tensor or np.ndarray (optional)

        Returns:
            x_aug: augmented volume
            y_aug: augmented mask (if provided)
        """
        # Convert to numpy if torch tensor
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            device = x.device
            x_dtype_torch = x.dtype
            x = x.detach().cpu().numpy()
            y_dtype_torch = None
            if y is not None:
                y_dtype_torch = y.dtype
                y = y.detach().cpu().numpy()

        # Ensure float for interpolation/intensity
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float32, copy=False)
        else:
            # keep float32 if possible to reduce bandwidth
            if x.dtype != np.float32:
                x = x.astype(np.float32, copy=False)

        # --- Spatial augmentations ---
        if np.random.rand() < self.p_flip:
            if y is not None:
                x, y = self.random_flip(x, y)
            else:
                x = self.random_flip(x)

        # Combine rotation + scaling into one affine warp
        if y is not None:
            x, y = self.random_affine_rotate_scale(x, y)
        else:
            x = self.random_affine_rotate_scale(x)

        # Elastic (separate warp)
        if np.random.rand() < self.p_elastic:
            if y is not None:
                x, y = self.elastic_deformation(x, y)
            else:
                x = self.elastic_deformation(x)

        # --- Intensity augmentations (in-place, MRI channels only) ---
        self.apply_intensity_ops_inplace(x)

        # Channel dropout (after spatial + intensity to avoid weird interpolation artifacts on dropped channels)
        if np.random.rand() < self.p_channel_dropout:
            n_channels = x.shape[0]
            n_drop = np.random.randint(*self.channel_dropout_range)
            if n_drop > 0 and n_drop < n_channels:
                drop_idx = np.random.choice(n_channels, n_drop, replace=False)
                x[drop_idx] = 0.0  # zero out dropped channels

        # Convert back to torch if needed
        if is_torch:
            x_out = torch.from_numpy(x).to(device=device, dtype=x_dtype_torch)
            if y is not None:
                # Preserve mask dtype if you want (commonly long/int64)
                y_np = y
                if np.issubdtype(y_np.dtype, np.floating):
                    # if labels accidentally became float, cast back to int
                    y_np = y_np.astype(np.int64, copy=False)
                y_out = torch.from_numpy(y_np).to(device=device, dtype=y_dtype_torch or torch.long)
                return x_out, y_out
            return x_out

        return (x, y) if y is not None else x


class EEGSpikeAugment:
    """
    EEG augmentations to disrupt patient-specific features
    without harming localization-relevant spike morphology.
    """

    def __init__(
        self,
        fs=256,

        # Time-domain perturbations
        p_time_jitter=1.0,
        max_time_jitter=5,

        p_time_warp=0.0,
        time_warp_strength=0.025,   # moderate morph stretch <- skipping because afraid of distorting morphology

        # Light masking that avoids spike center
        p_time_mask=0.,
        time_mask_width=32,         # small patches only
        mask_exclude_center=True,

        # Amplitude transforms
        p_amp_scaling=1.0,
        amp_scale_range=(0.85, 1.15),

        # Channel dropout
        p_channel_dropout=0.5,
        channel_dropout_ratio=0.05,    # <= 15% channels

        # Noise models
        p_gaussian_noise=1.0,
        noise_std=0.05,

        p_band_limited_noise=0.1,
        band_limited_noise_amplitude_range=(0.1, 0.4),
        band_limited_noise_f0_range=(20, 100),

        p_massive_noise=0.1,           # occasional extreme noise
        massive_noise_std_range=(0.5, 1.5),

        p_single_channel_massive_noise=0.1,  # extreme noise on a single random channel
        single_channel_massive_noise_std_range=(0.5, 1.5),

        # Frequency-domain distortions
        p_freq_dropout=0.,
        freq_bandwidth=3,              # narrow notches
        num_freq_bands=2,
    ):
        self.fs = fs

        self.p_massive_noise = p_massive_noise
        self.massive_noise_std_range = massive_noise_std_range

        self.p_single_channel_massive_noise = p_single_channel_massive_noise
        self.single_channel_massive_noise_std_range = single_channel_massive_noise_std_range

        self.p_time_jitter = p_time_jitter
        self.max_time_jitter = max_time_jitter

        self.p_time_warp = p_time_warp
        self.time_warp_strength = time_warp_strength

        self.p_time_mask = p_time_mask
        self.time_mask_width = time_mask_width
        self.mask_exclude_center = mask_exclude_center

        self.p_amp_scaling = p_amp_scaling
        self.amp_scale_range = amp_scale_range

        self.p_channel_dropout = p_channel_dropout
        self.channel_dropout_ratio = channel_dropout_ratio

        self.p_gaussian_noise = p_gaussian_noise
        self.noise_std = noise_std

        self.p_band_limited_noise = p_band_limited_noise
        self.band_limited_noise_amplitude_range = band_limited_noise_amplitude_range
        self.band_limited_noise_f0_range = band_limited_noise_f0_range

        self.p_freq_dropout = p_freq_dropout
        self.freq_bandwidth = freq_bandwidth
        self.num_freq_bands = num_freq_bands

    # Augmentations
    def time_jitter(self, x):
        """Shift waveform without distorting morphology."""
        shift = np.random.randint(-self.max_time_jitter, self.max_time_jitter + 1)
        return np.roll(x, shift, axis=1)

    def time_warp(self, x):
        """Controlled small warp — preserves shape."""
        C, L = x.shape
        factor = 1 + np.random.uniform(-self.time_warp_strength, self.time_warp_strength)

        t = np.linspace(0, 1, L)
        t_new = np.linspace(0, 1, int(L * factor))

        x_out = np.zeros((C, L))
        for c in range(C):
            stretched = np.interp(t_new, t, x[c])
            x_out[c] = np.interp(t, np.linspace(0, 1, len(stretched)), stretched)

        return x_out

    def time_mask(self, x):
        """Mask a small region *away from spike center*."""
        C, L = x.shape
        w = min(self.time_mask_width, L // 4)

        center = L // 2
        if self.mask_exclude_center:
            # choose a location at least ~64 samples from center
            valid_starts = list(range(0, center - 64 - w)) + list(range(center + 64, L - w))
            if len(valid_starts) == 0:
                return x
            start = np.random.choice(valid_starts)
        else:
            start = np.random.randint(0, L - w)

        x = x.copy()
        x[:, start:start + w] = 0
        return x

    def amplitude_scaling(self, x):
        scale = np.random.uniform(*self.amp_scale_range)
        return x * scale

    def channel_dropout(self, x):
        """Drop a few channels."""
        C, L = x.shape
        n_drop = max(1, int(C * self.channel_dropout_ratio))
        drop_idx = np.random.choice(C, n_drop, replace=False)
        x = x.copy()
        x[drop_idx] = 0
        return x

    def massive_noise(self, x):
        std = np.random.uniform(*self.massive_noise_std_range)
        return x + np.random.normal(0, std, x.shape)
    
    def single_channel_massive_noise(self, x):
        C, L = x.shape
        ch = np.random.randint(C)
        std = np.random.uniform(*self.single_channel_massive_noise_std_range)
        x = x.copy()
        x[ch] += np.random.normal(0, std, L)
        return x

    def gaussian_noise(self, x):
        return x + np.random.normal(0, self.noise_std, x.shape)

    def band_limited_noise(self, x, amplitude_range=(0.1, 0.4), f0_range=(20, 100)):
        """Inject smooth sinusoidal noise in a random frequency band."""
        C, L = x.shape
        t = np.arange(L) / self.fs
        f0 = np.random.uniform(*f0_range)
        amplitude = np.random.uniform(*amplitude_range)
        noise = np.sin(2 * np.pi * f0 * t)
        noise = sps.lfilter([1], [1], noise)
        return x + amplitude * noise.reshape(1, -1)

    def frequency_dropout(self, x):
        C, L = x.shape
        X = np.fft.rfft(x, axis=1)
        freqs = np.fft.rfftfreq(L, 1 / self.fs)

        for _ in range(self.num_freq_bands):
            f0 = np.random.uniform(2, self.fs // 2 - self.freq_bandwidth)
            mask = (freqs >= f0) & (freqs <= f0 + self.freq_bandwidth)
            X[:, mask] = 0

        return np.fft.irfft(X, n=L, axis=1)

    # Main pipeline
    def __call__(self, x):

        if np.random.rand() < self.p_massive_noise:
            x = self.massive_noise(x)
            return x  # skip other augments if extreme noise applied

        if np.random.rand() < self.p_single_channel_massive_noise:
            x = self.single_channel_massive_noise(x)

        if np.random.rand() < self.p_time_jitter:
            x = self.time_jitter(x)

        if np.random.rand() < self.p_time_warp:
            x = self.time_warp(x)

        if np.random.rand() < self.p_time_mask:
            x = self.time_mask(x)

        if np.random.rand() < self.p_amp_scaling:
            x = self.amplitude_scaling(x)

        if np.random.rand() < self.p_channel_dropout:
            x = self.channel_dropout(x)

        if np.random.rand() < self.p_gaussian_noise:
            x = self.gaussian_noise(x)

        if np.random.rand() < self.p_band_limited_noise:
            x = self.band_limited_noise(x, f0_range=self.band_limited_noise_f0_range)

        if np.random.rand() < self.p_freq_dropout:
            x = self.frequency_dropout(x)

        return x


if __name__ == "__main__":
    """
    Test MRI augmentation by loading a patient's .npz file and visualizing
    a 128x128x128 patch before and after augmentation. Doing this for both
    3-channel (MRI + Prior) and 2-channel (MRI only) inputs.
    """
    import torch
    import sys
    import os
    import time
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from util.debug import plot_mri_augmentation_comparison
    from datasets.multimodal import gaussian_prior_ijk, norm_to_mm, mm_to_vox, sigma_mm_to_vox
    
    print("=" * 80)
    print("MRI Augmentation Test")
    print("=" * 80)
    
    # Configuration
    patient_npz_path = r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\preprocessing\mri\RESP1358\RESP1358_preproc.npz"
    
    # Example prior prediction (from EEG model)
    prior_pred = {"mu": [-0.6324, -0.067, 0.477119], "sigma": [0.0827, 0.0675, 0.1521]}
    
    output_dir = r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\tmp\augmentation_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load patient data
    print(f"\nLoading patient data from: {patient_npz_path}")
    data = np.load(patient_npz_path)
    mri_image = data["image"]  # Shape: (2, D, H, W) - T1 and FLAIR
    affine = data["affine"]
    
    # Load ground truth if available
    has_gt = "gt" in data
    if has_gt:
        gt_label = data["gt"]  # Shape: (D, H, W)
        print(f"  Ground truth found: shape={gt_label.shape}, unique values={np.unique(gt_label)}")
    else:
        gt_label = None
        print(f"  No ground truth available in this dataset")
    
    print(f"  MRI shape: {mri_image.shape}")
    print(f"  MRI dtype: {mri_image.dtype}")
    print(f"  MRI range: [{mri_image.min():.3f}, {mri_image.max():.3f}]")
    
    # Generate prior channel
    print(f"\nGenerating prior from prediction: mu={prior_pred['mu']}, sigma={prior_pred['sigma']}")
    mu_mm, sig_mm = norm_to_mm(prior_pred["mu"], prior_pred["sigma"])
    mu_ijk = mm_to_vox(mu_mm, affine)
    sig_ijk = sigma_mm_to_vox(sig_mm, affine)
    
    prior = gaussian_prior_ijk(mri_image.shape[1:], mu_ijk, sig_ijk)
    print(f"  Prior shape: {prior.shape}")
    print(f"  Prior range: [{prior.min():.3f}, {prior.max():.3f}]")
    
    # Stack MRI + prior
    volume_3ch_before = np.concatenate([mri_image, prior[np.newaxis, ...]], axis=0)  # (3, D, H, W)
    print(f"\nFull volume shape: {volume_3ch_before.shape}")
    
    # Initialize augmentation
    print("\nInitializing MRIWithPriorAugment...")
    augmentor = MRIWithPriorAugment(
        p_rotation=1.0,          # Always apply everything for visualization
        p_scaling=1.0,
        p_flip=1.0,
        p_elastic=1.0,
        p_intensity_scale=1.0,
        p_gamma=1.0,
        p_gaussian_noise=1.0,
        p_gaussian_blur=1.0,
        p_contrast=1.0,
    )
    
    # Apply augmentation to FULL volume first (as done in dataset)
    print("\nApplying augmentation to full volume...")
    t0 = time.time()
    if has_gt:
        volume_3ch_after, gt_label_after = augmentor(volume_3ch_before.copy(), gt_label.copy())
        print(f"Augmented volume shape: {volume_3ch_after.shape}")
        print(f"Augmented label shape: {gt_label_after.shape}")
    else:
        volume_3ch_after = augmentor(volume_3ch_before.copy())
        gt_label_after = None
        print(f"Augmented volume shape: {volume_3ch_after.shape}")
    t1 = time.time()
    print(f"3-channel augmentation took {t1 - t0:.3f} seconds")
    
    # Now extract patches from both before and after (for visualization)
    D, H, W = volume_3ch_before.shape[1:]
    patch_size = 128
    
    d_start = (D - patch_size) // 2
    h_start = (H - patch_size) // 2
    w_start = (W - patch_size) // 2
    
    print(f"\nExtracting {patch_size}x{patch_size}x{patch_size} patches from center...")
    print(f"Patch location: D=[{d_start}:{d_start+patch_size}], H=[{h_start}:{h_start+patch_size}], W=[{w_start}:{w_start+patch_size}]")
    
    # Extract patches from BEFORE
    patch_before = volume_3ch_before[
        :,
        d_start:d_start + patch_size,
        h_start:h_start + patch_size,
        w_start:w_start + patch_size
    ].astype(np.float32)
    
    if has_gt:
        label_patch_before = gt_label[
            d_start:d_start + patch_size,
            h_start:h_start + patch_size,
            w_start:w_start + patch_size
        ].astype(np.float32)
    else:
        label_patch_before = None
    
    # Extract patches from AFTER
    patch_after = volume_3ch_after[
        :,
        d_start:d_start + patch_size,
        h_start:h_start + patch_size,
        w_start:w_start + patch_size
    ].astype(np.float32)
    
    if has_gt:
        label_patch_after = gt_label_after[
            d_start:d_start + patch_size,
            h_start:h_start + patch_size,
            w_start:w_start + patch_size
        ].astype(np.float32)
    else:
        label_patch_after = None
    
    print(f"Extracted patch shape: {patch_before.shape}")
    print(f"Augmented patch shape: {patch_after.shape}")
    print(f"Augmented patch range: [{patch_after.min():.3f}, {patch_after.max():.3f}]")
    
    # Visualize comparison
    output_path_3ch = os.path.join(output_dir, "augmentation_comparison_3ch.png")
    plot_mri_augmentation_comparison(
        volume_before=patch_before,
        volume_after=patch_after,
        channel_names=["T1", "FLAIR", "PRIOR"],
        figsize=(20, 10 if has_gt else 8),
        output_path=output_path_3ch,
        label_before=label_patch_before if has_gt else None,
        label_after=label_patch_after if has_gt else None
    )
    print(f"\n3-channel visualization saved to: {output_path_3ch}")

    # Also test 2-channel input (MRI only, no prior)
    print("\n" + "=" * 80)
    print("Testing with 2-channel input (MRI only, no prior)")
    print("=" * 80)
    t0 = time.time()
    if has_gt:
        mri_image_after, gt_label_2ch_after = augmentor(mri_image.copy(), gt_label.copy())
        print(f"Augmented 2-channel volume shape: {mri_image_after.shape}")
        print(f"Augmented label shape: {gt_label_2ch_after.shape}")
    else:
        mri_image_after = augmentor(mri_image.copy())
        gt_label_2ch_after = None
        print(f"Augmented 2-channel volume shape: {mri_image_after.shape}")
    t1 = time.time()
    print(f"2-channel augmentation took {t1 - t0:.3f} seconds")
    
    # Extract patches from before
    patch_2ch_before = mri_image[
        :,
        d_start:d_start + patch_size,
        h_start:h_start + patch_size,
        w_start:w_start + patch_size
    ].astype(np.float32)
    
    # Extract patches from after
    patch_2ch_after = mri_image_after[
        :,
        d_start:d_start + patch_size,
        h_start:h_start + patch_size,
        w_start:w_start + patch_size
    ].astype(np.float32)
    
    if has_gt:
        label_2ch_after = gt_label_2ch_after[
            d_start:d_start + patch_size,
            h_start:h_start + patch_size,
            w_start:w_start + patch_size
        ].astype(np.float32)
    else:
        label_2ch_after = None
    
    print(f"Extracted 2-channel patch shape: {patch_2ch_before.shape}")
    
    # Visualize 2-channel comparison
    output_path_2ch = os.path.join(output_dir, "augmentation_comparison_2ch.png")
    plot_mri_augmentation_comparison(
        volume_before=patch_2ch_before,
        volume_after=patch_2ch_after,
        channel_names=["T1", "FLAIR"],
        figsize=(20, 8 if has_gt else 6),
        output_path=output_path_2ch,
        label_before=label_patch_before if has_gt else None,
        label_after=label_2ch_after if has_gt else None
    )
    print(f"2-channel visualization saved to: {output_path_2ch}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
