"""
multimodal.py
Multimodal dataset utilities: MRI + PRIOR

Implements dataset and data loading utilities for multimodal MRI + PRIOR data.
The PRIOR is a spatial prior map (Gaussian blob) derived from the output of the EEG source localization model.

Will likely be extended later to support further EEG feature fusion, next to simply a prior map.
"""

import os
import warnings
import numpy as np
import torch
import nibabel as nib
from pathlib import Path

from datasets.augmentation import EEGSpikeAugment, MRIWithPriorAugment

# Constants (hardcoded from preprocessing pipeline)
EEG_MNI_EXTENT_MM = np.array([90.0, 126.0, 72.0], dtype=np.float32)   # x,y,z half-ranges in mm
MRI_CROP_SHAPE = np.array([160, 192, 160], dtype=np.int32)            # D,H,W after preprocessing
MRI_VOX_SIZE_MM = np.array([1.0, 1.0, 1.0], dtype=np.float32)         # voxel size in mm


def norm_to_mm(mu_norm_xyz, sigma_norm_xyz, extent_mm=EEG_MNI_EXTENT_MM):
    """Convert normalized MNI coordinates to mm."""
    mu_mm = np.asarray(mu_norm_xyz, dtype=np.float32) * extent_mm
    sig_mm = np.asarray(sigma_norm_xyz, dtype=np.float32) * extent_mm
    return mu_mm, sig_mm

def mm_to_vox(mu_mm_xyz, affine):
    """
    Map (x,y,z) in mm (MNI space) to voxel indices (i,j,k) using the affine matrix.

    Parameters
    ----------
    mu_mm_xyz : array-like (3,)
        Coordinates in mm (x, y, z)
    affine : ndarray (4,4)
        Affine matrix from voxel -> world (MNI) space

    Returns
    -------
    mu_ijk : ndarray (3,)
        Voxel indices (i, j, k)
    """
    affine_inv = np.linalg.inv(affine)
    mm_h = np.array([mu_mm_xyz[0], mu_mm_xyz[1], mu_mm_xyz[2], 1.0], dtype=np.float32)
    ijk_h = affine_inv @ mm_h
    return ijk_h[:3].astype(np.float32)

def sigma_mm_to_vox(sig_mm_xyz, affine):
    """
    Convert sigma in mm to sigma in voxel space using the affine matrix.

    Parameters
    ----------
    sig_mm_xyz : array-like (3,)
        Standard deviations in mm
    affine : ndarray (4,4)
        Affine matrix from voxel -> world (MNI) space

    Returns
    -------
    sig_ijk : ndarray (3,)
        Standard deviations in voxel space
    """
    R = affine[:3, :3]
    vox_size_mm = np.sqrt((R ** 2).sum(axis=0))
    sig_mm = np.asarray(sig_mm_xyz, dtype=np.float32)
    return sig_mm / vox_size_mm.astype(np.float32)

def gaussian_prior_ijk(shape, mu_ijk, sig_ijk, clamp_min_vox=1.0, eps=1e-6):
    """
    Generate a 3D Gaussian prior map in voxel space.

    Parameters
    ----------
    shape : tuple (D, H, W)
        Shape of the output prior
    mu_ijk : array-like (3,)
        Mean in voxel coordinates
    sig_ijk : array-like (3,)
        Standard deviation in voxel coordinates
    clamp_min_vox : float
        Minimum sigma value (in voxels)
    eps : float
        Small epsilon to avoid division by zero

    Returns
    -------
    prior : ndarray (D, H, W)
        Normalized Gaussian prior [0, 1]
    """
    D, H, W = map(int, shape)
    mu = np.asarray(mu_ijk, dtype=np.float32)
    sig = np.maximum(np.asarray(sig_ijk, dtype=np.float32), float(clamp_min_vox))

    i = np.arange(D, dtype=np.float32)
    j = np.arange(H, dtype=np.float32)
    k = np.arange(W, dtype=np.float32)

    ei = (i - mu[0]) ** 2 / (sig[0] ** 2 + eps)
    ej = (j - mu[1]) ** 2 / (sig[1] ** 2 + eps)
    ek = (k - mu[2]) ** 2 / (sig[2] ** 2 + eps)

    dist = ei[:, None, None] + ej[None, :, None] + ek[None, None, :]
    prior = np.exp(-0.5 * dist).astype(np.float32)
    
    m = float(prior.max())
    if m > 0:
        prior /= m
    
    return prior


class UNetWithPriorDataset(torch.utils.data.Dataset):
    """
    Dataset that loads MRI images and generates Gaussian priors from EEG predictions.

    Pre-loads all images, GTs, and priors into shared memory for efficient multi-worker
    data loading. Each sample combines T1w + FLAIR + prior as (3, D, H, W).

    Supports patch sampling to extract fixed-size patches (default 128x128x128) centered
    at the prior peak or random locations.

    Priors can be cached to disk (.npy and .nii.gz) for faster re-initialization.

    Parameters
    ----------
    cases : list of dict
        List of cases with keys: {"id": str, "npy": path}
    eeg_preds_by_id : dict
        Maps patient ID -> {"mu": [x,y,z] normalized, "sigma": float or [sx,sy,sz]}
    image_dtype : torch.dtype
        Data type for storing images
    prior_dtype : torch.dtype
        Data type for storing priors
    gt_dtype : torch.dtype
        Data type for storing GT masks
    clamp_min_sigma_vox : float
        Minimum sigma value in voxels when generating priors
    fallback_prior : str
        How to fill missing priors: "zeros" or "ones"
    return_float32 : bool
        If True, convert to float32 on __getitem__
    prior_cache_dir : str, optional
        Directory to cache priors as .npy and .nii.gz files. If None, no caching.
    overwrite_cache : bool
        If True, recompute and overwrite cached priors. If False, load from cache if available.
    patch_size : int
        Size of the patch in each dimension (default 128)
    enable_patch_sampling : bool
        If True, extract fixed-size patches from data. If False, return full volumes.
    patch_center_mode : str
        How to select patch center: "prior_peak" (default) to center at prior maximum,
        or "random" to sample random centers.
    enable_augmentation : bool
        If True, apply data augmentation (typically for training only).
    augmentation_params : dict, optional
        Parameters for MRIWithPriorAugment. If None, uses default parameters.
    """

    def __init__(
        self,
        cases,
        eeg_preds_by_id,
        image_dtype=torch.float16,
        prior_dtype=torch.float16,
        gt_dtype=torch.uint8,
        clamp_min_sigma_vox=1.0,
        fallback_prior="zeros",
        return_float32=True,
        prior_cache_dir=None,
        overwrite_cache=False,
        patch_size=128,
        enable_patch_sampling=False,
        patch_center_mode="random",
        enable_augmentation=False,
        augmentation_params=None,
    ):
        super().__init__()
        assert fallback_prior in ("ones", "zeros"), f"fallback_prior must be 'ones' or 'zeros', got {fallback_prior}"
        assert patch_center_mode in ("prior_peak", "random"), f"patch_center_mode must be 'prior_peak' or 'random', got {patch_center_mode}"

        self.cases = cases
        self.eeg_preds_by_id = eeg_preds_by_id
        self.return_float32 = return_float32
        self.id_to_index = {c["id"]: i for i, c in enumerate(cases)}
        self.prior_cache_dir = prior_cache_dir
        self.overwrite_cache = overwrite_cache
        self.patch_size = patch_size
        self.enable_patch_sampling = enable_patch_sampling
        self.patch_center_mode = patch_center_mode
        
        # Setup augmentation pipeline
        self.augment = None
        if enable_augmentation:
            aug_params = augmentation_params or {}
            # Default augmentation parameters
            default_params = dict(
                p_rotation=0.3,
                rotation_angles=(-15, 15),
                p_scaling=0.3,
                scaling_range=(0.9, 1.1),
                p_flip=0.5,
                p_elastic=0.2,
                elastic_alpha=10,
                elastic_sigma=3,
                p_intensity_scale=0.5,
                intensity_scale_range=(0.85, 1.15),
                p_gamma=0.3,
                gamma_range=(0.7, 1.3),
                p_gaussian_noise=0.3,
                noise_std=0.02,
                p_mri_heavy_noise=0.3,
                p_gaussian_blur=0.2,
                blur_sigma_range=(0.5, 1.0),
                p_contrast=0.3,
                contrast_range=(0.75, 1.25),
                p_channel_dropout=0.2,
                channel_dropout_range=(2, 2),  # Only dropout the prior channel
            )
            # Override defaults with user-provided params
            default_params.update(aug_params)
            self.augment = MRIWithPriorAugment(**default_params)

        # Create cache directory if needed
        if self.prior_cache_dir is not None:
            os.makedirs(self.prior_cache_dir, exist_ok=True)

        # Infer shape and store affine from first case
        npz0 = np.load(cases[0]["npy"], allow_pickle=True)
        arr0 = npz0["image"]
        self.affine = npz0["affine"].astype(np.float32)
        assert arr0.ndim == 4 and arr0.shape[0] == 2, f"Expected (2,D,H,W), got {arr0.shape}"
        _, D, H, W = arr0.shape
        npz0.close()
        del arr0

        # Pre-allocate shared tensors
        N = len(cases)
        self.images = torch.empty((N, 2, D, H, W), dtype=image_dtype)
        self.priors = torch.empty((N, D, H, W), dtype=prior_dtype)
        self.gts = torch.empty((N, D, H, W), dtype=gt_dtype)

        # Load all data and generate/load priors
        self._load_all_data(D, H, W, clamp_min_sigma_vox, fallback_prior)

        # Share memory for multi-worker dataloading
        self.images.share_memory_()
        self.priors.share_memory_()
        self.gts.share_memory_()

    def _load_all_data(self, D, H, W, clamp_min_sigma_vox, fallback_prior):
        """Load images, GTs, and generate/load priors for all cases."""
        for i, c in enumerate(self.cases):
            try:
                npz = np.load(c["npy"], allow_pickle=True)
                img = npz["image"].astype(np.float32)
                gt = npz["gt"].astype(np.uint8)
                npz.close()
            except Exception as e:
                warnings.warn(f"Error loading case {c['id']} from {c['npy']}: {e}")
                continue

            self.images[i].copy_(torch.from_numpy(img).to(self.images.dtype))
            self.gts[i].copy_(torch.from_numpy(gt).to(self.gts.dtype))

            # Generate or load prior from cache
            prior_np = self._get_prior(
                c["id"], D, H, W, img, clamp_min_sigma_vox, fallback_prior
            )
            self.priors[i].copy_(torch.from_numpy(prior_np).to(self.priors.dtype))

    def _get_cache_paths(self, patient_id):
        """Return paths for cached prior files (.npy and .nii.gz)."""
        if self.prior_cache_dir is None:
            return None, None
        
        npy_path = os.path.join(self.prior_cache_dir, f"{patient_id}_prior.npy")
        nii_path = os.path.join(self.prior_cache_dir, f"{patient_id}_prior.nii.gz")
        return npy_path, nii_path

    def _load_prior_from_cache(self, patient_id):
        """Load prior from cached .npy file if it exists."""
        if self.prior_cache_dir is None or self.overwrite_cache:
            return None
        
        npy_path, _ = self._get_cache_paths(patient_id)
        if npy_path and os.path.exists(npy_path):
            return np.load(npy_path).astype(np.float32)
        return None

    def _save_prior_to_cache(self, prior_np, patient_id):
        """Save prior to both .npy and .nii.gz formats."""
        if self.prior_cache_dir is None:
            return
        
        npy_path, nii_path = self._get_cache_paths(patient_id)
        
        # Save .npy
        np.save(npy_path, prior_np)
        
        # Save .nii.gz
        img = nib.Nifti1Image(prior_np, self.affine)
        nib.save(img, nii_path)

    def _get_prior(self, patient_id, D, H, W, img, clamp_min_sigma_vox, fallback_prior):
        """Get prior: load from cache or generate and cache."""
        # Try to load from cache first
        prior_np = self._load_prior_from_cache(patient_id)
        if prior_np is not None:
            return prior_np
        
        # Generate prior
        prior_np = self._generate_prior(
            patient_id, D, H, W, img, clamp_min_sigma_vox, fallback_prior
        )
        
        # Save to cache
        self._save_prior_to_cache(prior_np, patient_id)
        
        return prior_np

    def _generate_prior(self, patient_id, D, H, W, img, clamp_min_sigma_vox, fallback_prior):
        """Generate Gaussian prior for a single patient."""
        pred = self.eeg_preds_by_id.get(patient_id, None)

        if pred is None:
            if fallback_prior == "ones":
                return np.ones((D, H, W), np.float32)
            else:
                return np.zeros((D, H, W), np.float32)

        # Convert EEG prediction (normalized MNI) to voxel space
        mu_norm = np.asarray(pred["mu"], dtype=np.float32)
        sigma = pred["sigma"]
        sigma_norm = (
            np.array([float(sigma)] * 3, dtype=np.float32)
            if np.isscalar(sigma)
            else np.asarray(sigma, dtype=np.float32)
        )

        mu_mm, sig_mm = norm_to_mm(mu_norm, sigma_norm)
        mu_ijk = mm_to_vox(mu_mm, self.affine)
        sig_ijk = sigma_mm_to_vox(sig_mm, self.affine)

        prior = gaussian_prior_ijk(
            (D, H, W), mu_ijk, sig_ijk, clamp_min_vox=clamp_min_sigma_vox
        )
        # Mask prior with image TODO: verify if actually helps, might drop.
        img_mask = (img > 1e-5).all(axis=0).astype(np.float32)
        prior_masked = prior * img_mask
        return prior_masked

    def __len__(self):
        return len(self.cases)

    def _sample_patch_center(self, prior, gt):
        """
        Sample a patch center based on the selected mode.

        Parameters
        ----------
        prior : ndarray (D, H, W)
            Prior map
        gt : ndarray (D, H, W)
            Ground truth mask

        Returns
        -------
        center : ndarray (3,)
            Center coordinates (d, h, w) for patch sampling
        """
        if self.patch_center_mode == "prior_peak":
            # Find peak of prior
            center = np.array(np.unravel_index(np.argmax(prior), prior.shape), dtype=np.int32)
        elif self.patch_center_mode == "random":
            # Sample randomly within valid bounds, ensuring patch stays within image
            D, H, W = prior.shape
            half_patch = self.patch_size // 2
            center = np.array([
                np.random.randint(half_patch, D - half_patch),
                np.random.randint(half_patch, H - half_patch),
                np.random.randint(half_patch, W - half_patch)
            ], dtype=np.int32)
        else:
            raise ValueError(f"Unknown patch_center_mode: {self.patch_center_mode}")
        
        return center

    def _extract_patch(self, data, center, patch_size):
        """
        Extract a patch from 3D data centered at the given location.

        Parameters
        ----------
        data : ndarray (C, D, H, W) or (D, H, W)
            Input data
        center : ndarray (3,)
            Center coordinates (d, h, w)
        patch_size : int
            Size of patch in each dimension

        Returns
        -------
        patch : ndarray
            Extracted patch with same number of channels as input
        """
        is_3d = data.ndim == 3
        if is_3d:
            D, H, W = data.shape
            patch_data = data
        else:
            C, D, H, W = data.shape
            patch_data = data
        
        # Compute patch boundaries
        d_start = center[0] - patch_size // 2
        h_start = center[1] - patch_size // 2
        w_start = center[2] - patch_size // 2

        d_end = d_start + patch_size
        h_end = h_start + patch_size
        w_end = w_start + patch_size

        # Handle boundary cases with padding
        pad_before = [
            max(0, -d_start),
            max(0, -h_start),
            max(0, -w_start),
        ]
        pad_after = [
            max(0, d_end - D),
            max(0, h_end - H),
            max(0, w_end - W),
        ]

        # Clamp coordinates to valid range
        d_start = max(0, d_start)
        h_start = max(0, h_start)
        w_start = max(0, w_start)
        d_end = min(D, d_end)
        h_end = min(H, h_end)
        w_end = min(W, w_end)

        # Extract patch
        if is_3d:
            patch = patch_data[d_start:d_end, h_start:h_end, w_start:w_end]
        else:
            patch = patch_data[:, d_start:d_end, h_start:h_end, w_start:w_end]

        # Pad if necessary
        if any(pad_before) or any(pad_after):
            if is_3d:
                pad_width = [(pad_before[i], pad_after[i]) for i in range(3)]
            else:
                pad_width = [(0, 0)] + [(pad_before[i], pad_after[i]) for i in range(3)]
            
            patch = np.pad(patch, pad_width, mode="constant", constant_values=0)

        return patch


    def __getitem__(self, idx):
        """Return sample as {id, x, y} where x=(3,D,H,W) and y=(D,H,W)."""
        sid = self.cases[idx]["id"]
        i = self.id_to_index[sid]

        img = self.images[i]     # (2, D, H, W)
        prior = self.priors[i]   # (D, H, W)
        gt = self.gts[i]         # (D, H, W)

        if self.return_float32:
            img = img.float()
            prior = prior.float()
            gt = gt.float()

        # Stack channels: [T1, FLAIR, PRIOR]
        x = torch.cat([img, prior.unsqueeze(0)], dim=0)  # (3, D, H, W)

        # Apply augmentations (before patch sampling)
        if self.augment is not None:
            x, gt = self.augment(x, gt)

        # Apply patch sampling if enabled
        if self.enable_patch_sampling:
            x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
            gt_np = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt
            prior_np = x_np[2]  # Extract prior channel for center sampling
            
            center = self._sample_patch_center(prior_np, gt_np)
            
            x_patch = self._extract_patch(x_np, center, self.patch_size)
            gt_patch = self._extract_patch(gt_np, center, self.patch_size)
            
            x = torch.from_numpy(x_patch).float()
            gt = torch.from_numpy(gt_patch)
            
            if self.return_float32:
                gt = gt.float()

        return {"id": sid, "x": x, "y": gt}


class MultimodalMRIEEGPatchDataset(torch.utils.data.Dataset):
    """
    Patch-based MRI + EEG bag dataset for online EEG-conditioned segmentation.

    Returned sample format:
      {
        "subject_id": str,
        "mri": Tensor [2, D, H, W],
        "target": Tensor [D, H, W],
        "eeg_input": {"spikes": Tensor [N, C, L]},
        "patch_center": Tensor [3] normalized to [-1, 1],
      }

        By default, MRI/GT volumes and EEG spike arrays are preloaded into memory.
        Set ``force_load_into_memory=False`` to load them on demand in ``__getitem__``.
    """

    def __init__(
        self,
        cases,
        eeg_root,
        image_dtype=torch.float16,
        gt_dtype=torch.uint8,
        return_float32=True,
        patch_size=(128, 128, 128),
        patch_center_mode="random",
        enable_patch_sampling=True,
        enable_augmentation=False,
        augmentation_params=None,
        enable_eeg_augmentation=False,
        eeg_augmentation_params=None,
        disable_lr_flip=True,
        disable_strong_spatial_aug=True,
        eeg_file_suffix="_spikes_1-70Hz.npy",
        eeg_max_spikes_per_bag=32,
        eeg_min_spikes_per_patient=64,
        eeg_segment_length=256,
        eeg_window_size=128,
        eeg_max_offset=0,
        eeg_training=False,
        eeg_training_drop_ratio=0.0,
        force_load_into_memory=True,
    ):
        super().__init__()
        if patch_center_mode not in ("random", "gt_com"):
            raise ValueError(f"patch_center_mode must be 'random' or 'gt_com', got {patch_center_mode!r}")

        self.cases = list(cases)
        self.eeg_root = eeg_root
        self.image_dtype = image_dtype
        self.gt_dtype = gt_dtype
        self.return_float32 = bool(return_float32)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.patch_center_mode = patch_center_mode
        self.enable_patch_sampling = bool(enable_patch_sampling)
        self.enable_eeg_augmentation = bool(enable_eeg_augmentation)
        self.eeg_file_suffix = str(eeg_file_suffix)
        self.eeg_max_spikes_per_bag = int(eeg_max_spikes_per_bag)
        self.eeg_min_spikes_per_patient = int(eeg_min_spikes_per_patient)
        self.eeg_segment_length = int(eeg_segment_length)
        self.eeg_window_size = int(eeg_window_size)
        self.eeg_max_offset = int(eeg_max_offset)
        self.eeg_training = bool(eeg_training)
        self.eeg_training_drop_ratio = float(eeg_training_drop_ratio)
        self.force_load_into_memory = bool(force_load_into_memory)

        self.id_to_index = {}
        self.case_npz_paths = []
        self.case_eeg_paths = []
        self.images = None
        self.gts = None
        self.volume_shapes = []
        self.subject_ids = []
        self.patient_spikes = []

        default_aug = dict(
            p_rotation=0.3,
            rotation_angles=(-10, 10),
            p_scaling=0.2,
            scaling_range=(0.95, 1.05),
            p_flip=0.5,
            p_elastic=0.15,
            elastic_alpha=8,
            elastic_sigma=3,
            p_intensity_scale=0.5,
            intensity_scale_range=(0.85, 1.15),
            p_gamma=0.3,
            gamma_range=(0.8, 1.25),
            p_gaussian_noise=0.3,
            noise_std=0.02,
            p_mri_heavy_noise=0.3,
            p_gaussian_blur=0.2,
            blur_sigma_range=(0.5, 1.0),
            p_contrast=0.3,
            contrast_range=(0.8, 1.2),
            p_brightness=0.25,
            p_bias_field=0.15,
        )
        if disable_lr_flip:
            default_aug["p_flip"] = 0.0
        if disable_strong_spatial_aug:
            default_aug.update(
                p_rotation=0.1,
                rotation_angles=(-5, 5),
                p_scaling=0.1,
                scaling_range=(0.98, 1.02),
                p_elastic=0.0,
            )
        if augmentation_params is not None:
            default_aug.update(dict(augmentation_params))
        self.augment = MRIWithPriorAugment(**default_aug) if enable_augmentation else None

        self.eeg_augment = None
        if self.enable_eeg_augmentation:
            self.eeg_augment = EEGSpikeAugment(**(eeg_augmentation_params or {}))

        valid_cases = []
        valid_images = []
        valid_gts = []
        valid_volume_shapes = []
        valid_subject_ids = []
        valid_case_npz_paths = []
        valid_case_eeg_paths = []
        valid_spikes = []
        reference_shape = None

        for case in self.cases:
            sid = str(case.get("id", "<missing_id>"))
            npy_path = case.get("npy", None)

            if npy_path is None:
                warnings.warn(f"Skipping subject {sid}: case has no 'npy' path.")
                continue

            try:
                npz = np.load(npy_path, allow_pickle=True)
                img = np.asarray(npz["image"], dtype=np.float32)
                gt = np.asarray(npz["gt"], dtype=np.uint8)
                npz.close()
            except Exception as e:
                warnings.warn(f"Skipping subject {sid}: failed to load MRI npz ({npy_path}): {e}")
                continue

            if img.ndim != 4 or img.shape[0] != 2:
                warnings.warn(
                    f"Skipping subject {sid}: invalid MRI image shape {img.shape}, expected [2,D,H,W]."
                )
                continue

            if gt.ndim != 3:
                warnings.warn(f"Skipping subject {sid}: invalid GT shape {gt.shape}, expected [D,H,W].")
                continue

            if tuple(img.shape[1:]) != tuple(gt.shape):
                warnings.warn(
                    f"Skipping subject {sid}: MRI/GT shape mismatch image={img.shape[1:]} gt={gt.shape}."
                )
                continue

            if reference_shape is None:
                reference_shape = tuple(img.shape[1:])
            elif tuple(img.shape[1:]) != reference_shape:
                warnings.warn(
                    f"Skipping subject {sid}: volume shape {img.shape[1:]} does not match reference {reference_shape}."
                )
                continue

            eeg_path = os.path.join(self.eeg_root, f"{sid}{self.eeg_file_suffix}")
            if not os.path.exists(eeg_path):
                warnings.warn(f"Skipping subject {sid}: missing EEG spike file at {eeg_path}.")
                continue

            try:
                spikes = np.load(eeg_path)
            except Exception as e:
                warnings.warn(f"Skipping subject {sid}: failed to load EEG spikes ({eeg_path}): {e}")
                continue

            if spikes.ndim != 3:
                warnings.warn(
                    f"Skipping subject {sid}: invalid EEG spike shape {spikes.shape}, expected [N,C,L]."
                )
                continue

            if spikes.shape[0] < self.eeg_min_spikes_per_patient:
                warnings.warn(
                    f"Skipping subject {sid}: only {spikes.shape[0]} spikes, below minimum "
                    f"{self.eeg_min_spikes_per_patient}."
                )
                continue

            if spikes.shape[-1] < self.eeg_window_size:
                warnings.warn(
                    f"Skipping subject {sid}: EEG length {spikes.shape[-1]} shorter than window {self.eeg_window_size}."
                )
                continue

            valid_cases.append({"id": sid, "npy": npy_path})
            valid_case_npz_paths.append(npy_path)
            valid_case_eeg_paths.append(eeg_path)
            if self.force_load_into_memory:
                valid_images.append(img)
                valid_gts.append(gt)
            valid_volume_shapes.append(np.array(gt.shape, dtype=np.float32))
            valid_subject_ids.append(sid)
            if self.force_load_into_memory:
                valid_spikes.append(spikes)

        if not valid_cases:
            raise RuntimeError("No valid subjects available after filtering malformed/missing data.")

        d0, h0, w0 = reference_shape
        self.cases = valid_cases
        self.id_to_index = {c["id"]: i for i, c in enumerate(self.cases)}
        self.subject_ids = valid_subject_ids
        self.volume_shapes = valid_volume_shapes
        self.case_npz_paths = valid_case_npz_paths
        self.case_eeg_paths = valid_case_eeg_paths
        self.patient_spikes = valid_spikes if self.force_load_into_memory else []

        if self.force_load_into_memory:
            self.images = torch.empty((len(self.cases), 2, d0, h0, w0), dtype=self.image_dtype)
            self.gts = torch.empty((len(self.cases), d0, h0, w0), dtype=self.gt_dtype)

            for idx in range(len(self.cases)):
                self.images[idx].copy_(torch.from_numpy(valid_images[idx]).to(self.image_dtype))
                self.gts[idx].copy_(torch.from_numpy(valid_gts[idx]).to(self.gt_dtype))

            self.images.share_memory_()
            self.gts.share_memory_()
            self.patient_spikes = [
                torch.from_numpy(s.astype(np.float32)).share_memory_()
                for s in self.patient_spikes
            ]
        
        # Summary of loaded data
        print(f">>> Loaded {len(self.cases)} valid subjects with complete MRI and EEG data.")
        print(f">>> Loaded into memory: {self.force_load_into_memory}")
        print(f">>> Subject IDs: {self.subject_ids}")
        if self.force_load_into_memory:
            print(f">>> Images: {self.images.shape}, GTs: {self.gts.shape}, EEG spikes: {[s.shape for s in self.patient_spikes]}")

    def __len__(self):
        return len(self.cases)

    def _sample_patch_center(self, gt_np):
        d, h, w = gt_np.shape
        pd, ph, pw = self.patch_size
        if self.patch_center_mode == "gt_com":
            if gt_np.max() > 0:
                coords = np.argwhere(gt_np > 0)
                center = np.round(coords.mean(axis=0)).astype(np.int32)
            else:
                center = np.array([d // 2, h // 2, w // 2], dtype=np.int32)
        else:
            center = np.array(
                [
                    np.random.randint(pd // 2, max(pd // 2 + 1, d - pd // 2)),
                    np.random.randint(ph // 2, max(ph // 2 + 1, h - ph // 2)),
                    np.random.randint(pw // 2, max(pw // 2 + 1, w - pw // 2)),
                ],
                dtype=np.int32,
            )
        return center

    @staticmethod
    def _extract_patch(data_np, center, patch_size):
        is_4d = data_np.ndim == 4
        if is_4d:
            _, d, h, w = data_np.shape
        else:
            d, h, w = data_np.shape

        pd, ph, pw = patch_size
        d0 = center[0] - pd // 2
        h0 = center[1] - ph // 2
        w0 = center[2] - pw // 2
        d1 = d0 + pd
        h1 = h0 + ph
        w1 = w0 + pw

        pad_before = [max(0, -d0), max(0, -h0), max(0, -w0)]
        pad_after = [max(0, d1 - d), max(0, h1 - h), max(0, w1 - w)]

        d0 = max(0, d0)
        h0 = max(0, h0)
        w0 = max(0, w0)
        d1 = min(d, d1)
        h1 = min(h, h1)
        w1 = min(w, w1)

        if is_4d:
            patch = data_np[:, d0:d1, h0:h1, w0:w1]
            pad_width = [(0, 0)] + [(pad_before[i], pad_after[i]) for i in range(3)]
        else:
            patch = data_np[d0:d1, h0:h1, w0:w1]
            pad_width = [(pad_before[i], pad_after[i]) for i in range(3)]

        if any(pad_before) or any(pad_after):
            patch = np.pad(patch, pad_width, mode="constant", constant_values=0)
        return patch

    @staticmethod
    def _compute_patch_bbox(center, patch_size):
        patch_size = np.asarray(patch_size, dtype=np.int32)
        center = np.asarray(center, dtype=np.int32)
        start = center - (patch_size // 2)
        end = start + patch_size
        return np.concatenate([start, end]).astype(np.int32)

    def _build_eeg_bag(self, spikes):
        n_spikes_total, c, l = spikes.shape
        if n_spikes_total > self.eeg_max_spikes_per_bag:
            idx = np.random.choice(n_spikes_total, self.eeg_max_spikes_per_bag, replace=False)
            spikes = spikes[idx]

        n = spikes.shape[0]

        actual_start = l // 2 - self.eeg_window_size // 2
        if self.eeg_training:
            starts = np.random.randint(
                actual_start - self.eeg_max_offset,
                actual_start + self.eeg_max_offset + 1,
                size=n,
            )
        else:
            starts = np.full(n, actual_start)

        ends = starts + self.eeg_window_size
        cropped = np.stack([spikes[i, :, starts[i]:ends[i]] for i in range(n)])

        if self.eeg_training and self.eeg_training_drop_ratio > 0.0:
            keep_count = max(1, int(n * (1.0 - self.eeg_training_drop_ratio)))
            keep_idx = np.random.choice(n, keep_count, replace=False)
            cropped = cropped[keep_idx]

        return torch.tensor(cropped, dtype=torch.float32)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]

        if self.force_load_into_memory:
            img = self.images[idx]
            gt = self.gts[idx]
            spikes = self.patient_spikes[idx].numpy()
        else:
            npz_path = self.case_npz_paths[idx]
            eeg_path = self.case_eeg_paths[idx]

            npz = np.load(npz_path, allow_pickle=True)
            img_np = np.asarray(npz["image"], dtype=np.float32)
            gt_np = np.asarray(npz["gt"], dtype=np.float32)
            npz.close()

            spikes = np.load(eeg_path)
            img = torch.from_numpy(img_np)
            gt = torch.from_numpy(gt_np)

        volume_shape = self.volume_shapes[idx]

        if self.return_float32:
            img = img.float()
            gt = gt.float()

        x = img
        y = gt

        if self.augment is not None:
            x, y = self.augment(x, y)

        x_np = x.cpu().numpy() if torch.is_tensor(x) else x
        y_np = y.cpu().numpy() if torch.is_tensor(y) else y

        if self.enable_patch_sampling:
            center = self._sample_patch_center(y_np)
            x_np = self._extract_patch(x_np, center, self.patch_size)
            y_np = self._extract_patch(y_np, center, self.patch_size)
            patch_bbox = self._compute_patch_bbox(center, self.patch_size)
        else:
            center = np.array(volume_shape / 2.0, dtype=np.int32)
            patch_bbox = np.array([0, 0, 0, *volume_shape.astype(np.int32).tolist()], dtype=np.int32)

        center_norm = 2.0 * center.astype(np.float32) / np.maximum(volume_shape - 1.0, 1.0) - 1.0

        eeg_spikes = self._build_eeg_bag(spikes)
        if self.eeg_augment is not None:
            eeg_np = eeg_spikes.detach().cpu().numpy()
            eeg_np = np.asarray([self.eeg_augment(spike) for spike in eeg_np], dtype=np.float32)
            eeg_spikes = torch.from_numpy(eeg_np)

        return {
            "subject_id": sid,
            "mri": torch.from_numpy(np.asarray(x_np, dtype=np.float32)),
            "target": torch.from_numpy(np.asarray(y_np, dtype=np.float32)),
            "eeg_input": {"spikes": eeg_spikes},
            "patch_center": torch.from_numpy(center_norm.astype(np.float32)),
            "patch_bbox": torch.from_numpy(patch_bbox.astype(np.int32)),
            "volume_shape": torch.from_numpy(volume_shape.astype(np.int32)),
        }


def multimodal_mri_eeg_collate(batch):
    """
    Collate multimodal MRI+EEG patch samples with variable-size EEG bags.

    Returns:
      {
        "subject_id": list[str],
        "mri": Tensor [B,2,D,H,W],
        "target": Tensor [B,D,H,W],
        "patch_center": Tensor [B,3],
        "patch_bbox": Tensor [B,6],
        "volume_shape": Tensor [B,3],
        "eeg_input": {"spikes": Tensor [B,Nmax,C,L], "mask": Tensor [B,Nmax]},
      }
    """
    subject_ids = [b["subject_id"] for b in batch]
    mri = torch.stack([b["mri"] for b in batch], dim=0)
    target = torch.stack([b["target"] for b in batch], dim=0)
    patch_center = torch.stack([b["patch_center"] for b in batch], dim=0)
    patch_bbox = torch.stack([b["patch_bbox"] for b in batch], dim=0)
    volume_shape = torch.stack([b["volume_shape"] for b in batch], dim=0)

    spikes_list = [b["eeg_input"]["spikes"] for b in batch]
    max_spikes = max(s.shape[0] for s in spikes_list)

    padded = []
    masks = []
    for spikes in spikes_list:
        n, c, l = spikes.shape
        pad_n = max_spikes - n
        padded.append(torch.cat([spikes, torch.zeros((pad_n, c, l), dtype=spikes.dtype)], dim=0))
        mask = torch.zeros(max_spikes, dtype=torch.float32)
        mask[:n] = 1.0
        masks.append(mask)

    eeg_input = {
        "spikes": torch.stack(padded, dim=0),
        "mask": torch.stack(masks, dim=0),
    }

    return {
        "subject_id": subject_ids,
        "mri": mri,
        "target": target,
        "patch_center": patch_center,
        "patch_bbox": patch_bbox,
        "volume_shape": volume_shape,
        "eeg_input": eeg_input,
    }


if __name__ == "__main__":
    # Test dataset initialization without caching
    cases = [
        {"id": "RESP1358", "npy": "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\mri\\RESP1358\\RESP1358_preproc.npz",},
        {"id": "RESP1227", "npy": "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\mri\\RESP1227\\RESP1227_preproc.npz",}
    ]
    preds_by_id = {
        "RESP1358": {"mu": [-0.6324, -0.067, 0.477119], "sigma": [0.0827, 0.0675, 0.1521]},
        "RESP1227": {"mu": [-0.1914, 0.3717, 0.4696], "sigma": [0.08, 0.07, 0.15]},
    }
    prior_cache_dir = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\tmp\\priors"

    # Training dataset with augmentation
    dataset_train = UNetWithPriorDataset(
        cases,
        preds_by_id,
        prior_cache_dir=prior_cache_dir,
        overwrite_cache=True,
        enable_patch_sampling=True,
        patch_size=128,
        enable_augmentation=True,
    )
    
    # Validation dataset without augmentation
    dataset_val = UNetWithPriorDataset(
        cases,
        preds_by_id,
        prior_cache_dir=prior_cache_dir,
        overwrite_cache=True,
        enable_patch_sampling=True,
        patch_size=128,
        enable_augmentation=False,
    )

    print(f"Train dataset: {len(dataset_train)} samples (with augmentation)")
    print(f"Val dataset: {len(dataset_val)} samples (no augmentation)")

    for i in range(len(dataset_train)):
        sample = dataset_train[i]
        print(f"Sample {i}: id={sample['id']}, x shape={sample['x'].shape}, y shape={sample['y'].shape}")

    # Verify prior correctness for first sample
    from scipy.ndimage import center_of_mass
    print("\n=== Prior Alignment Check ===\n")
    sample = dataset_train[0]
    prior = sample["x"][2].detach().cpu().numpy()
    gt = sample["y"].detach().cpu().numpy()

    peak = np.array(np.unravel_index(np.argmax(prior), prior.shape), dtype=np.float32)
    com = np.array(center_of_mass(gt > 0.1), dtype=np.float32)

    print(f"prior peak ijk: {peak}")
    print(f"gt com ijk: {com}")
    print(f"delta: {peak - com}")

    # # visualize one sample
    # import matplotlib.pyplot as plt
    # # take first sample for inspection
    # sample = dataset[0]
    # x = sample["x"]  # (3, D, H, W)

    # # convert to numpy if torch tensor
    # if hasattr(x, "detach"):
    #     x = x.detach().cpu().numpy()

    # C, D, H, W = x.shape
    # z = np.argmax(x[2].max(axis=(1, 2)))  # slice with maximal prior intensity

    # titles = ["T1", "FLAIR", "Prior"]

    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # for c in range(3):
    #     im = axes[c].imshow(x[c, z].T, cmap="gray", origin="lower")
    #     axes[c].set_title(titles[c])
    #     axes[c].axis("off")
    #     if c != 2:
    #         contours = axes[c].contour(x[2, z].T, levels=[0.1], colors='red', linewidths=0.5, origin='lower')

    # plt.suptitle(f"Sample ID: {sample['id']}, slice z={z}")
    # plt.tight_layout()
    # plt.show()
