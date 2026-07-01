"""
mri.py
MRI-only dataset utilities for 3D U-Net segmentation.

Simplified dataset for training MRI-only (T1 + FLAIR) segmentation models
without EEG prior information. Reuses augmentation and patch sampling
functionality from the multimodal pipeline.

Author: Sjors Verschuren
Date: January 2026
"""

import os
import numpy as np
import torch
import nibabel as nib

from datasets.augmentation import MRIWithPriorAugment


class MRIDataset(torch.utils.data.Dataset):
    """
    Dataset that loads MRI images (T1w + FLAIR) for segmentation.

    Pre-loads all images and GTs into shared memory for efficient multi-worker
    data loading. Each sample is (2, D, H, W) containing T1w + FLAIR.

    Supports patch sampling to extract fixed-size patches (default 128×128×128)
    centered at random locations or at the ground truth center of mass.

    Parameters
    ----------
    cases : list of dict
        List of cases with keys: {"id": str, "npy": path}
    image_dtype : torch.dtype
        Data type for storing images
    gt_dtype : torch.dtype
        Data type for storing GT masks
    return_float32 : bool
        If True, convert to float32 on __getitem__
    patch_size : int
        Size of the patch in each dimension (default 128)
    enable_patch_sampling : bool
        If True, extract fixed-size patches from data. If False, return full volumes.
    patch_center_mode : str
        How to select patch center: "gt_com" to center at GT center of mass,
        or "random" to sample random centers.
    enable_augmentation : bool
        If True, apply data augmentation (typically for training only).
    augmentation_params : dict, optional
        Parameters for MRIWithPriorAugment. If None, uses default parameters.
    """

    def __init__(
        self,
        cases,
        image_dtype=torch.float16,
        gt_dtype=torch.uint8,
        return_float32=True,
        patch_size=128,
        enable_patch_sampling=False,
        patch_center_mode="random",
        enable_augmentation=False,
        augmentation_params=None,
    ):
        super().__init__()
        assert patch_center_mode in ("gt_com", "random"), \
            f"patch_center_mode must be 'gt_com' or 'random', got {patch_center_mode}"

        self.cases = cases
        self.return_float32 = return_float32
        self.id_to_index = {c["id"]: i for i, c in enumerate(cases)}
        self.patch_size = patch_size
        self.enable_patch_sampling = enable_patch_sampling
        self.patch_center_mode = patch_center_mode
        
        # Setup augmentation pipeline
        self.augment = None
        if enable_augmentation:
            if augmentation_params is None:
                augmentation_params = {}
            
            # Default augmentation parameters for training
            default_params = dict(
                p_rotation=0.3,
                p_scaling=0.3,
                p_flip=0.5,
                p_elastic=0.3,
                p_intensity_scale=0.5,
                p_intensity_shift=0.5,
                p_gamma=0.3,
                p_gaussian_noise=0.3,
                p_gaussian_blur=0.2,
                p_contrast=0.3,
                p_brightness=0.3,
            )
            default_params.update(augmentation_params)
            self.augment = MRIWithPriorAugment(**default_params)

        # Infer shape from first case
        npz0 = np.load(cases[0]["npy"], allow_pickle=True)
        arr0 = npz0["image"]
        assert arr0.ndim == 4 and arr0.shape[0] == 2, \
            f"Expected (2,D,H,W), got {arr0.shape}"
        _, D, H, W = arr0.shape
        npz0.close()
        del arr0

        # Pre-allocate shared tensors
        N = len(cases)
        self.images = torch.empty((N, 2, D, H, W), dtype=image_dtype)
        self.gts = torch.empty((N, D, H, W), dtype=gt_dtype)

        # Load all data
        self._load_all_data()

        # Share memory for multi-worker dataloading
        self.images.share_memory_()
        self.gts.share_memory_()

    def _load_all_data(self):
        """Load images and GTs for all cases."""
        for i, c in enumerate(self.cases):
            npz = np.load(c["npy"], allow_pickle=True)
            
            # Load image (T1, FLAIR)
            img = npz["image"]  # (2, D, H, W)
            self.images[i] = torch.from_numpy(img)
            
            # Load GT if available
            if "gt" in npz:
                gt = npz["gt"]  # (D, H, W)
                self.gts[i] = torch.from_numpy(gt)
            else:
                # No GT available - fill with zeros
                self.gts[i] = 0
            
            npz.close()

    def __len__(self):
        return len(self.cases)

    def _sample_patch_center(self, gt):
        """
        Sample a patch center location.
        
        Returns:
            center (ndarray): 3D coordinates (i, j, k) for patch center
        """
        D, H, W = gt.shape
        
        if self.patch_center_mode == "gt_com":
            # Center at GT center of mass
            from scipy.ndimage import center_of_mass
            gt_np = gt.numpy() if torch.is_tensor(gt) else gt
            
            if gt_np.max() > 0:
                com = np.array(center_of_mass(gt_np > 0.1), dtype=np.float32)
            else:
                # No GT - use volume center
                com = np.array([D / 2, H / 2, W / 2], dtype=np.float32)
            
            center = com.astype(np.int32)
        else:
            # Random center
            center = np.array([
                np.random.randint(self.patch_size // 2, D - self.patch_size // 2),
                np.random.randint(self.patch_size // 2, H - self.patch_size // 2),
                np.random.randint(self.patch_size // 2, W - self.patch_size // 2),
            ], dtype=np.int32)
        
        return center

    def _extract_patch(self, data, center, patch_size):
        """
        Extract a patch from data centered at center with given patch_size.
        Handles boundary conditions by padding if necessary.
        
        Args:
            data: numpy array of shape (C, D, H, W) or (D, H, W)
            center: center coordinates (i, j, k)
            patch_size: size of patch in each dimension
            
        Returns:
            patch: extracted patch of shape (C, P, P, P) or (P, P, P)
        """
        half_size = patch_size // 2
        
        if data.ndim == 4:
            C, D, H, W = data.shape
        else:
            D, H, W = data.shape
            
        # Calculate extraction bounds
        i_start = center[0] - half_size
        i_end = center[0] + half_size
        j_start = center[1] - half_size
        j_end = center[1] + half_size
        k_start = center[2] - half_size
        k_end = center[2] + half_size
        
        # Check if we need padding
        pad_i_before = max(0, -i_start)
        pad_i_after = max(0, i_end - D)
        pad_j_before = max(0, -j_start)
        pad_j_after = max(0, j_end - H)
        pad_k_before = max(0, -k_start)
        pad_k_after = max(0, k_end - W)
        
        # Clamp extraction bounds
        i_start = max(0, i_start)
        i_end = min(D, i_end)
        j_start = max(0, j_start)
        j_end = min(H, j_end)
        k_start = max(0, k_start)
        k_end = min(W, k_end)
        
        # Extract
        if data.ndim == 4:
            patch = data[:, i_start:i_end, j_start:j_end, k_start:k_end]
        else:
            patch = data[i_start:i_end, j_start:j_end, k_start:k_end]
        
        # Pad if necessary
        if any([pad_i_before, pad_i_after, pad_j_before, pad_j_after, pad_k_before, pad_k_after]):
            if data.ndim == 4:
                pad_width = (
                    (0, 0),
                    (pad_i_before, pad_i_after),
                    (pad_j_before, pad_j_after),
                    (pad_k_before, pad_k_after)
                )
            else:
                pad_width = (
                    (pad_i_before, pad_i_after),
                    (pad_j_before, pad_j_after),
                    (pad_k_before, pad_k_after)
                )
            
            patch = np.pad(patch, pad_width, mode='constant', constant_values=0)
        
        return patch

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - "x": MRI tensor (2, D, H, W) or (2, P, P, P) if patch sampling
                - "y": GT tensor (D, H, W) or (P, P, P) if patch sampling
                - "id": patient ID (str)
        """
        case = self.cases[idx]
        patient_id = case["id"]
        
        # Get pre-loaded data
        img = self.images[idx]  # (2, D, H, W)
        gt = self.gts[idx]      # (D, H, W)
        
        # Convert to numpy for augmentation/processing
        img_np = img.numpy() if torch.is_tensor(img) else img
        gt_np = gt.numpy() if torch.is_tensor(gt) else gt
        
        # Apply augmentation if enabled (before patch extraction)
        if self.augment is not None:
            img_np, gt_np = self.augment(img_np, gt_np)
        
        # Extract patch if enabled
        if self.enable_patch_sampling:
            center = self._sample_patch_center(gt_np)
            img_np = self._extract_patch(img_np, center, self.patch_size)
            gt_np = self._extract_patch(gt_np, center, self.patch_size)
        
        # Convert to desired output dtype
        if self.return_float32:
            img_t = torch.from_numpy(img_np).float()
            gt_t = torch.from_numpy(gt_np).float()
        else:
            img_t = torch.from_numpy(img_np)
            gt_t = torch.from_numpy(gt_np)
        
        return {
            "x": img_t,
            "y": gt_t,
            "id": patient_id,
        }


if __name__ == "__main__":
    # Test dataset initialization
    cases = [
        {"id": "RESP1358", "npy": "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\mri\\RESP1358\\RESP1358_preproc.npz"},
        {"id": "RESP1227", "npy": "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\preprocessing\\mri\\RESP1227\\RESP1227_preproc.npz"}
    ]

    # Training dataset with augmentation
    dataset_train = MRIDataset(
        cases,
        enable_patch_sampling=True,
        patch_size=128,
        enable_augmentation=True,
    )
    
    # Validation dataset without augmentation
    dataset_val = MRIDataset(
        cases,
        enable_patch_sampling=True,
        patch_size=128,
        enable_augmentation=False,
    )

    print(f"Train dataset: {len(dataset_train)} samples (with augmentation)")
    print(f"Val dataset: {len(dataset_val)} samples (no augmentation)")

    for i in range(len(dataset_train)):
        sample = dataset_train[i]
        print(f"\nSample {i} ({sample['id']}):")
        print(f"  x shape: {sample['x'].shape}, dtype: {sample['x'].dtype}")
        print(f"  y shape: {sample['y'].shape}, dtype: {sample['y'].dtype}")
        print(f"  x range: [{sample['x'].min():.3f}, {sample['x'].max():.3f}]")
        print(f"  y range: [{sample['y'].min():.3f}, {sample['y'].max():.3f}]")
