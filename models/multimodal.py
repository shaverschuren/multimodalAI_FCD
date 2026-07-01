"""
models.multimodal

Multimodal segmentation model:
- MRI backbone: 3D Residual Encoder U-Net (nnUNet-style ResEncUNet_3D)
- Input: T1 + FLAIR + PRIOR (PRIOR can be zeros if EEG missing/dropped)
- Output: segmentation logits (num_classes channels)
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mri import ResEncUNet_3D
from typing import Any, Optional


class ResEncUNet_3D_with_prior(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        use_prior: bool = True,
        prior_channel_index: int = 2,
    ):
        """
        Args:
            num_classes: output channels/classes (default 2)
            use_prior: if True, backbone expects 3 input channels; else expects 2
            prior_channel_index: the channel index position of PRIOR if x is already concatenated.
                                default assumes [T1, FLAIR, PRIOR] -> index 2
        """
        super().__init__()
        self.use_prior = bool(use_prior)
        self.prior_channel_index = int(prior_channel_index)

        in_ch = 3 if self.use_prior else 2
        self.backbone = ResEncUNet_3D(input_channels=in_ch, num_classes=num_classes, zero_init=self.use_prior)

    def forward(
        self,
        x_mri: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_mri: MRI tensor [B,2,D,H,W] (T1, FLAIR) OR [B,3,D,H,W] with an already concatenated prior.
            prior: optional prior tensor [B,1,D,H,W]. If provided and x_mri has 2 channels, we concat.

        Returns:
            logits: [B,num_classes,D,H,W]
        """
        if not self.use_prior:
            # Ignore any provided prior
            if x_mri.shape[1] != 2:
                raise ValueError(f"use_prior=False but x_mri has {x_mri.shape[1]} channels (expected 2).")
            return self.backbone(x_mri)

        # use_prior=True
        if x_mri.shape[1] == 3:
            # assume prior already included
            x = x_mri
        elif x_mri.shape[1] == 2:
            if prior is None:
                # EEG missing / dropped -> zeros prior
                prior = torch.zeros(
                    (x_mri.shape[0], 1, *x_mri.shape[2:]),
                    device=x_mri.device,
                    dtype=x_mri.dtype,
                )
            else:
                # Basic shape checks
                if prior.ndim != 5 or prior.shape[1] != 1:
                    raise ValueError(f"prior must be [B,1,D,H,W], got {tuple(prior.shape)}")
                if prior.shape[0] != x_mri.shape[0] or prior.shape[2:] != x_mri.shape[2:]:
                    raise ValueError(
                        f"prior spatial/batch dims must match x_mri. "
                        f"x_mri={tuple(x_mri.shape)}, prior={tuple(prior.shape)}"
                    )

            x = torch.cat([x_mri, prior], dim=1)  # [B,3,D,H,W]
        else:
            raise ValueError(f"x_mri must have 2 or 3 channels, got {x_mri.shape[1]}.")

        return self.backbone(x)

    def load_mri_only_pretrained(
        self,
        checkpoint_path: str,
        device=None,
        verbose: bool = True,
        checkpoint_key: str = "network_weights",
    ):
        """
        Load a MRI-only (2-channel) nnUNet-style checkpoint into this model.

        If use_prior=True, this will expand relevant conv weights (2->3) and zero-init the prior channel.
        If use_prior=False, it will just load normally.

        Args:
            checkpoint_path: path to nnUNet-style .pth file
            checkpoint_key: key containing weights in checkpoint (default 'network_weights')
        """
        pretrained_in_ch = 2  # MRI-only pretrained models have 2 input channels
        self.backbone.load_from_pth(
            path=checkpoint_path,
            device=device,
            checkpoint_key=checkpoint_key,
            pretrained_in_channels=pretrained_in_ch,
            verbose=verbose,
        )


class EEGConditionedUNet(nn.Module):
    """
    MRI U-Net segmentation conditioned by EEG subject-level embeddings + patch center.

    Conditioning is injected only at the MRI bottleneck using residual FiLM:
        z_fused = z_mri + alpha * (gamma * z_mri + beta)
    """

    def __init__(
        self,
        mri_backbone: Optional[nn.Module] = None,
        eeg_model: Optional[nn.Module] = None,
        num_classes: int = 2,
        fusion_mode: str = "residual_film",
        eeg_dim: int = 128,
        coord_dim: int = 3,
        bottleneck_channels: int = 320,
        conditioner_hidden_dim: int = 256,
        conditioner_dropout: float = 0.1,
        eeg_dropout_p: float = 0.3,
        eeg_null_strategy: str = "zero",
        alpha_init: float = 0.0,
        alpha_max: float = 0.2,
        film_init_delta: float = 1e-3,
        skip_gate_hidden_dim: int = 64,
        skip_gate_min: float = 0.75,
        skip_gate_max: float = 1.25,
        skip_gate_reg_weight: float = 0.0,
        debug_shapes: bool = False,
        enable_eeg_training: bool = False,
        verbose_fusion_debug: bool = False,
        skip_gate_channels: int | None = None,
    ):
        super().__init__()

        if fusion_mode not in {"residual_film", "bottleneck_film_skip_gate"}:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}. Expected 'residual_film' or 'bottleneck_film_skip_gate'."
            )
        if eeg_null_strategy not in {"zero", "learned"}:
            raise ValueError(
                f"eeg_null_strategy must be 'zero' or 'learned', got {eeg_null_strategy!r}"
            )

        self.mri_backbone = mri_backbone or ResEncUNet_3D(input_channels=2, num_classes=num_classes)
        self.eeg_model = eeg_model

        self.fusion_mode = fusion_mode
        self.use_skip_gate = fusion_mode == "bottleneck_film_skip_gate"
        self.eeg_dim = int(eeg_dim)
        self.coord_dim = int(coord_dim)
        self.bottleneck_channels = int(bottleneck_channels)
        self.eeg_dropout_p = float(eeg_dropout_p)
        self.eeg_null_strategy = eeg_null_strategy
        self.alpha_max = float(alpha_max)
        self.film_init_delta = float(film_init_delta)
        self.skip_gate_hidden_dim = int(skip_gate_hidden_dim)
        self.skip_gate_min = float(skip_gate_min)
        self.skip_gate_max = float(skip_gate_max)
        self.skip_gate_reg_weight = float(skip_gate_reg_weight)
        self.enable_eeg_training = bool(enable_eeg_training)
        self.verbose_fusion_debug = bool(verbose_fusion_debug)
        self.debug_shapes = bool(debug_shapes)
        self._debug_logged_once = False
        self.latest_fusion_stats: dict[str, float] = {}
        self.latest_film_stats: dict[str, float] = {}
        self.latest_skip_gate_stats: dict[str, float] = {}
        self.latest_skip_gate_reg_loss: torch.Tensor = torch.tensor(0.0)

        if self.alpha_max <= 0.0:
            raise ValueError(f"alpha_max must be > 0, got {self.alpha_max}.")
        if self.film_init_delta < 0.0:
            raise ValueError(f"film_init_delta must be >= 0, got {self.film_init_delta}.")
        if self.skip_gate_min >= self.skip_gate_max:
            raise ValueError(
                f"skip_gate_min must be smaller than skip_gate_max, got {self.skip_gate_min} >= {self.skip_gate_max}."
            )

        cond_dim = self.eeg_dim + self.coord_dim
        self.conditioner = nn.Sequential(
            nn.Linear(cond_dim, conditioner_hidden_dim),
            nn.LayerNorm(conditioner_hidden_dim),
            nn.GELU(),
            nn.Dropout(conditioner_dropout),
            nn.Linear(conditioner_hidden_dim, 2 * self.bottleneck_channels),
        )
        self._init_conditioner_film_head(delta=self.film_init_delta)
        self.alpha_bottleneck = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.alpha_skip = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        if self.eeg_null_strategy == "learned":
            self.learned_null_embedding = nn.Parameter(torch.zeros(self.eeg_dim, dtype=torch.float32))
        else:
            self.learned_null_embedding = None

        skip_gate_channels = int(skip_gate_channels) if skip_gate_channels is not None else int(self.bottleneck_channels)
        self.skip_gate_channels = skip_gate_channels
        self.skip_gate_channel_mlp = nn.Sequential(
            nn.Linear(self.eeg_dim, self.skip_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.skip_gate_hidden_dim, self.skip_gate_channels),
            nn.Tanh(),
        )
        self.skip_gate_spatial_adapter = nn.Sequential(
            nn.Conv3d(1, self.skip_gate_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(self.skip_gate_hidden_dim, 1, kernel_size=1),
        )

        self.eeg_projector: Optional[nn.Linear] = None

    def _init_conditioner_film_head(self, delta: float) -> None:
        # Residual FiLM uses z + alpha * (gamma * z + beta), so gamma,beta near 0 keeps fusion near identity.
        film_head = self.conditioner[-1]
        if not isinstance(film_head, nn.Linear):
            raise RuntimeError("Expected final conditioner layer to be nn.Linear for FiLM initialization.")
        nn.init.uniform_(film_head.weight, -delta, delta)
        nn.init.uniform_(film_head.bias, -delta, delta)

    @staticmethod
    def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
        flat = tensor.detach().float().reshape(-1)
        return {
            "mean": float(flat.mean().cpu()),
            "std": float(flat.std(unbiased=False).cpu()) if flat.numel() > 1 else 0.0,
            "min": float(flat.min().cpu()),
            "max": float(flat.max().cpu()),
            "mean_abs": float(flat.abs().mean().cpu()),
            "max_abs": float(flat.abs().max().cpu()),
        }

    def _collect_film_stats(
        self,
        layer_name: str,
        h: torch.Tensor,
        dgamma: torch.Tensor,
        dbeta: torch.Tensor,
        alpha: torch.Tensor | float | None,
    ) -> dict[str, float]:
        alpha_tensor = torch.as_tensor(1.0 if alpha is None else alpha, device=h.device, dtype=h.dtype)
        while alpha_tensor.ndim < dgamma.ndim:
            alpha_tensor = alpha_tensor.view(*alpha_tensor.shape, *([1] * (dgamma.ndim - alpha_tensor.ndim)))

        gamma_eff = 1.0 + alpha_tensor * dgamma
        beta_eff = alpha_tensor * dbeta
        scale_delta = alpha_tensor * dgamma * h
        shift_delta = beta_eff
        h_fused = h + scale_delta + shift_delta
        delta = h_fused - h
        eps = torch.finfo(h.dtype).eps
        h_std = h.detach().float().std(unbiased=False)
        h_norm = h.detach().float().norm()
        beta_eff_std = beta_eff.detach().float().std(unbiased=False)

        stats = {}
        g_stats = self._tensor_stats(dgamma)
        b_stats = self._tensor_stats(dbeta)
        ge_stats = self._tensor_stats(gamma_eff)
        stats[f"film/{layer_name}/dgamma_mean_abs"] = g_stats["mean_abs"]
        stats[f"film/{layer_name}/dgamma_std"] = g_stats["std"]
        stats[f"film/{layer_name}/dgamma_max_abs"] = g_stats["max_abs"]
        stats[f"film/{layer_name}/dbeta_mean_abs"] = b_stats["mean_abs"]
        stats[f"film/{layer_name}/dbeta_std"] = b_stats["std"]
        stats[f"film/{layer_name}/dbeta_max_abs"] = b_stats["max_abs"]
        stats[f"film/{layer_name}/gamma_eff_mean"] = ge_stats["mean"]
        stats[f"film/{layer_name}/gamma_eff_std"] = ge_stats["std"]
        stats[f"film/{layer_name}/gamma_eff_min"] = ge_stats["min"]
        stats[f"film/{layer_name}/gamma_eff_max"] = ge_stats["max"]
        stats[f"film/{layer_name}/relative_change"] = float(delta.detach().float().norm().cpu() / (h_norm.cpu() + eps))
        stats[f"film/{layer_name}/scale_change"] = float(scale_delta.detach().float().norm().cpu() / (h_norm.cpu() + eps))
        stats[f"film/{layer_name}/shift_change"] = float(shift_delta.detach().float().norm().cpu() / (h_norm.cpu() + eps))
        stats[f"film/{layer_name}/beta_to_feature_std"] = float(beta_eff_std.cpu() / (h_std.cpu() + eps))
        return stats

    def get_fusion_stats(self) -> dict[str, float]:
        return dict(self.latest_fusion_stats)

    def get_film_stats(self) -> dict[str, float]:
        return self.get_fusion_stats()

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_bottleneck

    @property
    def alpha_bottleneck_value(self) -> torch.Tensor:
        return self.alpha_bottleneck.detach()

    @property
    def alpha_skip_value(self) -> torch.Tensor:
        return self.alpha_skip.detach()

    @staticmethod
    def _normalize_zero_centered_attention(attention: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        attention = attention - attention.mean(dim=(2, 3, 4), keepdim=True)
        attention = attention / (attention.std(dim=(2, 3, 4), keepdim=True, unbiased=False) + eps)
        return torch.tanh(attention)

    @staticmethod
    def _crop_3d_region(volume: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5:
            raise ValueError(f"Expected volume as [B,C,D,H,W], got {tuple(volume.shape)}")
        if bbox.ndim != 2 or bbox.shape[1] != 6:
            raise ValueError(f"bbox must be [B,6], got {tuple(bbox.shape)}")

        crops = []
        for sample, box in zip(volume, bbox):
            d0, h0, w0, d1, h1, w1 = [int(v) for v in box.tolist()]
            target_d, target_h, target_w = d1 - d0, h1 - h0, w1 - w0
            if target_d <= 0 or target_h <= 0 or target_w <= 0:
                raise ValueError(f"Invalid bbox extents: {(d0, h0, w0, d1, h1, w1)}")

            d_max, h_max, w_max = [int(v) for v in sample.shape[-3:]]
            d0_clip, h0_clip, w0_clip = max(d0, 0), max(h0, 0), max(w0, 0)
            d1_clip, h1_clip, w1_clip = min(d1, d_max), min(h1, h_max), min(w1, w_max)

            crop = sample[:, d0_clip:d1_clip, h0_clip:h1_clip, w0_clip:w1_clip]

            pad_d_before, pad_h_before, pad_w_before = max(0, -d0), max(0, -h0), max(0, -w0)
            pad_d_after, pad_h_after, pad_w_after = max(0, d1 - d_max), max(0, h1 - h_max), max(0, w1 - w_max)
            if any(v > 0 for v in (pad_d_before, pad_d_after, pad_h_before, pad_h_after, pad_w_before, pad_w_after)):
                crop = F.pad(crop, (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after))

            if tuple(int(v) for v in crop.shape[-3:]) != (target_d, target_h, target_w):
                raise RuntimeError(
                    "Failed to crop/pad EEG spatial volume to target bbox size. "
                    f"target={(target_d, target_h, target_w)}, got={tuple(crop.shape[-3:])}"
                )
            crops.append(crop)
        return torch.stack(crops, dim=0)

    def _extract_eeg_spatial_feature(self, eeg_out: Any, batch_size: int) -> torch.Tensor:
        if isinstance(eeg_out, dict):
            for key in ("deconv_spatial", "spatial", "heatmap"):
                value = eeg_out.get(key)
                if value is None:
                    continue
                if isinstance(value, dict):
                    for nested_key in ("prob", "logits", "heatmap", "map"):
                        nested_value = value.get(nested_key)
                        if torch.is_tensor(nested_value) and nested_value.ndim == 5 and nested_value.shape[0] == batch_size:
                            return nested_value
                elif torch.is_tensor(value) and value.ndim == 5 and value.shape[0] == batch_size:
                    return value
            raise RuntimeError("EEG model output did not contain a usable spatial feature/heatmap.")

        if torch.is_tensor(eeg_out) and eeg_out.ndim == 5 and eeg_out.shape[0] == batch_size:
            return eeg_out

        raise RuntimeError(f"Unsupported EEG output type for spatial feature extraction: {type(eeg_out).__name__}")

    def _materialize_eeg_outputs(
        self,
        eeg_input: Any,
        batch_size: int,
        use_no_grad: bool,
        require_spatial: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.eeg_model is None:
            raise ValueError("eeg_model is not set, but eeg_input was provided without eeg_embedding.")

        if isinstance(eeg_input, dict):
            spikes = eeg_input.get("spikes", eeg_input.get("x", None))
            mask = eeg_input.get("mask", None)
        elif isinstance(eeg_input, (tuple, list)):
            if len(eeg_input) < 1:
                raise ValueError("eeg_input tuple/list is empty.")
            spikes = eeg_input[0]
            mask = eeg_input[1] if len(eeg_input) > 1 else None
        else:
            spikes = eeg_input
            mask = None

        if spikes is None:
            raise ValueError("Could not parse EEG spikes from eeg_input.")

        if use_no_grad:
            with torch.no_grad():
                eeg_out = self.eeg_model(spikes, mask=mask, return_embeddings=True)
        else:
            eeg_out = self.eeg_model(spikes, mask=mask, return_embeddings=True)

        eeg_embedding = self._extract_embedding(eeg_out, batch_size=batch_size)
        eeg_spatial = self._extract_eeg_spatial_feature(eeg_out, batch_size=batch_size) if require_spatial else None
        return eeg_embedding, eeg_spatial

    def _apply_eeg_dropout(
        self,
        eeg_embedding: torch.Tensor,
        eeg_spatial: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.training or self.eeg_dropout_p <= 0.0:
            return eeg_embedding, eeg_spatial

        if self.eeg_null_strategy == "learned" and self.learned_null_embedding is not None:
            null_embed = self.learned_null_embedding.unsqueeze(0).expand(eeg_embedding.shape[0], -1)
        else:
            null_embed = torch.zeros_like(eeg_embedding)

        drop_mask = (torch.rand(eeg_embedding.shape[0], device=eeg_embedding.device) < self.eeg_dropout_p).unsqueeze(1)
        eeg_embedding = torch.where(drop_mask, null_embed, eeg_embedding)
        if eeg_spatial is not None:
            eeg_spatial = torch.where(drop_mask.view(-1, 1, 1, 1, 1), torch.zeros_like(eeg_spatial), eeg_spatial)
        return eeg_embedding, eeg_spatial

    def _project_eeg_if_needed(self, eeg_embedding: torch.Tensor) -> torch.Tensor:
        if eeg_embedding.ndim != 2:
            raise ValueError(f"Expected eeg_embedding as [B,D], got {tuple(eeg_embedding.shape)}")

        in_dim = eeg_embedding.shape[1]
        if in_dim == self.eeg_dim:
            return eeg_embedding

        if self.eeg_projector is None or self.eeg_projector.in_features != in_dim:
            warnings.warn(f"Projecting EEG embedding from {in_dim} to {self.eeg_dim} dims using a linear layer.")
            self.eeg_projector = nn.Linear(in_dim, self.eeg_dim).to(eeg_embedding.device)
        return self.eeg_projector(eeg_embedding)

    def _project_eeg_spatial_if_needed(self, eeg_spatial: torch.Tensor) -> torch.Tensor:
        if eeg_spatial.ndim != 5:
            raise ValueError(f"Expected eeg_spatial as [B,C,D,H,W], got {tuple(eeg_spatial.shape)}")
        return eeg_spatial

    def _align_eeg_spatial_to_patch(
        self,
        eeg_spatial: torch.Tensor,
        patch_bbox: torch.Tensor,
        volume_shape: torch.Tensor,
        target_spatial_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        if patch_bbox is None:
            raise ValueError("patch_bbox is required for eeg spatial gating to avoid misalignment.")
        if volume_shape is None:
            raise ValueError("volume_shape is required for eeg spatial gating to avoid misalignment.")
        if patch_bbox.ndim != 2 or patch_bbox.shape[1] != 6:
            raise ValueError(f"patch_bbox must be [B,6], got {tuple(patch_bbox.shape)}")
        if volume_shape.ndim != 2 or volume_shape.shape[1] != 3:
            raise ValueError(f"volume_shape must be [B,3], got {tuple(volume_shape.shape)}")

        target_volume_shape = tuple(int(v) for v in volume_shape[0].tolist())
        if any(tuple(int(v) for v in volume_shape[i].tolist()) != target_volume_shape for i in range(volume_shape.shape[0])):
            raise ValueError("All samples in a batch must share the same volume_shape for eeg spatial alignment.")

        if tuple(int(v) for v in eeg_spatial.shape[-3:]) != target_volume_shape:
            eeg_spatial = F.interpolate(
                eeg_spatial,
                size=target_volume_shape,
                mode="trilinear",
                align_corners=False,
            )

        cropped = self._crop_3d_region(eeg_spatial, patch_bbox.to(device=eeg_spatial.device))
        if tuple(int(v) for v in cropped.shape[-3:]) != tuple(int(v) for v in target_spatial_shape):
            cropped = F.interpolate(
                cropped,
                size=tuple(int(v) for v in target_spatial_shape),
                mode="trilinear",
                align_corners=False,
            )
        return cropped

    def _collect_skip_gate_stats(
        self,
        skip: torch.Tensor,
        skip_mod: torch.Tensor,
        gate: torch.Tensor,
        attention: torch.Tensor,
        channel_gate: torch.Tensor,
    ) -> dict[str, float]:
        eps = torch.finfo(skip.dtype).eps
        delta = skip_mod - skip
        stats = {
            "skip_gate/gate_mean": float(gate.detach().mean().cpu()),
            "skip_gate/gate_std": float(gate.detach().std(unbiased=False).cpu()),
            "skip_gate/gate_min": float(gate.detach().min().cpu()),
            "skip_gate/gate_max": float(gate.detach().max().cpu()),
            "skip_gate/relative_change": float(delta.detach().float().norm().cpu() / (skip.detach().float().norm().cpu() + eps)),
            "skip_gate/A_spatial_mean": float(attention.detach().mean().cpu()),
            "skip_gate/A_spatial_std": float(attention.detach().std(unbiased=False).cpu()),
            "skip_gate/G_channel_mean": float(channel_gate.detach().mean().cpu()),
            "skip_gate/G_channel_std": float(channel_gate.detach().std(unbiased=False).cpu()),
            "skip_gate/reg_loss": float(((gate - 1.0) ** 2).mean().detach().cpu()),
        }
        return stats

    def _apply_deepest_skip_gate(
        self,
        skips: list[torch.Tensor],
        eeg_embedding: torch.Tensor,
        eeg_spatial: Optional[torch.Tensor],
        patch_bbox: torch.Tensor,
        volume_shape: torch.Tensor,
    ) -> list[torch.Tensor]:
        if not self.use_skip_gate:
            self.latest_skip_gate_stats = {}
            return skips
        if eeg_spatial is None:
            raise ValueError("Skip-gated fusion requires eeg_spatial.")

        if len(skips) < 2:
            raise RuntimeError("MRI encoder did not return enough skips to apply deepest skip gating.")

        deep_skip_idx = len(skips) - 2
        deep_skip = skips[deep_skip_idx]
        if deep_skip.ndim != 5:
            raise RuntimeError(f"Deepest skip must be 5D [B,C,D,H,W], got {tuple(deep_skip.shape)}")

        if deep_skip.shape[1] != self.skip_gate_channels:
            raise RuntimeError(
                f"Deepest skip channel count mismatch. Expected {self.skip_gate_channels}, got {deep_skip.shape[1]}."
            )

        eeg_spatial = self._project_eeg_spatial_if_needed(eeg_spatial)
        eeg_spatial_patch = self._align_eeg_spatial_to_patch(
            eeg_spatial=eeg_spatial,
            patch_bbox=patch_bbox,
            volume_shape=volume_shape,
            target_spatial_shape=tuple(deep_skip.shape[-3:]),
        )
        attention = self._normalize_zero_centered_attention(self.skip_gate_spatial_adapter(eeg_spatial_patch))
        channel_gate = self.skip_gate_channel_mlp(eeg_embedding).view(
            eeg_embedding.shape[0], self.skip_gate_channels, 1, 1, 1
        )
        gate = 1.0 + self.alpha_skip.view(1, 1, 1, 1, 1) * attention * channel_gate
        gate = torch.clamp(gate, min=self.skip_gate_min, max=self.skip_gate_max)
        self.latest_skip_gate_reg_loss = ((gate - 1.0) ** 2).mean()
        skip_mod = deep_skip * gate
        skips = list(skips)
        skips[deep_skip_idx] = skip_mod
        self.latest_skip_gate_stats = self._collect_skip_gate_stats(
            skip=deep_skip,
            skip_mod=skip_mod,
            gate=gate,
            attention=attention,
            channel_gate=channel_gate,
        )
        self.latest_fusion_stats.update(self.latest_skip_gate_stats)
        return skips

    def _get_encoder(self) -> nn.Module:
        candidates = []
        if hasattr(self.mri_backbone, "encoder"):
            candidates.append(self.mri_backbone.encoder)
        if hasattr(self.mri_backbone, "backbone") and hasattr(self.mri_backbone.backbone, "encoder"):
            candidates.append(self.mri_backbone.backbone.encoder)
        if hasattr(self, "backbone") and hasattr(self.backbone, "encoder"):
            candidates.append(self.backbone.encoder)

        for module in candidates:
            if isinstance(module, nn.Module):
                return module
        raise AttributeError("Could not find MRI encoder handle on conditioned UNet.")

    def _get_decoder(self) -> nn.Module:
        candidates = []
        if hasattr(self.mri_backbone, "decoder"):
            candidates.append(self.mri_backbone.decoder)
        if hasattr(self.mri_backbone, "backbone") and hasattr(self.mri_backbone.backbone, "decoder"):
            candidates.append(self.mri_backbone.backbone.decoder)
        if hasattr(self, "backbone") and hasattr(self.backbone, "decoder"):
            candidates.append(self.backbone.decoder)

        for module in candidates:
            if isinstance(module, nn.Module):
                return module
        raise AttributeError("Could not find MRI decoder handle on conditioned UNet.")

    def encoder_parameters(self):
        return list(self._get_encoder().parameters())

    def decoder_parameters(self):
        # Some decoder implementations hold a reference to the encoder module and
        # expose encoder params via decoder.parameters(). Filter those out so
        # optimizer groups remain disjoint.
        encoder_param_ids = {id(p) for p in self._get_encoder().parameters()}
        return [p for p in self._get_decoder().parameters() if id(p) not in encoder_param_ids]

    def eeg_parameters(self):
        if self.eeg_model is not None:
            return list(self.eeg_model.parameters())
        return []

    def fusion_parameters(self):
        params = list(self.conditioner.parameters()) + [self.alpha_bottleneck, self.alpha_skip]
        params.extend(list(self.skip_gate_channel_mlp.parameters()))
        params.extend(list(self.skip_gate_spatial_adapter.parameters()))
        if self.learned_null_embedding is not None:
            params.append(self.learned_null_embedding)
        if self.eeg_projector is not None:
            params.extend(list(self.eeg_projector.parameters()))
        return params

    def get_parameter_groups(self):
        return {
            "encoder": self.encoder_parameters(),
            "decoder": self.decoder_parameters(),
            "eeg": self.eeg_parameters(),
            "fusion": self.fusion_parameters(),
        }

    @staticmethod
    def _set_requires_grad(module: Optional[nn.Module], trainable: bool):
        if module is None:
            return 0
        n = 0
        for p in module.parameters():
            p.requires_grad = bool(trainable)
            n += p.numel()
        return n

    def set_trainability(
        self,
        train_mri_encoder: bool = False,
        train_mri_decoder: bool = False,
        train_eeg: bool = False,
        train_fusion: bool = True,
        train_segmentation_head: bool = False,
    ):
        encoder = self._get_encoder()
        decoder = self._get_decoder()
        if train_segmentation_head:
            train_mri_decoder = True
        encoder_n = self._set_requires_grad(encoder, train_mri_encoder)
        decoder_n = self._set_requires_grad(decoder, train_mri_decoder)
        eeg_n = self._set_requires_grad(self.eeg_model, train_eeg and self.enable_eeg_training)
        fusion_n = self._set_requires_grad(self.conditioner, train_fusion)
        fusion_n += self._set_requires_grad(self.skip_gate_channel_mlp, train_fusion)
        fusion_n += self._set_requires_grad(self.skip_gate_spatial_adapter, train_fusion)
        if train_fusion:
            self.alpha_bottleneck.requires_grad = True
            self.alpha_skip.requires_grad = True
        else:
            self.alpha_bottleneck.requires_grad = False
            self.alpha_skip.requires_grad = False
        return {
            "encoder_params": encoder_n,
            "decoder_params": decoder_n,
            "eeg_params": eeg_n,
            "fusion_params": fusion_n,
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = dict(state_dict)
        if "alpha_bottleneck" not in state_dict and "alpha_raw" in state_dict:
            state_dict["alpha_bottleneck"] = self.alpha_max * torch.sigmoid(state_dict.pop("alpha_raw"))
        if "alpha_skip" not in state_dict and "alpha_skip_raw" in state_dict:
            state_dict["alpha_skip"] = state_dict.pop("alpha_skip_raw")
        return super().load_state_dict(state_dict, strict=strict)

    def _extract_embedding(self, eeg_out: Any, batch_size: int) -> torch.Tensor:
        if isinstance(eeg_out, dict):
            if "embedding" in eeg_out:
                embedding = eeg_out["embedding"]
                if torch.is_tensor(embedding) and embedding.ndim == 2 and embedding.shape[0] == batch_size:
                    return embedding
                raise RuntimeError(
                    "EEG model returned embedding with unexpected shape. "
                    f"Got shape={tuple(embedding.shape)} for batch_size={batch_size}."
                )
            raise RuntimeError("EEG model output dict does not contain 'embedding' key.")

        if torch.is_tensor(eeg_out):
            if eeg_out.ndim == 2 and eeg_out.shape[0] == batch_size:
                return eeg_out
            raise RuntimeError(
                "EEG model returned a tensor that is not [B,D]. "
                f"Got shape={tuple(eeg_out.shape)} for batch_size={batch_size}."
            )

        raise RuntimeError(
            "Unsupported EEG output type for embedding extraction: "
            f"{type(eeg_out).__name__}"
        )

    def _forward_eeg(self, eeg_input: Any, batch_size: int, use_no_grad: bool) -> torch.Tensor:
        if self.eeg_model is None:
            raise ValueError("eeg_model is not set, but eeg_input was provided without eeg_embedding.")

        if isinstance(eeg_input, dict):
            spikes = eeg_input.get("spikes", eeg_input.get("x", None))
            mask = eeg_input.get("mask", None)
        elif isinstance(eeg_input, (tuple, list)):
            if len(eeg_input) < 1:
                raise ValueError("eeg_input tuple/list is empty.")
            spikes = eeg_input[0]
            mask = eeg_input[1] if len(eeg_input) > 1 else None
        else:
            spikes = eeg_input
            mask = None

        if spikes is None:
            raise ValueError("Could not parse EEG spikes from eeg_input.")

        if use_no_grad:
            with torch.no_grad():
                eeg_out = self.eeg_model(spikes, mask=mask, return_embeddings=True)
        else:
            eeg_out = self.eeg_model(spikes, mask=mask, return_embeddings=True)

        return self._extract_embedding(eeg_out, batch_size=batch_size)

    def _run_encoder_get_skips(self, mri_patch: torch.Tensor):
        encoder = self._get_encoder()
        encoder_out = encoder(mri_patch)

        if isinstance(encoder_out, (list, tuple)):
            if len(encoder_out) == 0:
                raise RuntimeError("Encoder returned an empty list/tuple of skips.")
            if not torch.is_tensor(encoder_out[-1]):
                raise RuntimeError("Encoder skip list does not end with a tensor bottleneck.")
            return list(encoder_out)

        if isinstance(encoder_out, dict):
            for key in ("skips", "features", "encoder_outputs", "activations"):
                value = encoder_out.get(key)
                if isinstance(value, (list, tuple)) and len(value) > 0 and torch.is_tensor(value[-1]):
                    return list(value)

        raise RuntimeError(
            "Unsupported encoder output structure. Expected list/tuple of skip tensors or dict with skips. "
            f"Got type={type(encoder_out).__name__}."
        )

    def _run_decoder(self, skips: list[torch.Tensor]) -> torch.Tensor:
        decoder = self._get_decoder()

        # Try the common decoder signatures used by dynamic-network-architectures variants.
        call_attempts = [
            (skips,),
            (skips[-1], skips[:-1]),
            (skips[:-1], skips[-1]),
            (*skips,),
        ]

        last_error = None
        for args in call_attempts:
            try:
                out = decoder(*args)
                if not torch.is_tensor(out):
                    raise RuntimeError(
                        "Decoder returned non-tensor output "
                        f"type={type(out).__name__} for args signature length={len(args)}"
                    )
                return out
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Could not run decoder with available skip signatures. Last error: {last_error}")

    def forward(
        self,
        mri_patch,
        eeg_input=None,
        eeg_embedding=None,
        eeg_spatial=None,
        patch_center=None,
        patch_bbox=None,
        volume_shape=None,
        force_zero_eeg=False,
        return_aux=False,
    ):
        if mri_patch.ndim != 5:
            raise ValueError(f"mri_patch must be [B,C,D,H,W], got {tuple(mri_patch.shape)}")
        if mri_patch.shape[1] != 2:
            raise ValueError(f"mri_patch must have 2 channels [T1,FLAIR]. Got channels={mri_patch.shape[1]}")

        self.latest_fusion_stats = {}
        self.latest_film_stats = {}
        self.latest_skip_gate_stats = {}

        batch_size = mri_patch.shape[0]

        if patch_center is None:
            raise ValueError("patch_center is required and must be [B,3] normalized voxel indices.")
        if patch_center.ndim != 2 or patch_center.shape[0] != batch_size or patch_center.shape[1] != 3:
            raise ValueError(
                "patch_center must be [B,3] with the same batch size as mri_patch. "
                f"Got shape={tuple(patch_center.shape)} and batch_size={batch_size}."
            )
        patch_center = patch_center.to(mri_patch.device, dtype=mri_patch.dtype)

        if force_zero_eeg:
            eeg_embedding = torch.zeros((batch_size, self.eeg_dim), device=mri_patch.device, dtype=mri_patch.dtype)
            if self.use_skip_gate:
                if volume_shape is not None:
                    full_volume_shape = tuple(int(v) for v in volume_shape[0].tolist())
                else:
                    full_volume_shape = tuple(int(v) for v in mri_patch.shape[-3:])
                eeg_spatial = torch.zeros((batch_size, 1, *full_volume_shape), device=mri_patch.device, dtype=mri_patch.dtype)
            else:
                eeg_spatial = None
        elif eeg_embedding is None or (self.use_skip_gate and eeg_spatial is None):
            if eeg_input is None and eeg_embedding is None:
                raise ValueError("Either eeg_embedding or eeg_input must be provided.")
            if eeg_input is None:
                raise ValueError("Skip-gated fusion requires eeg_input or an explicit eeg_spatial tensor.")
            use_no_grad = not (self.training and self.enable_eeg_training)
            eeg_embedding, eeg_spatial = self._materialize_eeg_outputs(
                eeg_input=eeg_input,
                batch_size=batch_size,
                use_no_grad=use_no_grad,
                require_spatial=self.use_skip_gate,
            )
        else:
            if eeg_embedding.ndim != 2 or eeg_embedding.shape[0] != batch_size:
                raise ValueError(
                    "eeg_embedding must be [B,D] with same batch size as mri_patch. "
                    f"Got shape={tuple(eeg_embedding.shape)} and batch_size={batch_size}."
                )
            eeg_embedding = eeg_embedding.to(mri_patch.device)
            if self.use_skip_gate and eeg_spatial is None:
                raise ValueError("Skip-gated fusion requires eeg_spatial when eeg_embedding is provided explicitly.")

        if eeg_spatial is not None:
            eeg_spatial = eeg_spatial.to(mri_patch.device)

        eeg_embedding = self._project_eeg_if_needed(eeg_embedding)
        eeg_spatial = eeg_spatial if eeg_spatial is None else self._project_eeg_spatial_if_needed(eeg_spatial)
        eeg_embedding, eeg_spatial = self._apply_eeg_dropout(eeg_embedding, eeg_spatial)

        cond = torch.cat([eeg_embedding, patch_center], dim=1)
        expected_cond = self.eeg_dim + self.coord_dim
        if cond.shape[1] != expected_cond:
            raise RuntimeError(f"Conditioning dim mismatch. Expected {expected_cond}, got {cond.shape[1]}.")

        skips = self._run_encoder_get_skips(mri_patch)
        z_mri = skips[-1]

        if z_mri.ndim != 5:
            raise RuntimeError(f"Expected bottleneck tensor [B,C,d,h,w], got {tuple(z_mri.shape)}")
        if z_mri.shape[0] != batch_size:
            raise RuntimeError("Bottleneck batch size does not match input batch size.")
        if z_mri.shape[1] != self.bottleneck_channels:
            raise RuntimeError(
                f"Bottleneck channels mismatch. Expected {self.bottleneck_channels}, got {z_mri.shape[1]}."
            )

        gamma_beta = self.conditioner(cond)
        dgamma, dbeta = gamma_beta.chunk(2, dim=1)
        dgamma = dgamma[:, :, None, None, None]
        dbeta = dbeta[:, :, None, None, None]
        z_fused = (1.0 + self.alpha_bottleneck * dgamma) * z_mri + self.alpha_bottleneck * dbeta
        self.latest_film_stats = self._collect_film_stats(
            layer_name="bottleneck",
            h=z_mri,
            dgamma=dgamma,
            dbeta=dbeta,
            alpha=self.alpha_bottleneck,
        )
        self.latest_fusion_stats.update(self.latest_film_stats)

        skips = list(skips)
        skips[-1] = z_fused
        skips = self._apply_deepest_skip_gate(
            skips=skips,
            eeg_embedding=eeg_embedding,
            eeg_spatial=eeg_spatial,
            patch_bbox=patch_bbox,
            volume_shape=volume_shape,
        )
        out = self._run_decoder(skips)

        if self.debug_shapes and not self._debug_logged_once:
            deep_skip = skips[-2] if len(skips) >= 2 else None
            print(
                "[EEGConditionedUNet] "
                f"mri_patch={tuple(mri_patch.shape)}, "
                f"patch_center={tuple(patch_center.shape)}, "
                f"patch_bbox={None if patch_bbox is None else tuple(patch_bbox.shape)}, "
                f"volume_shape={None if volume_shape is None else tuple(volume_shape.shape)}, "
                f"eeg_embedding={tuple(eeg_embedding.shape)}, "
                f"eeg_spatial={None if eeg_spatial is None else tuple(eeg_spatial.shape)}, "
                f"bottleneck={tuple(z_mri.shape)}, "
                f"deep_skip={None if deep_skip is None else tuple(deep_skip.shape)}, "
                f"output={tuple(out.shape)}, "
                f"alpha_bottleneck={float(self.alpha_bottleneck.detach().cpu())}, "
                f"alpha_skip={float(self.alpha_skip.detach().cpu())}"
            )
            self._debug_logged_once = True

        if return_aux:
            aux = {
                "eeg_embedding": eeg_embedding,
                "eeg_embedding_norm_mean": eeg_embedding.norm(dim=1).mean().detach(),
                "eeg_embedding_norm_std": eeg_embedding.norm(dim=1).std(unbiased=False).detach(),
                "mri_bottleneck": z_mri,
                "alpha_bottleneck": self.alpha_bottleneck.detach(),
                "alpha_skip": self.alpha_skip.detach(),
            }
            return out, aux

        return out


class MGU3D(nn.Module):
    """Minimal gated multimodal unit for 3D output-level fusion."""

    def __init__(
        self,
        in_channels_per_modality: int = 1,
        hidden_channels: int = 8,
        kernel_size: int = 3,
        use_residual_mri: bool = True,
        init_residual_alpha: float = 0.0,
        gate_bias_init: float = 1.0,
    ):
        super().__init__()

        padding = kernel_size // 2
        c = int(in_channels_per_modality)

        self.use_residual_mri = bool(use_residual_mri)

        self.mri_proj = nn.Conv3d(c, int(hidden_channels), kernel_size, padding=padding)
        self.eeg_proj = nn.Conv3d(c, int(hidden_channels), kernel_size, padding=padding)
        self.gate_proj = nn.Conv3d(2 * c, int(hidden_channels), kernel_size, padding=padding)

        self.out_proj = nn.Conv3d(int(hidden_channels), 1, kernel_size=1)

        if self.use_residual_mri:
            self.alpha = nn.Parameter(torch.tensor(float(init_residual_alpha)))
        else:
            self.register_parameter("alpha", None)

        self._init_weights(gate_bias_init=float(gate_bias_init))

    def _init_weights(self, gate_bias_init: float):
        nn.init.kaiming_normal_(self.mri_proj.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.eeg_proj.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.gate_proj.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.out_proj.weight, nonlinearity="linear")

        nn.init.zeros_(self.mri_proj.bias)
        nn.init.zeros_(self.eeg_proj.bias)
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, mri_logit, eeg_logit):
        if mri_logit.shape != eeg_logit.shape:
            raise ValueError(
                "Expected mri_logit and eeg_logit to have the same shape, "
                f"got {mri_logit.shape} and {eeg_logit.shape}"
            )

        x = torch.cat([mri_logit, eeg_logit], dim=1)

        h_mri = torch.tanh(self.mri_proj(mri_logit))
        h_eeg = torch.tanh(self.eeg_proj(eeg_logit))

        gate = torch.sigmoid(self.gate_proj(x))
        h_fused = gate * h_mri + (1.0 - gate) * h_eeg

        fused_logit_raw = self.out_proj(h_fused)

        if self.use_residual_mri:
            fused_logit = mri_logit + self.alpha * fused_logit_raw
        else:
            fused_logit = fused_logit_raw

        aux = {
            "mgu_gate": gate,
            "mgu_h_mri": h_mri,
            "mgu_h_eeg": h_eeg,
            "mgu_fused_logit_raw": fused_logit_raw,
        }

        if self.use_residual_mri:
            aux["mgu_alpha"] = self.alpha

        return fused_logit, aux


class MGUOutputFusionModel(nn.Module):
    """Wrap pretrained MRI and EEG models and apply output-level MGU fusion."""

    def __init__(self, mri_model: nn.Module, eeg_model: nn.Module, mgu: MGU3D):
        super().__init__()
        self.mri_model = mri_model
        self.eeg_model = eeg_model
        self.mgu = mgu
        self._warned_eeg_resize = False
        self.debug_eeg_alignment = False  # Set to True to enable debug output for EEG alignment

    @staticmethod
    def _extract_logits(output: Any, context: str = "model") -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (list, tuple)) and len(output) > 0 and torch.is_tensor(output[0]):
            return output[0]
        if isinstance(output, dict):
            for key in ("logits", "out", "pred", "prediction"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
        raise TypeError(f"Could not extract logits from {context} output type {type(output)!r}")

    @staticmethod
    def _to_single_channel_foreground_logit(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5:
            raise ValueError(f"Expected logits [B,C,D,H,W], got {tuple(logits.shape)}")
        if logits.shape[1] == 1:
            return logits
        if logits.shape[1] >= 2:
            return logits[:, 1:2]
        raise ValueError(f"Invalid channel count in logits: {logits.shape[1]}")

    @staticmethod
    def _extract_eeg_spatial_logits(eeg_out: Any) -> torch.Tensor:
        if torch.is_tensor(eeg_out) and eeg_out.ndim == 5:
            return eeg_out

        if isinstance(eeg_out, dict):
            deconv = eeg_out.get("deconv_spatial")
            if isinstance(deconv, dict):
                if torch.is_tensor(deconv.get("logits")):
                    return deconv["logits"]
                if torch.is_tensor(deconv.get("prob")):
                    prob = torch.clamp(deconv["prob"], min=1e-6, max=1.0 - 1e-6)
                    return torch.logit(prob)
            for key in ("logits", "spatial", "heatmap"):
                value = eeg_out.get(key)
                if torch.is_tensor(value) and value.ndim == 5:
                    return value
                if isinstance(value, dict):
                    for nested_key in ("logits", "prob", "map"):
                        nested_value = value.get(nested_key)
                        if torch.is_tensor(nested_value) and nested_value.ndim == 5:
                            if nested_key == "prob":
                                prob = torch.clamp(nested_value, min=1e-6, max=1.0 - 1e-6)
                                return torch.logit(prob)
                            return nested_value

        if isinstance(eeg_out, (list, tuple)):
            for item in eeg_out:
                if torch.is_tensor(item) and item.ndim == 5:
                    return item

        raise RuntimeError("EEG model output did not contain a usable spatial logits map.")

    @staticmethod
    def _crop_3d_region(volume: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5:
            raise ValueError(f"Expected volume as [B,C,D,H,W], got {tuple(volume.shape)}")
        if bbox.ndim != 2 or bbox.shape[1] != 6:
            raise ValueError(f"bbox must be [B,6], got {tuple(bbox.shape)}")

        crops = []
        for sample, box in zip(volume, bbox):
            d0, h0, w0, d1, h1, w1 = [int(v) for v in box.tolist()]
            target_d, target_h, target_w = d1 - d0, h1 - h0, w1 - w0
            if target_d <= 0 or target_h <= 0 or target_w <= 0:
                raise ValueError(f"Invalid bbox extents: {(d0, h0, w0, d1, h1, w1)}")

            d_max, h_max, w_max = [int(v) for v in sample.shape[-3:]]
            d0_clip, h0_clip, w0_clip = max(d0, 0), max(h0, 0), max(w0, 0)
            d1_clip, h1_clip, w1_clip = min(d1, d_max), min(h1, h_max), min(w1, w_max)

            crop = sample[:, d0_clip:d1_clip, h0_clip:h1_clip, w0_clip:w1_clip]

            pad_d_before, pad_h_before, pad_w_before = max(0, -d0), max(0, -h0), max(0, -w0)
            pad_d_after, pad_h_after, pad_w_after = max(0, d1 - d_max), max(0, h1 - h_max), max(0, w1 - w_max)
            if any(v > 0 for v in (pad_d_before, pad_d_after, pad_h_before, pad_h_after, pad_w_before, pad_w_after)):
                crop = F.pad(crop, (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after))

            if tuple(int(v) for v in crop.shape[-3:]) != (target_d, target_h, target_w):
                raise RuntimeError(
                    "Failed to crop/pad EEG volume to target bbox size. "
                    f"target={(target_d, target_h, target_w)}, got={tuple(crop.shape[-3:])}"
                )
            crops.append(crop)
        return torch.stack(crops, dim=0)

    @staticmethod
    def _validate_patch_alignment(
        mri_shape: tuple[int, int, int],
        eeg_shape: tuple[int, int, int],
        patch_bbox: Optional[torch.Tensor],
        volume_shape: Optional[torch.Tensor],
    ):
        """Validate that patch extraction is correctly configured.
        
        Args:
            mri_shape: MRI patch shape [D, H, W]
            eeg_shape: EEG patch shape after alignment [D, H, W]
            patch_bbox: Patch bbox in volume coordinates [B,6]
            volume_shape: Full volume shape [B,3]
        """
        errors = []
        
        if mri_shape != eeg_shape:
            errors.append(f"Shape mismatch: MRI {mri_shape} vs EEG {eeg_shape}")
        
        if patch_bbox is not None and volume_shape is not None:
            for i, (bbox, vshape) in enumerate(zip(patch_bbox, volume_shape)):
                d0, h0, w0, d1, h1, w1 = [int(v) for v in bbox.tolist()]
                vd, vh, vw = [int(v) for v in vshape.tolist()]
                
                if d1 <= d0 or h1 <= h0 or w1 <= w0:
                    errors.append(f"Sample {i}: Invalid bbox extents (d0={d0}, h0={h0}, w0={w0}, d1={d1}, h1={h1}, w1={w1})")
                
                # Check bounds - allow padding for edge cases
                if d0 < -100 or h0 < -100 or w0 < -100:
                    errors.append(f"Sample {i}: Bbox start < -100 (likely error): ({d0}, {h0}, {w0})")
                if d1 > vd + 100 or h1 > vh + 100 or w1 > vw + 100:
                    errors.append(f"Sample {i}: Bbox end > volume+100 (likely error): ({d1}, {h1}, {w1}) vs volume {(vd, vh, vw)}")
        
        return errors

    def _align_eeg_to_patch(
        self,
        eeg_logit: torch.Tensor,
        target_shape: tuple[int, int, int],
        patch_bbox: Optional[torch.Tensor],
        volume_shape: Optional[torch.Tensor],
        debug: bool = False,
    ):
        if patch_bbox is None or volume_shape is None:
            if tuple(int(v) for v in eeg_logit.shape[-3:]) != tuple(int(v) for v in target_shape):
                if not self._warned_eeg_resize:
                    warnings.warn(
                        "EEG spatial logit did not match MRI patch shape and patch metadata was missing. "
                        "Falling back to direct interpolation to MRI patch size.",
                        RuntimeWarning,
                    )
                    self._warned_eeg_resize = True
                if debug:
                    print(f"[DEBUG] No bbox/volume_shape: Interpolating EEG from {tuple(int(v) for v in eeg_logit.shape[-3:])} to target {target_shape}")
                eeg_logit = F.interpolate(eeg_logit, size=target_shape, mode="trilinear", align_corners=False)
            return eeg_logit

        patch_bbox = patch_bbox.to(device=eeg_logit.device)
        target_volume_shape = tuple(int(v) for v in volume_shape[0].tolist())
        eeg_shape_native = tuple(int(v) for v in eeg_logit.shape[-3:])
        
        if debug:
            bbox_d0, bbox_h0, bbox_w0, bbox_d1, bbox_h1, bbox_w1 = [int(v) for v in patch_bbox[0].tolist()]
            print(f"[DEBUG] EEG Alignment:")
            print(f"  EEG native shape: {eeg_shape_native} (expected to represent full volume)")
            print(f"  Target volume shape: {target_volume_shape}")
            print(f"  Patch bbox: d=[{bbox_d0}:{bbox_d1}] h=[{bbox_h0}:{bbox_h1}] w=[{bbox_w0}:{bbox_w1}]")
            print(f"  Expected patch shape from bbox: ({bbox_d1-bbox_d0}, {bbox_h1-bbox_h0}, {bbox_w1-bbox_w0})")
            print(f"  Target MRI patch shape: {target_shape}")
        
        # Step 1: Interpolate EEG to full volume shape if needed
        if eeg_shape_native != target_volume_shape:
            if debug:
                print(f"  Step 1: Interpolating EEG {eeg_shape_native} -> {target_volume_shape}")
            eeg_logit = F.interpolate(eeg_logit, size=target_volume_shape, mode="trilinear", align_corners=False)

        # Step 2: Crop to patch region
        eeg_patch = self._crop_3d_region(eeg_logit, patch_bbox)
        if debug:
            print(f"  Step 2: Cropped to {tuple(int(v) for v in eeg_patch.shape[-3:])}")
        
        # Step 3: Final interpolation if patch size doesn't match MRI patch
        eeg_patch_shape = tuple(int(v) for v in eeg_patch.shape[-3:])
        if eeg_patch_shape != tuple(int(v) for v in target_shape):
            if not self._warned_eeg_resize:
                warnings.warn(
                    f"EEG patch logit size {eeg_patch_shape} differed from MRI logit size {target_shape} after bbox crop. "
                    "Applying trilinear interpolation.",
                    RuntimeWarning,
                )
                self._warned_eeg_resize = True
            if debug:
                print(f"  Step 3: Final interpolation {eeg_patch_shape} -> {target_shape}")
            eeg_patch = F.interpolate(eeg_patch, size=target_shape, mode="trilinear", align_corners=False)
        elif debug:
            print(f"  Step 3: Patch shapes match, no interpolation needed")
        
        return eeg_patch

    def forward(
        self,
        mri_patch: torch.Tensor,
        eeg_input: Any,
        patch_center: Optional[torch.Tensor] = None,
        patch_bbox: Optional[torch.Tensor] = None,
        volume_shape: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ):
        mri_out = self.mri_model(mri_patch)
        mri_logits_full = self._extract_logits(mri_out, context="mri")
        mri_fg_logit = self._to_single_channel_foreground_logit(mri_logits_full)

        if isinstance(eeg_input, dict):
            spikes = eeg_input.get("spikes", eeg_input.get("x", None))
            mask = eeg_input.get("mask", None)
        elif isinstance(eeg_input, (tuple, list)):
            spikes = eeg_input[0] if len(eeg_input) > 0 else None
            mask = eeg_input[1] if len(eeg_input) > 1 else None
        else:
            spikes = eeg_input
            mask = None

        if spikes is None:
            raise ValueError("Could not parse EEG spikes from eeg_input.")

        eeg_out = self.eeg_model(spikes, mask=mask, return_embeddings=True)
        eeg_logit = self._extract_eeg_spatial_logits(eeg_out)
        eeg_logit = eeg_logit.to(device=mri_fg_logit.device, dtype=mri_fg_logit.dtype)
        eeg_fg_logit = self._to_single_channel_foreground_logit(eeg_logit)
        eeg_fg_logit = self._align_eeg_to_patch(
            eeg_logit=eeg_fg_logit,
            target_shape=tuple(int(v) for v in mri_fg_logit.shape[-3:]),
            patch_bbox=patch_bbox,
            volume_shape=volume_shape,
            debug=self.debug_eeg_alignment,
        )

        # Validate alignment
        if self.debug_eeg_alignment or (patch_bbox is not None and volume_shape is not None):
            errors = self._validate_patch_alignment(
                mri_shape=tuple(int(v) for v in mri_fg_logit.shape[-3:]),
                eeg_shape=tuple(int(v) for v in eeg_fg_logit.shape[-3:]),
                patch_bbox=patch_bbox,
                volume_shape=volume_shape,
            )
            if errors:
                if self.debug_eeg_alignment:
                    print("[VALIDATION ERRORS]")
                    for e in errors:
                        print(f"  {e}")
                else:
                    warnings.warn(f"EEG patch alignment issues: {'; '.join(errors)}", RuntimeWarning)

        fused_fg_logit, mgu_aux = self.mgu(mri_fg_logit, eeg_fg_logit)
        aux = {
            "mri_logits_full": mri_logits_full,
            "mri_logit": mri_fg_logit,
            "eeg_logit": eeg_fg_logit,
            "patch_center": patch_center,
            **mgu_aux,
        }
        if "mgu_alpha" not in aux and hasattr(self.mgu, "alpha") and self.mgu.alpha is not None:
            aux["mgu_alpha"] = self.mgu.alpha

        if return_aux:
            return fused_fg_logit, aux
        return fused_fg_logit

if __name__ == "__main__":
    # Dummy data
    x_mri = torch.randn(2, 2, 128, 128, 128)  # T1, FLAIR
    prior = torch.randn(2, 1, 128, 128, 128)  # PRIOR
    x_eeg = torch.randn(2, 32, 21, 128)  # Simulated EEG spikes [B, num_spikes, channels, samples]
    patch_center = torch.tensor([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])  # Normalized patch centers

    # # Simple test (with prior)
    # model = ResEncUNet_3D_with_prior(num_classes=2, use_prior=True)
    # model.load_mri_only_pretrained(
    #     checkpoint_path="L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\tmp\\checkpoint_best.pth"
    # )
    # logits = model(x_mri, prior)
    # print("Logits shape (with prior):", logits.shape)
    # # Simple test (without prior)
    # model_no_prior = ResEncUNet_3D_with_prior(num_classes=2, use_prior=False)
    # model_no_prior.load_mri_only_pretrained(
    #     checkpoint_path="L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\tmp\\checkpoint_best.pth"
    # )
    # x_mri_no_prior = torch.randn(2, 2, 64, 64, 64)  # T1, FLAIR
    # logits_no_prior = model_no_prior(x_mri)
    # print("Logits shape (no prior):", logits_no_prior.shape)

    # # Check if logits and logits_no_prior are (almost) the same
    # diff = torch.abs(logits - logits_no_prior).max().item()
    # print(f"Max difference between logits and logits_no_prior: {diff}")
    # if diff < 1e-5:
    #     print("Logits are (almost) identical")
    # else:
    #     print("Logits differ significantly.")
    #     print("Logits (with prior):", logits)
    #     print("Logits (no prior):", logits_no_prior)
    
    # Simple test for EEGConditionedUNet
    from models.eeg import SpikeMILModel
    mri_model = ResEncUNet_3D(input_channels=2, num_classes=2)
    eeg_model = SpikeMILModel()
    conditioned_model = EEGConditionedUNet(
        mri_backbone=mri_model,
        eeg_model=eeg_model,
        num_classes=2,
        fusion_mode="residual_film",
        eeg_dim=64,
        coord_dim=3,
        bottleneck_channels=320,
        conditioner_hidden_dim=256,
        conditioner_dropout=0.1,
        eeg_dropout=0.3,
        eeg_null_strategy="zero",
        alpha_init=0.0,
        debug_shapes=True,
    )

    logits_conditioned = conditioned_model(x_mri, eeg_input=x_eeg, patch_center=patch_center)
    print("Logits shape (conditioned):", logits_conditioned.shape)