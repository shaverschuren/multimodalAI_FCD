"""
mri.py
3D Residual Encoder U-Net model for MRI segmentation of FCDs.

This architecture is based on the widely adopted nnU-Net design
by Isensee et al. (2021). Architectural parameters were determined
via the dataset fingerprinting + planner functionality of nnUNetv2. 
Also includes functionality to load nnU-Net-style checkpoints, with support
for expanding the first convolutional layer to accommodate an additional input channel
for EEG prior information.

References:
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature Methods, 18(2), 203-211.
https://github.com/MIC-DKFZ/nnUNet.git
"""

import warnings
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


def expand_mismatched_conv3d_weights_to_match_model(
    state_dict: dict,
    model: nn.Module,
    verbose: bool = True,
) -> dict:
    """
    For each parameter in `state_dict` that exists in `model.state_dict()`:
      - if it's a 5D conv weight and only differs in in_channels (dim=1),
        expand checkpoint weight to match model weight shape by zero-padding extra channels.
    """
    sd = dict(state_dict)
    msd = model.state_dict()

    expanded = []
    skipped = []

    for k, w_ckpt in list(sd.items()):
        if k not in msd:
            continue
        w_model = msd[k]
        if not (torch.is_tensor(w_ckpt) and torch.is_tensor(w_model)):
            continue

        # Only handle Conv3d-like weights
        if w_ckpt.ndim == 5 and w_model.ndim == 5:
            # Match out_channels + kernel, differ only in in_channels
            if (w_ckpt.shape[0] == w_model.shape[0]) and (w_ckpt.shape[2:] == w_model.shape[2:]):
                in_old = w_ckpt.shape[1]
                in_new = w_model.shape[1]
                if in_new > in_old:
                    w_new = torch.zeros(
                        (w_ckpt.shape[0], in_new, *w_ckpt.shape[2:]),
                        dtype=w_ckpt.dtype,
                        device=w_ckpt.device,
                    )
                    w_new[:, :in_old, ...] = w_ckpt
                    sd[k] = w_new
                    expanded.append((k, in_old, in_new))
                elif in_new != in_old:
                    skipped.append((k, tuple(w_ckpt.shape), tuple(w_model.shape)))

    if verbose:
        if expanded:
            print(f"[INFO] Expanded {len(expanded)} conv weights to match model in_channels:")
            for k, a, b in expanded:
                if "stem" in k:
                    print(f"  - {k}: {a} -> {b} (stem)")
            # Print a few non-stem if any
            nonstem = [(k,a,b) for k,a,b in expanded if "stem" not in k]
            for k,a,b in nonstem[:5]:
                print(f"  - {k}: {a} -> {b}")
        if skipped:
            print(f"[WARN] Found {len(skipped)} conv weights with incompatible shapes (not just in_channels).")
            for k, s1, s2 in skipped[:5]:
                print(f"  - {k}: ckpt {s1} vs model {s2}")

    return sd


def normalize_mri_checkpoint_state_dict(state_dict: dict, verbose: bool = True) -> dict:
    """
    Normalize MRI checkpoint keys so they can be loaded into `ResEncUNet_3D.backbone`.

    Checkpoints saved from the wrapped MRI module usually contain keys prefixed with
    `backbone.`. DataParallel checkpoints may additionally include `module.`.
    This helper strips those outer wrapper prefixes while leaving the internal nnU-Net
    module names intact.
    """
    sd = dict(state_dict)
    normalized = {}
    changed = False

    for key, value in sd.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
            changed = True
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone.") :]
            changed = True
        normalized[new_key] = value

    if verbose and changed:
        print("[INFO] Normalized MRI checkpoint keys by stripping wrapper prefixes (module./backbone.).")

    return normalized


def prefix_mri_checkpoint_state_dict_for_wrapper(state_dict: dict, verbose: bool = True) -> dict:
    """
    Prefix MRI checkpoint keys with `backbone.` so they can be loaded into the
    full `ResEncUNet_3D` wrapper instead of its bare backbone module.

    This is the inverse of loading directly into `model.backbone`.
    """
    sd = dict(state_dict)
    prefixed = {}
    changed = False

    for key, value in sd.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
            changed = True
        if not new_key.startswith("backbone."):
            new_key = f"backbone.{new_key}"
            changed = True
        prefixed[new_key] = value

    if verbose and changed:
        print("[INFO] Prefixed MRI checkpoint keys with backbone. for wrapper loading.")

    return prefixed


class ResEncUNet_3D(torch.nn.Module):
    def __init__(self, input_channels: int = 2, num_classes: int = 2, zero_init: bool = False):
        """
        Args:
            input_channels: number of input modalities/channels.
                - 2 for (T1, FLAIR)
                - 3 for (T1, FLAIR, PRIOR)
            num_classes: output channels/classes (default 2).
        """
        super().__init__()

        # nnU-Net architectural parameters (from fingerprinting + planner)
        self.nnUNet_kwargs = dict(
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=torch.nn.Conv3d,
            kernel_sizes=[(3, 3, 3)] * 6,
            strides=[
                (1, 1, 1),
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
            ],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs=dict(eps=1e-5, affine=True),
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs=dict(inplace=True),
        )

        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.zero_init = bool(zero_init)

        # Backbone (Residual Encoder U-Net, nnU-Net default)
        self.backbone = ResidualEncoderUNet(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            **self.nnUNet_kwargs,
        )

        if self.zero_init:
            # Zero-initialize all weights and biases (including convs, norms) to start with zero output logits.
            for param in self.parameters():
                param.data.zero_()

    @property
    def encoder(self):
        """Expose the underlying encoder module for targeted freezing or inspection."""
        return self.backbone.encoder
    
    @property
    def decoder(self):
        """Expose the underlying decoder module for targeted freezing or inspection."""
        return self.backbone.decoder

    @property
    def stem_conv(self):
        """Expose the first encoder convolution that receives the input channels."""
        return self.backbone.encoder.stem.convs[0].conv

    def forward(self, x):
        """
        x: [B, C, D, H, W], where C is 2 (T1,FLAIR) or 3 (T1,FLAIR,PRIOR)
        """
        return self.backbone(x)

    def load_from_pth(
        self,
        path: str,
        device=None,
        checkpoint_key: str = "network_weights",
        pretrained_in_channels: int = 2,
        verbose: bool = True,
    ):
        """
        Loads a nnUNet-style checkpoint. If this model has more input channels than the
        checkpoint (e.g., model=3, ckpt=2), expands the first conv weight tensors
        and zero-initializes the new channel(s).
        This function is mainly intended to load pretrained MRI-only weights into
        a multimodal MRI+EEG prior model later.

        Args:
            path: path to .pth checkpoint
            checkpoint_key: key in checkpoint dict containing network weights
            pretrained_in_channels: input channels used during pretraining (typically 2)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if verbose:
            print(f"[INFO] Loading checkpoint from: {path} to device: {device}")

        ckpt = torch.load(path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict) and checkpoint_key in ckpt:
            state_dict = ckpt[checkpoint_key]
        elif isinstance(ckpt, dict):
            # fallback: maybe checkpoint is already a state_dict
            state_dict = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format at: {path}")

        state_dict = normalize_mri_checkpoint_state_dict(state_dict, verbose=verbose)

        # Expand conv3d weights if needed
        if self.input_channels != pretrained_in_channels:
            state_dict = expand_mismatched_conv3d_weights_to_match_model(
                state_dict=state_dict,
                model=self.backbone,
                verbose=verbose,
            )

        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)

        if len(missing) > 0 or len(unexpected) > 0:
            warnings.warn(
                f"While loading weights from {path}, "
                f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
            )
            verbose = True  # force verbose if there are issues

        if verbose:
            print(f"[INFO] Loaded weights from: {path}")
            print(f"[INFO] missing keys: {len(missing)}")
            print(f"[INFO] unexpected keys: {len(unexpected)}")
            if len(missing) < 50 and len(missing) > 0:
                print("[INFO] missing:", missing)
            if len(unexpected) < 50 and len(unexpected) > 0:
                print("[INFO] unexpected:", unexpected)

if __name__ == "__main__":
    # Simple test
    model = ResEncUNet_3D(input_channels=2, num_classes=2)
    model.load_from_pth(
        path="L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\tmp\\checkpoint_best.pth",
        pretrained_in_channels=2,
        verbose=True,
    )
    x = torch.randn((2, 2, 128, 128, 128))
    with torch.no_grad():
        y = model(x)
    print("Output shape:", y.shape)  # expect [2, 2, 128, 128, 128]
    # print(model.encoder)
    print(model.decoder)  