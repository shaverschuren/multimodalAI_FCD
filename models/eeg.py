"""
models/eeg.py

Multi-Instance Learning (MIL) models for EEG spike localization.

Core architecture (SpikeMILModel):
  - Spike encoder:  SpikeEncoder_T_S  (temporal 1D CNN + GNN spatial mixing)
  - MIL pooling:    attention or mean pooling over spikes per patient
  - Heads:          deconv 3D spatial head (primary), plus optional coordinate
                    regression, hemisphere classification, and lobe classification

Experimental/deprecated model classes are kept for reference:
  - SpikeMILClassifier  (used by eeg_spike_mil_training.py)
  - SpikeMILRegressor   (used by eeg_spike_mil_regression_training.py)
  - ChannelAlignedMILClassifier (used by eeg_spike_mil_channel_training.py)
"""

import os
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel order
CHANNEL_ORDER = [
    "Fp1", "Fp2", "F9", "F10", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2"
]
# Standard channel positions for GNN adjacency
_script_dir = os.path.dirname(os.path.abspath(__file__))
_coord_file = os.path.join(_script_dir, "..", "..", "preprocessing", "eeg", "standard_1005_mni_coordinates.json")
with open(_coord_file, "r") as f:
    coord_data = json.load(f)
    CHANNEL_MNI_COORDS = {
        ch: (
            data["mni"]["x"],
            data["mni"]["y"],
            data["mni"]["z"],
        )
        for ch, data in coord_data["data"].items()
    }

# -----------------------------------------------
# Channel-aligned model for by-channel predictions
# -----------------------------------------------
class SpikeEncoder1DChannelAligned(nn.Module):
    """
    Channel-aligned spike encoder.

    Input:
        x: (B, N, C, L)
    Output:
        H: (B, N, C, D)
    """
    def __init__(self, in_channels=21, emb_dim=4):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim

        # Depthwise temporal convolution: one encoder per channel
        self.temporal = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels * emb_dim,
                kernel_size=7,
                padding=3,
                groups=in_channels  # <-- hard channel separation
            ),
            nn.GroupNorm(in_channels, in_channels * emb_dim),
            nn.ELU(),

            nn.Conv1d(
                in_channels=in_channels * emb_dim,
                out_channels=in_channels * emb_dim,
                kernel_size=5,
                padding=2,
                groups=in_channels
            ),
            nn.GroupNorm(in_channels, in_channels * emb_dim),
            nn.ELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        B, N, C, L = x.shape
        assert C == self.in_channels

        x = x.view(B * N, C, L)                 # (BN, C, L)
        x = self.temporal(x)                    # (BN, C*D, L)
        x = self.pool(x).squeeze(-1)            # (BN, C*D)

        x = x.view(B, N, C, self.emb_dim)       # (B, N, C, D)
        return x


class ChannelWiseAttentionPooling(nn.Module):
    """
    Attention pooling over spikes, independently per channel.

    Input:
        H: (B, N, C, D)
    Output:
        pooled: (B, C, D)
        attn: (B, C, N)
    """
    def __init__(self, emb_dim, hidden=16):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, H, mask=None):
        B, N, C, D = H.shape

        # Compute attention per channel
        scores = self.attn(H).squeeze(-1)   # (B, N, C)
        scores = scores.permute(0, 2, 1)    # (B, C, N)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (B, C, N)

        pooled = (H.permute(0, 2, 1, 3) * weights.unsqueeze(-1)).sum(dim=2)
        # pooled: (B, C, D)

        return pooled, weights


class ChannelInteractionBlock(nn.Module):
    """
    Residual self-attention over channels.
    Input/Output: (B, C, D)
    """
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, D)
        attn_out, attn_weights = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x, attn_weights  # weights: (B, heads, C, C)


class ChannelAlignedMILClassifier(nn.Module):
    """
    MIL model for per-channel classification.

    Predicts:
        y_hat: (B, C)  -- soft scores per channel
    """
    def __init__(
        self,
        in_channels=21,
        emb_dim=4,
        hidden=16,
        dropout=0.3,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.hidden = hidden
        self.dropout = dropout

        self.encoder = SpikeEncoder1DChannelAligned(
            in_channels=in_channels,
            emb_dim=emb_dim
        )

        self.pool = ChannelWiseAttentionPooling(emb_dim, hidden=hidden)

        self.channel_interaction = ChannelInteractionBlock(
            dim=emb_dim,
            num_heads=2,
            dropout=0.1
        )

        # Shared channel-wise head
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, mask=None):
        """
        x: (B, N, C, L)
        mask: (B, N)

        Returns:
            y_hat: (B, C)
            attn_weights: (B, C, N)
        """
        H = self.encoder(x)                      # (B, N, C, D)
        pooled, attn = self.pool(H, mask)

        pooled, attn_channel = self.channel_interaction(pooled)

        if self.training:
            # channel dropout: drop entire channel representations
            drop_prob = 0.1
            channel_mask = (torch.rand(
                pooled.shape[:2], device=pooled.device
            ) > drop_prob).unsqueeze(-1)  # (B, C, 1)
            pooled = pooled * channel_mask
        
        y_hat = self.head(pooled).squeeze(-1)    # raw logits for BCEWithLogitsLoss

        return y_hat, attn


# -----------------------------------------------
# Temporal-then-spatial spike encoder
# -----------------------------------------------
def build_distance_weighted_adjacency(
    channel_order: list[str],
    channel_coords: dict[str, tuple[float, float, float]],
    k: int = 4,
    sigma: float = 40.0,
    include_self: bool = False,
) -> torch.Tensor:
    """
    Build a sparse, symmetric, distance-weighted adjacency matrix.

    Args:
        channel_order:
            Ordered list of channel names.

        channel_coords:
            Dict mapping channel name to 3D coordinate.

        k:
            Number of nearest neighbors per channel.

        sigma:
            Distance scale controlling edge decay.
            Larger sigma = more similar weights across neighbors.
            Smaller sigma = stronger preference for very nearby channels.

        include_self:
            Whether to include diagonal self-connections.
            Usually False here, because the model can add self-loops internally.

    Returns:
        adjacency:
            [C, C] float tensor.
    """
    coords = torch.tensor(
        [channel_coords[ch] for ch in channel_order],
        dtype=torch.float32,
    )  # [C, 3]

    distances = torch.cdist(coords, coords, p=2)  # [C, C]

    n_channels = len(channel_order)
    adjacency = torch.zeros(n_channels, n_channels, dtype=torch.float32)

    for i in range(n_channels):
        # Exclude self from nearest-neighbor search.
        dist_i = distances[i].clone()
        dist_i[i] = float("inf")

        nearest = torch.topk(
            dist_i,
            k=k,
            largest=False,
        ).indices

        for j in nearest:
            d = distances[i, j]
            weight = torch.exp(-(d ** 2) / (2 * sigma ** 2))
            adjacency[i, j] = weight

    # Make graph symmetric.
    adjacency = torch.maximum(adjacency, adjacency.T)

    if include_self:
        adjacency.fill_diagonal_(1.0)

    return adjacency


class SpikeEncoder_T_S(nn.Module):
    """
    Temporal + spatial spike encoder.

    Expected input:
        x: [batch, channels, time]
           e.g. [B, 21, 128]

    Output:
        out: [batch, spatial_output_dim]

    Structure:
        1. Shared temporal 1D CNN applied independently to each channel.
        2. Spatial mixing using either:
            - fully connected MLP over flattened channel embeddings
            - simple dense GNN-style message passing over channel nodes

    temporal_mode:
        "simple"      -> original sequential 1D CNN
        "multiscale"  -> multi-scale temporal CNN with parallel kernels

    spatial_mode:
        "fc"   -> flatten [B, C, D] to [B, C*D], then MLP
        "gnn"  -> treat channels as graph nodes and perform message passing
    """

    def __init__(
        self,
        n_channels: int = 21,
        input_length: int = 128,
        temporal_embedding_dim: int = 64,
        spatial_hidden_dim: int = 64,
        spatial_output_dim: int = 64,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        temporal_mode: str = "multiscale",
        spatial_mode: str = "gnn",
        adjacency: Optional[torch.Tensor] = None,
        gnn_num_layers: int = 1,
        add_self_loops: bool = True,
        normalize_adjacency: bool = True,
    ):
        super().__init__()

        if temporal_mode not in {"simple", "multiscale"}:
            raise ValueError(
                f"temporal_mode must be 'simple' or 'multiscale', got {temporal_mode!r}"
            )

        if spatial_mode not in {"fc", "gnn"}:
            raise ValueError(
                f"spatial_mode must be 'fc' or 'gnn', got {spatial_mode!r}"
            )

        self.n_channels = n_channels
        self.input_length = input_length
        self.temporal_embedding_dim = temporal_embedding_dim
        self.spatial_output_dim = spatial_output_dim
        self.temporal_mode = temporal_mode
        self.spatial_mode = spatial_mode
        self.dropout_p = dropout
        self.use_layernorm = use_layernorm

        # -------------------------
        # Shared temporal encoder
        # -------------------------
        if temporal_mode == "simple":
            self.temporal_cnn = nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=7,
                    padding=3,
                ),
                nn.GELU(),

                nn.Conv1d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                nn.GELU(),

                nn.Conv1d(
                    in_channels=64,
                    out_channels=96,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                nn.GELU(),

                nn.Conv1d(
                    in_channels=96,
                    out_channels=128,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),

                nn.AdaptiveAvgPool1d(1),
            )

        else:
            self.temporal_stem = nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=7,
                    padding=3,
                ),
                nn.GELU(),
            )

            self.ms_block_1 = self._make_multiscale_block(
                in_channels=32,
                out_channels=64,
                dropout=dropout,
            )

            self.downsample_1 = nn.Sequential(
                nn.Conv1d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                nn.GELU(),
            )

            self.ms_block_2 = self._make_multiscale_block(
                in_channels=64,
                out_channels=96,
                dropout=dropout,
            )

            self.downsample_2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=96,
                    out_channels=128,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                nn.GELU(),
            )

            self.ms_block_3 = self._make_multiscale_block(
                in_channels=128,
                out_channels=128,
                dropout=dropout,
            )

            self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        temporal_projection_layers = [
            nn.Flatten(start_dim=1),  # [B*C, 128, 1] -> [B*C, 128]
            nn.Linear(128, temporal_embedding_dim),
        ]

        if use_layernorm:
            temporal_projection_layers.append(nn.LayerNorm(temporal_embedding_dim))

        if dropout > 0:
            temporal_projection_layers.append(nn.Dropout(dropout))

        self.temporal_projection = nn.Sequential(*temporal_projection_layers)

        # -------------------------
        # Fully connected spatial module
        # -------------------------
        spatial_input_dim = n_channels * temporal_embedding_dim

        spatial_layers = []

        if use_layernorm:
            spatial_layers.append(nn.LayerNorm(spatial_input_dim))

        spatial_layers.extend([
            nn.Linear(spatial_input_dim, spatial_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(spatial_hidden_dim, spatial_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(spatial_hidden_dim, spatial_output_dim),
        ])

        self.spatial_mlp = nn.Sequential(*spatial_layers)

        # -------------------------
        # GNN spatial module
        # -------------------------
        if self.spatial_mode == "gnn":
            if gnn_num_layers < 1:
                raise ValueError(
                    f"gnn_num_layers must be >= 1 for spatial_mode='gnn', got {gnn_num_layers}"
                )

            if adjacency is None:
                adjacency = build_distance_weighted_adjacency(
                    channel_order=CHANNEL_ORDER,
                    channel_coords=CHANNEL_MNI_COORDS,
                    k=4,
                    sigma=40.0,
                    include_self=False,
                )
            else:
                adjacency = adjacency.float()

            if adjacency.shape != (n_channels, n_channels):
                raise ValueError(
                    f"Expected adjacency shape {(n_channels, n_channels)}, "
                    f"got {tuple(adjacency.shape)}"
                )

            adjacency = self._prepare_adjacency(
                adjacency,
                add_self_loops=add_self_loops,
                normalize=normalize_adjacency,
            )

            self.register_buffer("adjacency", adjacency)

            # Message-passing layers.
            # First layer maps temporal_embedding_dim -> spatial_hidden_dim.
            # Later layers keep spatial_hidden_dim -> spatial_hidden_dim.
            self.gnn_layers = nn.ModuleList()
            self.gnn_norms = nn.ModuleList()

            gnn_in_dim = temporal_embedding_dim

            for _ in range(gnn_num_layers):
                self.gnn_layers.append(
                    nn.Linear(gnn_in_dim, spatial_hidden_dim)
                )

                if use_layernorm:
                    self.gnn_norms.append(nn.LayerNorm(spatial_hidden_dim))
                else:
                    self.gnn_norms.append(nn.Identity())

                gnn_in_dim = spatial_hidden_dim

            self.gnn_dropout = nn.Dropout(dropout)

            # Location-preserving readout:
            # do NOT mean/max-pool over channels here.
            # Flattening keeps fixed electrode identity, which matters for coordinate prediction.
            gnn_flat_dim = n_channels * spatial_hidden_dim

            gnn_readout_layers = []

            if use_layernorm:
                gnn_readout_layers.append(nn.LayerNorm(gnn_flat_dim))

            gnn_readout_layers.extend([
                nn.Linear(gnn_flat_dim, spatial_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(spatial_hidden_dim, spatial_output_dim),
            ])

            self.gnn_output = nn.Sequential(*gnn_readout_layers)

    def _make_multiscale_block(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
    ) -> nn.ModuleDict:
        """
        Multi-scale temporal block.

        Branches:
            kernel 3   -> sharp local transients
            kernel 7   -> medium spike morphology
            kernel 15  -> broader slow-wave/context features
            pool + 1x1 -> local context branch

        The branches are concatenated, mixed with a 1x1 convolution,
        and added to a residual projection.
        """
        if out_channels % 4 != 0:
            raise ValueError(
                f"out_channels must be divisible by 4 for multiscale block, got {out_channels}"
            )

        branch_channels = out_channels // 4

        return nn.ModuleDict({
            "branch_k3": nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
            ),

            "branch_k7": nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    branch_channels,
                    kernel_size=7,
                    padding=3,
                ),
                nn.GELU(),
            ),

            "branch_k15": nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    branch_channels,
                    kernel_size=15,
                    padding=7,
                ),
                nn.GELU(),
            ),

            "branch_pool": nn.Sequential(
                nn.MaxPool1d(
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Conv1d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                ),
                nn.GELU(),
            ),

            "mix": nn.Sequential(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            ),

            "residual": (
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                )
                if in_channels != out_channels
                else nn.Identity()
            ),
        })

    def _run_multiscale_block(
        self,
        x: torch.Tensor,
        block: nn.ModuleDict,
    ) -> torch.Tensor:
        h = torch.cat(
            [
                block["branch_k3"](x),
                block["branch_k7"](x),
                block["branch_k15"](x),
                block["branch_pool"](x),
            ],
            dim=1,
        )

        h = block["mix"](h)
        residual = block["residual"](x)

        return h + residual

    @staticmethod
    def _prepare_adjacency(
        adjacency: torch.Tensor,
        add_self_loops: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Prepare adjacency matrix for dense message passing.

        Args:
            adjacency:
                [C, C] adjacency matrix.

            add_self_loops:
                Whether to add identity connections.

            normalize:
                If True, use simple row normalization:
                    A_norm[i, j] = A[i, j] / sum_j A[i, j]

        Returns:
            adjacency: [C, C]
        """
        adjacency = adjacency.clone()

        if add_self_loops:
            eye = torch.eye(
                adjacency.shape[0],
                dtype=adjacency.dtype,
                device=adjacency.device,
            )
            adjacency = adjacency + eye

        if normalize:
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            adjacency = adjacency / degree

        return adjacency

    def _run_temporal_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]

        Returns:
            channel_embeddings: [B, C, temporal_embedding_dim]
        """
        batch_size, n_channels, n_time = x.shape

        x_temporal = x.reshape(batch_size * n_channels, 1, n_time)

        if self.temporal_mode == "simple":
            h = self.temporal_cnn(x_temporal)  # [B*C, 128, 1]

        elif self.temporal_mode == "multiscale":
            h = self.temporal_stem(x_temporal)                 # [B*C, 32, 128]

            h = self._run_multiscale_block(h, self.ms_block_1)  # [B*C, 64, 128]
            h = self.downsample_1(h)                            # [B*C, 64, 64]

            h = self._run_multiscale_block(h, self.ms_block_2)  # [B*C, 96, 64]
            h = self.downsample_2(h)                            # [B*C, 128, 32]

            h = self._run_multiscale_block(h, self.ms_block_3)  # [B*C, 128, 32]

            h = self.temporal_pool(h)                           # [B*C, 128, 1]

        else:
            raise RuntimeError(f"Invalid temporal_mode: {self.temporal_mode}")

        z = self.temporal_projection(h)                         # [B*C, D]

        channel_embeddings = z.reshape(
            batch_size,
            n_channels,
            self.temporal_embedding_dim,
        )                                                        # [B, C, D]

        return channel_embeddings

    def _run_fc_spatial(
        self,
        channel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            channel_embeddings: [B, C, D]

        Returns:
            out: [B, spatial_output_dim]
        """
        batch_size, n_channels, embedding_dim = channel_embeddings.shape

        spatial_input = channel_embeddings.reshape(
            batch_size,
            n_channels * embedding_dim,
        )                                                        # [B, C*D]

        out = self.spatial_mlp(spatial_input)                    # [B, spatial_output_dim]

        return out

    def _run_gnn_spatial(
        self,
        channel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Location-preserving dense GNN-style spatial mixing.

        Args:
            channel_embeddings: [B, C, D]

        Returns:
            out: [B, spatial_output_dim]

        Rationale:
            The GNN performs local anatomy-informed message passing between
            neighboring electrodes, but the final readout preserves fixed
            electrode identity by flattening [C, H] rather than globally pooling
            over channels. This is important for coordinate prediction, where
            left/right and lobe-specific scalp topography should not be erased.
        """
        h = channel_embeddings  # [B, C, D]

        for layer_idx, (linear, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            # adjacency: [C, C]
            # h:         [B, C, D]
            # agg:       [B, C, D]
            agg = torch.einsum("ij,bjd->bid", self.adjacency, h)

            update = linear(agg)
            update = norm(update)
            update = F.gelu(update)
            update = self.gnn_dropout(update)

            # Residual connection only when shapes match.
            # This keeps deeper GNNs stable without forcing an extra projection.
            if update.shape == h.shape:
                h = h + update
            else:
                h = update

        batch_size = h.shape[0]

        # Preserve channel identity.
        # [B, C, H] -> [B, C*H]
        h_flat = h.reshape(batch_size, -1)

        out = self.gnn_output(h_flat)  # [B, spatial_output_dim]

        return out

    def forward(
        self,
        x: torch.Tensor,
        return_channel_embeddings: bool = False,
    ):
        """
        Args:
            x:
                Tensor of shape [B, C, T].

            return_channel_embeddings:
                If True, return both the final spatial output and the
                per-channel temporal embeddings.

        Returns:
            If return_channel_embeddings is False:
                out: [B, spatial_output_dim]

            If return_channel_embeddings is True:
                out: [B, spatial_output_dim]
                channel_embeddings: [B, C, temporal_embedding_dim]
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [batch, channels, time], got {tuple(x.shape)}"
            )

        batch_size, n_channels, n_time = x.shape

        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {n_channels}"
            )

        if n_time != self.input_length:
            raise ValueError(
                f"Expected time dimension {self.input_length}, got {n_time}"
            )

        channel_embeddings = self._run_temporal_encoder(x)        # [B, C, D]

        if self.spatial_mode == "fc":
            out = self._run_fc_spatial(channel_embeddings)

        elif self.spatial_mode == "gnn":
            out = self._run_gnn_spatial(channel_embeddings)

        else:
            raise RuntimeError(f"Invalid spatial_mode: {self.spatial_mode}")

        if return_channel_embeddings:
            return out, channel_embeddings

        return out


# Attention-based bag pooling
class AttentionPooling(nn.Module):
    """
    Attention-based MIL pooling.
    Given per-instance embeddings H (B, N, D) returns:
       - pooled vector (B, D)
       - attention weights (B, N)
    Masking supported: mask (B, N) with 1 for real, 0 for padding. (useful for patients who have fewer spikes)
    """
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.dim = dim
        self.attn_v = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)   # produces score per instance
        )

    def forward(self, H, mask=None):
        # H: (B, N, D)
        scores = self.attn_v(H).squeeze(-1)      # (B, N)
        if mask is not None:
            # mask==0 -> padding -> set score to -inf so softmax ignores them
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=1)   # (B, N)
        pooled = (H * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return pooled, weights


def _masked_mean_pool(H, mask=None):
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        return (H * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    return H.mean(dim=1)

def _masked_mean_std_pool(H, mask=None, eps=1e-6):
    """
    Deterministic MIL pooling with first- and second-order bag statistics.

    Args:
        H:    (B, N, D) per-instance embeddings
        mask: (B, N), 1 for valid instances, 0 for padding

    Returns:
        pooled: (B, 2D) = concat(mean, std)
    """
    if mask is not None:
        m = mask.unsqueeze(-1).float()  # (B, N, 1)
        count = m.sum(dim=1).clamp(min=1.0)

        mean = (H * m).sum(dim=1) / count

        var = (((H - mean.unsqueeze(1)) ** 2) * m).sum(dim=1) / count
        std = torch.sqrt(var + eps)
    else:
        mean = H.mean(dim=1)
        std = H.std(dim=1, unbiased=False)

    return torch.cat([mean, std], dim=1)

def _mean_max_topk_pool(H, mask=None, topk_k=10):
    mean_pooled = _masked_mean_pool(H, mask)

    if mask is not None:
        valid_mask = mask.bool().unsqueeze(-1)
        masked_for_max = H.masked_fill(~valid_mask, float("-inf"))
        max_pooled = masked_for_max.max(dim=1).values
        max_pooled = torch.where(
            torch.isfinite(max_pooled),
            max_pooled,
            torch.zeros_like(max_pooled),
        )

        masked_for_topk = H.masked_fill(~valid_mask, float("-inf"))
        topk_count = min(topk_k, H.size(1))
        topk_values = torch.topk(masked_for_topk, k=topk_count, dim=1).values
        finite_topk = torch.isfinite(topk_values)
        topk_sum = topk_values.masked_fill(~finite_topk, 0.0).sum(dim=1)
        valid_topk = finite_topk.sum(dim=1).clamp(min=1)
        topk_mean = topk_sum / valid_topk
    else:
        max_pooled = H.max(dim=1).values
        topk_count = min(topk_k, H.size(1))
        topk_mean = torch.topk(H, k=topk_count, dim=1).values.mean(dim=1)

    return torch.cat([mean_pooled, max_pooled, topk_mean], dim=1)


class GaussianMixtureHead(nn.Module):
    """Predict a K-component Gaussian mixture in 3D space from bag embeddings."""

    def __init__(
        self,
        in_dim: int,
        num_gaussians: int = 3,
        coord_dim: int = 3,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        isotropic: bool = True,
        output_space: str = "normalized",
    ):
        super().__init__()
        if num_gaussians < 1:
            raise ValueError(f"num_gaussians must be >= 1, got {num_gaussians}")
        if coord_dim < 1:
            raise ValueError(f"coord_dim must be >= 1, got {coord_dim}")
        if output_space not in {"normalized", "mni_mm"}:
            raise ValueError(
                f"output_space must be 'normalized' or 'mni_mm', got {output_space!r}"
            )

        if sigma_min is None:
            sigma_min = 0.02 if output_space == "normalized" else 2.0
        if sigma_max is None:
            sigma_max = 0.25 if output_space == "normalized" else 100.0

        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be > 0, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(
                f"sigma_max must be > sigma_min, got sigma_min={sigma_min}, sigma_max={sigma_max}"
            )
        if output_space == "normalized" and sigma_max > 1.0:
            raise ValueError(
                f"sigma_max={sigma_max} is too large for output_space='normalized'. "
                "Use values in (0, 1] or switch output_space to 'mni_mm'."
            )

        self.num_gaussians = num_gaussians
        self.coord_dim = coord_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.isotropic = isotropic
        self.output_space = output_space

        sigma_dim = 1 if isotropic else coord_dim
        hidden = max(in_dim, 128)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.mu_head = nn.Linear(hidden, num_gaussians * coord_dim)
        self.sigma_head = nn.Linear(hidden, num_gaussians * sigma_dim)
        self.weight_head = nn.Linear(hidden, num_gaussians)

        if output_space == "mni_mm":
            # Fixed extent-based bounds matching normalized->mm conversion in training.
            bounds = torch.tensor([90.0, 126.0, 72.0], dtype=torch.float32)
            if coord_dim != 3:
                bounds = torch.ones(coord_dim, dtype=torch.float32)
            self.register_buffer("mni_bounds_mm", bounds)

    def forward(self, x: torch.Tensor):
        h = self.net(x)

        mu_raw = self.mu_head(h).view(-1, self.num_gaussians, self.coord_dim)
        mu_unit = torch.tanh(mu_raw)
        if self.output_space == "normalized":
            mu = mu_unit
        else:
            mu = mu_unit * self.mni_bounds_mm.view(1, 1, -1)

        sigma_dim = 1 if self.isotropic else self.coord_dim
        sigma_raw = self.sigma_head(h).view(-1, self.num_gaussians, sigma_dim)
        sigma_unit = torch.sigmoid(sigma_raw)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * sigma_unit

        logits = self.weight_head(h)
        weights = torch.softmax(logits, dim=-1)

        return {
            "mu": mu,
            "sigma": sigma,
            "logits": logits,
            "weights": weights,
        }


class ConvUpBlock3D(nn.Module):
    """3D upsampling block using trilinear interpolation + Conv3d."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        num_groups = min(8, out_ch)
        while out_ch % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        layers = [
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        layers.extend([
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.GELU(),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvSpatialHead(nn.Module):
    """Low-resolution 3D decoder that maps pooled embedding to voxel prior logits."""

    def __init__(
        self,
        in_dim: int,
        output_shape: tuple[int, int, int] = (32, 40, 32),
        latent_shape: tuple[int, int, int] = (4, 5, 4),
        base_channels: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_shape = tuple(int(v) for v in output_shape)
        self.latent_shape = tuple(int(v) for v in latent_shape)
        self.base_channels = int(base_channels)

        if any(v <= 0 for v in self.output_shape):
            raise ValueError(f"output_shape must be positive, got {self.output_shape}")
        if any(v <= 0 for v in self.latent_shape):
            raise ValueError(f"latent_shape must be positive, got {self.latent_shape}")
        if self.base_channels < 8:
            raise ValueError(f"base_channels must be >= 8, got {self.base_channels}")

        lx, ly, lz = self.latent_shape
        flat_dim = self.base_channels * lx * ly * lz

        self.fc = nn.Sequential(
            nn.Linear(in_dim, flat_dim),
            nn.GELU(),
            nn.LayerNorm(flat_dim),
        )

        c1 = max(self.base_channels // 2, 8)
        c2 = max(c1 // 2, 8)
        c3 = max(c2 // 2, 8)

        self.decoder = nn.Sequential(
            ConvUpBlock3D(self.base_channels, c1, dropout=dropout),
            ConvUpBlock3D(c1, c2, dropout=dropout),
            ConvUpBlock3D(c2, c3, dropout=dropout),
        )
        self.out_conv = nn.Conv3d(c3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        lx, ly, lz = self.latent_shape

        h = self.fc(x)
        h = h.view(batch_size, self.base_channels, lx, ly, lz)
        h = self.decoder(h)

        if tuple(h.shape[-3:]) != self.output_shape:
            h = F.interpolate(
                h,
                size=self.output_shape,
                mode="trilinear",
                align_corners=False,
            )

        logits = self.out_conv(h)
        prob = torch.sigmoid(logits)
        return {
            "logits": logits,
            "prob": prob,
        }


# -----------------------------------------------
# [EXPERIMENTAL] Classification-only MIL model
# Used by eeg_spike_mil_training.py (deprecated training script).
# -----------------------------------------------

# Full bag-level model (single-label classification)
class SpikeMILClassifier(nn.Module):
    """
    Full model:
      - Configurable spike encoder -> per-spike embeddings (B, N, D)
      - Configurable bag pooling -> pooled (B, D) + optional weights
      - Classifier MLP -> logits (B, num_classes)
    """
    def __init__(
        self,
        in_channels=21,
        emb_dim=32,
        hidden=32,
        num_classes=12,
        dropout=0.3,
        pooling="mean",
        encoder_type="t_s_cnn",
    ):
        super().__init__()

        valid_pooling  = ("attention", "mean", "mean-std", "mean-max-topk")
        valid_encoders = ("t_s_cnn",)
        assert pooling in valid_pooling, (
            f"pooling must be one of {valid_pooling}, got {pooling!r}"
        )
        assert encoder_type in valid_encoders, (
            f"encoder_type must be one of {valid_encoders}, got {encoder_type!r}"
        )

        self.pooling = pooling
        self.encoder_type = encoder_type
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.dropout = dropout
        self.topk_k = 10

        self.encoder = SpikeEncoder_T_S(
            n_channels=in_channels,
            spatial_output_dim=emb_dim,
            dropout=dropout,
        )

        self.pool = AttentionPooling(emb_dim, hidden=hidden) if pooling == "attention" else None

        pooled_dim = emb_dim * 3 if pooling == "mean-max-topk" else emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)   # logits
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, N, C, L) → (B, N, D) via SpikeEncoder_T_S's flat (B, C, L) API."""
        B, N, C, L = x.shape
        return self.encoder(x.reshape(B * N, C, L)).view(B, N, -1)

    def forward(self, x, mask=None):
        """
        x: (B, N, C, L)
        mask: (B, N) with 1 for real spikes, 0 for pad. If None, all are real.
        Returns:
          logits: (B, num_classes)
          attn_weights: (B, N)
        """
        H = self._encode(x)                      # (B, N, D)

        if self.pooling == "attention":
            pooled, attn_weights = self.pool(H, mask)
        elif self.pooling == "mean":
            pooled = _masked_mean_pool(H, mask)
            attn_weights = None
        else:
            pooled = _mean_max_topk_pool(H, mask, topk_k=self.topk_k)
            attn_weights = None

        logits = self.classifier(pooled)         # (B, num_classes)
        return logits, attn_weights


# -----------------------------------------------
# [EXPERIMENTAL] Coordinate regression MIL model  
# Used by eeg_spike_mil_regression_training.py (deprecated training script).
# -----------------------------------------------

# -----------------------------------------------
# Bag-level model for spatial regression of epileptogenic zone
# -----------------------------------------------
class SpikeMILRegressor(nn.Module):
    """
    MIL model for EEG-based spatial regression of epileptogenic zone.

    Predicts:
      - mu_hat: (B, 3)     normalized MNI coordinates (x,y,z)
      - log_sigma_hat: (B, 3) log std-dev per axis (heteroscedastic uncertainty)

    Uses:
      - t_s_cnn spike encoder
      - Configurable bag pooling: attention, mean, mean-std, or concat(mean, max, topk_mean)
    """
    def __init__(
        self,
        in_channels=21,
        emb_dim=32,
        hidden=32,
        dropout=0.3,
        pooling="mean",
        encoder_type="t_s_cnn",
    ):
        super().__init__()

        valid_pooling  = ("attention", "mean", "mean-std", "mean-max-topk")
        valid_encoders = ("t_s_cnn",)
        assert pooling in valid_pooling, (
            f"pooling must be one of {valid_pooling}, got {pooling!r}"
        )
        assert encoder_type in valid_encoders, (
            f"encoder_type must be one of {valid_encoders}, got {encoder_type!r}"
        )
        self.pooling = pooling
        self.encoder_type = encoder_type
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.dropout = dropout
        self.topk_k = 10

        self.encoder = SpikeEncoder_T_S(
            n_channels=in_channels,
            spatial_output_dim=emb_dim,
            dropout=dropout,
        )

        # Attention pooling (only instantiated when needed)
        self.pool = AttentionPooling(emb_dim, hidden=hidden) if pooling == "attention" else None

        pooled_dim = emb_dim * 3 if pooling == "mean-max-topk" else emb_dim

        # Shared trunk after pooling
        self.trunk = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden)
        )

        # Mean head (μx, μy, μz)
        self.mu_head = nn.Linear(hidden, 3)

        # Log-std head (log σx, log σy, log σz)
        self.log_sigma_head = nn.Linear(hidden, 3)

        # Initialize log_sigma to something reasonable
        nn.init.constant_(self.log_sigma_head.bias, -1.0)  # exp(-1) ≈ 0.37

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, N, C, L) → (B, N, D) via SpikeEncoder_T_S's flat (B, C, L) API."""
        B, N, C, L = x.shape
        return self.encoder(x.reshape(B * N, C, L)).view(B, N, -1)

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, N, C, L) EEG spikes
        mask : torch.Tensor or None
            Shape (B, N), 1 for valid spikes, 0 for padding

        Returns
        -------
        mu_hat : torch.Tensor
            Shape (B, 3), normalized coordinates
        log_sigma_hat : torch.Tensor
            Shape (B, 3), log std-dev per axis
        attn_weights : torch.Tensor
            Shape (B, N), attention weights over spikes
        """
        H = self._encode(x)                     # (B, N, D)

        if self.pooling == "attention":
            pooled, attn_weights = self.pool(H, mask)  # (B, D), (B, N)
        elif self.pooling == "mean":
            pooled = _masked_mean_pool(H, mask)
            attn_weights = None
        elif self.pooling == "mean-std":
            pooled = _masked_mean_std_pool(H, mask)
            attn_weights = None
        else:
            pooled = _mean_max_topk_pool(H, mask, topk_k=self.topk_k)
            attn_weights = None

        h = self.trunk(pooled)                  # (B, hidden)

        mu_hat = self.mu_head(h)                # (B, 3)
        log_sigma_hat = self.log_sigma_head(h)  # (B, 3)

        return mu_hat, log_sigma_hat, attn_weights


class SpikeMILModel(nn.Module):
    """
    Multi-head MIL model for EEG spike localization.

    Each head is optional and controlled by the ``use_*_head`` flags:
      - coordinate regression  (mu, log_sigma): (B, 3)  [use_coord_head]
      - hemisphere classification:               (B, n_hemi_classes)  [use_hemi_head]
      - lobe classification:                     (B, n_lobe_classes)  [use_lobe_head]
            - Gaussian mixture spatial prior:          dict(mu, sigma, logits, weights)
                [spatial_head='gaussian_mixture']

    Disabled heads are completely absent from the model (no parameters, no output).
    The corresponding key in the forward output dict will be ``None``.
    """

    def __init__(
        self,
        in_channels: int = 21,
        emb_dim: int = 64,
        hidden: int = 64,
        dropout: float = 0.3,
        n_hemi_classes: int = 2,
        n_lobe_classes: int = 4,
        encoder_type: str = "t_s_cnn",
        pooling: str = "mean",
        use_coord_head: bool = True,
        use_hemi_head: bool = True,
        use_lobe_head: bool = True,
        spatial_head: str = "none",
        num_gaussians: int = 3,
        gaussian_coord_dim: int = 3,
        gaussian_sigma_min: Optional[float] = None,
        gaussian_sigma_max: Optional[float] = None,
        gaussian_isotropic: bool = True,
        gaussian_output_space: str = "normalized",
        gaussian_make_heatmap: bool = False,
        gaussian_heatmap_shape: Optional[tuple[int, int, int]] = None,
        deconv_output_shape: tuple[int, int, int] = (32, 40, 32),
        deconv_latent_shape: tuple[int, int, int] = (4, 5, 4),
        deconv_base_channels: int = 128,
        deconv_dropout: float = 0.0,
    ):
        super().__init__()

        valid_pooling  = ("attention", "mean", "mean-std", "mean-max-topk")
        valid_encoders = ("t_s_cnn",)
        assert pooling in valid_pooling, (
            f"pooling must be one of {valid_pooling}, got {pooling!r}"
        )
        assert encoder_type in valid_encoders, (
            f"encoder_type must be one of {valid_encoders}, got {encoder_type!r}"
        )
        if not (use_coord_head or use_hemi_head or use_lobe_head or spatial_head != "none"):
            raise ValueError("At least one head must be enabled.")

        self.pooling = pooling
        self.encoder_type = encoder_type
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.dropout = dropout
        self.topk_k = 10
        self.use_coord_head = use_coord_head
        self.use_hemi_head = use_hemi_head
        self.use_lobe_head = use_lobe_head
        if spatial_head not in {"none", "coordinate", "gaussian_mixture", "deconv"}:
            raise ValueError(
                "spatial_head must be one of {'none', 'coordinate', 'gaussian_mixture', 'deconv'}, "
                f"got {spatial_head!r}"
            )
        self.spatial_head = spatial_head
        self.num_gaussians = num_gaussians
        self.gaussian_coord_dim = gaussian_coord_dim
        self.gaussian_sigma_min = gaussian_sigma_min
        self.gaussian_sigma_max = gaussian_sigma_max
        self.gaussian_isotropic = gaussian_isotropic
        self.gaussian_output_space = gaussian_output_space
        self.gaussian_make_heatmap = gaussian_make_heatmap
        self.gaussian_heatmap_shape = gaussian_heatmap_shape
        self.use_gaussian_mixture_head = spatial_head == "gaussian_mixture"
        self.deconv_output_shape = tuple(int(v) for v in deconv_output_shape)
        self.deconv_latent_shape = tuple(int(v) for v in deconv_latent_shape)
        self.deconv_base_channels = int(deconv_base_channels)
        self.deconv_dropout = float(deconv_dropout)
        self.use_deconv_spatial_head = spatial_head == "deconv"

        self.encoder = SpikeEncoder_T_S(
            n_channels=in_channels,
            spatial_output_dim=emb_dim,
            dropout=dropout,
        )

        self.pool = AttentionPooling(emb_dim, hidden=hidden) if pooling == "attention" else None

        if pooling == "mean-max-topk":
            pooled_dim = emb_dim * 3 
        elif pooling == "mean-std":
            pooled_dim = emb_dim * 2
        else:
            pooled_dim = emb_dim

        self.trunk = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if use_coord_head:
            self.mu_head = nn.Linear(hidden, 3)
            # self.log_sigma_head = nn.Linear(hidden, 3)
            # nn.init.constant_(self.log_sigma_head.bias, -1.0)
            self.log_sigma_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 3),
            )

            nn.init.constant_(self.log_sigma_head[-1].bias, -1.0)

        if use_hemi_head:
            self.hemi_head = nn.Linear(hidden, n_hemi_classes)

        if use_lobe_head:
            self.lobe_head = nn.Linear(hidden, n_lobe_classes)

        if self.use_gaussian_mixture_head:
            self.gaussian_mixture_head = GaussianMixtureHead(
                in_dim=hidden,
                num_gaussians=num_gaussians,
                coord_dim=gaussian_coord_dim,
                sigma_min=gaussian_sigma_min,
                sigma_max=gaussian_sigma_max,
                isotropic=gaussian_isotropic,
                output_space=gaussian_output_space,
            )
        if self.use_deconv_spatial_head:
            self.deconv_spatial_head = DeconvSpatialHead(
                in_dim=hidden,
                output_shape=self.deconv_output_shape,
                latent_shape=self.deconv_latent_shape,
                base_channels=self.deconv_base_channels,
                dropout=self.deconv_dropout,
            )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, N, C, L) → (B, N, D), handling t_s_cnn's flat (B, C, L) API."""
        if self.encoder_type == "t_s_cnn":
            B, N, C, L = x.shape
            return self.encoder(x.reshape(B * N, C, L)).view(B, N, -1)
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_embeddings: bool = False,
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input spikes with shape (B, N, C, L).
        mask : torch.Tensor | None
            Optional valid-spike mask with shape (B, N), where 1=valid and 0=pad.
        """
        H = self._encode(x)  # (B, N, D)

        if self.pooling == "attention":
            pooled, attn_weights = self.pool(H, mask)
        elif self.pooling == "mean":
            pooled = _masked_mean_pool(H, mask)
            attn_weights = None
        elif self.pooling == "mean-std":
            pooled = _masked_mean_std_pool(H, mask)
            attn_weights = None
        else:
            pooled = _mean_max_topk_pool(H, mask, topk_k=self.topk_k)
            attn_weights = None

        h = self.trunk(pooled)

        mu_hat        = self.mu_head(h) if self.use_coord_head else None
        log_sigma_hat = self.log_sigma_head(h) if self.use_coord_head else None
        hemi_logits = self.hemi_head(h) if self.use_hemi_head else None
        lobe_logits = self.lobe_head(h) if self.use_lobe_head else None
        gaussian_mixture = self.gaussian_mixture_head(h) if self.use_gaussian_mixture_head else None
        deconv_spatial = self.deconv_spatial_head(h) if self.use_deconv_spatial_head else None

        out = {
            "mu":          mu_hat,
            "log_sigma":   log_sigma_hat,
            "hemi_logits": hemi_logits,
            "lobe_logits": lobe_logits,
            "attn_weights": attn_weights,
            "gaussian_mixture": gaussian_mixture,
            "deconv_spatial": deconv_spatial,
        }

        if return_embeddings:
            out["pooled_embedding"] = pooled
            out["embedding"] = h

        if return_features:
            # `instance_embeddings` is the exact tensor immediately before MIL pooling.
            # `pooled_embedding` is the direct output immediately after MIL pooling.
            out["features"] = {
                "instance_embeddings": H,
                "pooled_embedding": pooled,
            }

        return out

# ---------------------------------------------------------------------------
# Flat-dataset spike encoder pretraining wrapper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Gradient Reversal Layer (for domain-adversarial / subject-adversarial training)
# ---------------------------------------------------------------------------


class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer.  Forward: identity.  Backward: scale by ``-λ``."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    """Apply gradient reversal with scaling factor *lambd*."""
    return GradReverse.apply(x, lambd)


class SpikeEncoderPretrainModel(nn.Module):
    """
    Thin wrapper around the shared spike encoder for flat-segment pretraining.

    Accepts individual segments ``(B, C, L)``, adds the bag dimension N=1
    expected by the encoders, then removes it so the prediction heads receive
    ``(B, emb_dim)``.

    Because ``self.encoder`` mirrors exactly the ``encoder`` sub-module used
    by :class:`SpikeMILClassifier` / :class:`SpikeMILRegressor`, the encoder
    state dict can be transferred directly into those models after pretraining.

    Outputs
    -------
    perception_pred  : ``(B,)``        raw scalar (no activation applied)
    channel_logits   : ``(B, n_ch)``   raw logits for ``BCEWithLogitsLoss``
    embeddings       : ``(B, emb_dim)``
    """

    def __init__(
        self,
        in_channels: int = 21,
        emb_dim: int = 32,
        hidden: int = 64,
        dropout: float = 0.3,
        n_channels: int = 21,
        encoder_type: str = "t_s_cnn",
        window_size: int = 128,
        use_subject_adversary: bool = False,
        num_subjects: Optional[int] = None,
        subject_adv_hidden_dim: int = 64,
        subject_adv_dropout: float = 0.2,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoder_type = encoder_type
        self.n_channels = n_channels
        self.window_size = window_size
        self.use_subject_adversary = use_subject_adversary

        if use_subject_adversary and num_subjects is None:
            raise ValueError("num_subjects must be provided when use_subject_adversary=True")

        if encoder_type == "t_s_cnn":
            self.encoder = SpikeEncoder_T_S(
                n_channels=in_channels,
                input_length=window_size,
                spatial_output_dim=emb_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"encoder_type must be 't_s_cnn', got {encoder_type!r}"
            )

        self.perception_head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

        self.channel_head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_channels),
        )

        if use_subject_adversary:
            self.subject_head = nn.Sequential(
                nn.Linear(emb_dim, subject_adv_hidden_dim),
                nn.ReLU(),
                nn.Dropout(subject_adv_dropout),
                nn.Linear(subject_adv_hidden_dim, num_subjects),
            )
        else:
            self.subject_head = None

    def forward(self, x: torch.Tensor, subject_adv_lambda: float = 0.0):
        """
        Parameters
        ----------
        x : torch.Tensor  (B, C, L)
        subject_adv_lambda : float
            Scaling factor for the gradient reversal layer.  Ignored (and no
            subject logits produced) when ``use_subject_adversary=False``.

        Returns
        -------
        perception_pred : (B,)
        channel_logits  : (B, n_channels)
        embeddings      : (B, emb_dim)
        subject_logits  : (B, num_subjects) or None when adversary disabled
        """
        # SpikeEncoder_T_S accepts (B, C, L) directly
        emb = self.encoder(x)            # (B, emb_dim)

        perception_pred = self.perception_head(emb).squeeze(-1)   # (B,)
        channel_logits  = self.channel_head(emb)                  # (B, n_ch)

        subject_logits = None
        if self.use_subject_adversary and self.subject_head is not None:
            subject_logits = self.subject_head(grad_reverse(emb, subject_adv_lambda))

        return perception_pred, channel_logits, emb, subject_logits


# Test usage
if __name__ == "__main__":

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------
    # Full model tests
    # -----------------------------------------------
    B = 4               # batch size (patients)
    N = 128             # max spikes per patient
    C = 21              # channels
    L = 128             # length of each spike segment (in samples)
    num_classes = 12    # e.g., 12 localization classes (L/R: frontal, temporal, parietal, occipital, insular)

    # Instantiate model and create dummy input
    model_classifier = SpikeMILClassifier(in_channels=C, num_classes=num_classes).to(device)
    model_regressor = SpikeMILRegressor(in_channels=C).to(device)
    x = torch.randn(B, N, C, L).to(device)

    # Masks: test patient with fewer spikes
    lengths = torch.tensor([128, 128, 128, 89]).to(device)
    mask = (torch.arange(N).unsqueeze(0) < lengths.unsqueeze(1)).to(torch.long).to(device)  # (B,N) 1/0

    # Forward classifier pass
    logits, attn = model_classifier(x, mask=mask)
    print("logits", logits.shape)           # (B, num_classes)
    if attn is not None:
        print("attn", attn.shape)               # (B, N)
    # Test loss computation
    targets = torch.tensor([0, 3, 1, 7], dtype=torch.long).to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    print("loss", loss.item())

    # Forward regressor pass
    mu_hat, log_sigma_hat, attn = model_regressor(x, mask=mask)
    print("mu_hat", mu_hat.shape)               # (B, 3)
    print("log_sigma_hat", log_sigma_hat.shape) # (B, 3)
    if attn is not None:
        print("attn", attn.shape)                   # (B, N)
    # Test loss computation (negative log-likelihood)
    mu_target = torch.randn(B, 3).to(device)
    sigma_target = torch.abs(torch.randn(B, 3)).to(device) + 1e-3  # avoid zero std
    sigma = torch.exp(log_sigma_hat) + 1e-3
    loss = (0.5 * ((mu_target - mu_hat) ** 2 / (sigma ** 2) + 2 * log_sigma_hat)).mean()
    print("NLL loss", loss.item())

    # Forward multi-head pass
    for encoder_type in ["t_s_cnn"]:
        model_multihead = SpikeMILModel(in_channels=C, emb_dim=64, encoder_type=encoder_type, pooling="mean-std").to(device)
        out = model_multihead(x, mask=mask)
        assert out["mu"].shape == (B, 3), f"Expected (B, 3), got {out['mu'].shape}"
        assert out["log_sigma"].shape == (B, 3), (
            f"Expected (B, 3), got {out['log_sigma'].shape}"
        )
        assert out["hemi_logits"].shape == (B, 2), (
            f"Expected (B, 2), got {out['hemi_logits'].shape}"
        )
        assert out["lobe_logits"].shape[0] == B, (
            f"Expected batch dim {B}, got {out['lobe_logits'].shape}"
        )
        if out["attn_weights"] is not None:
            assert out["attn_weights"].shape == (B, N), (
                f"Expected (B, N), got {out['attn_weights'].shape}"
            )
        print(f"SpikeMILModel with {encoder_type} encoder smoke test passed")
    
    # Forward pretrain model pass
    pretrain_model = SpikeEncoderPretrainModel(in_channels=C, emb_dim=64, encoder_type="t_s_cnn").to(device)
    perception_pred, channel_logits, embeddings, subject_logits = pretrain_model(x[:, 0])  # (B, C, L)
    assert perception_pred.shape == (B,), f"Expected (B,), got {perception_pred.shape}"
    assert channel_logits.shape == (B, C), f"Expected (B, C), got {channel_logits.shape}"
    assert embeddings.shape == (B, 64), f"Expected (B, 64), got {embeddings.shape}"
    assert subject_logits is None, "Expected None when use_subject_adversary=False"
    print("SpikeEncoderPretrainModel smoke test passed")