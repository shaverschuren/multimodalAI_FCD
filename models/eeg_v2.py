"""
models/eeg_v2.py

Multi-Instance Learning (MIL) model for EEG spike localization.
This is version 2 of the EEG MIL model, in which we update some architectural features
to improve localisation performance and hopefully enable feature level fusion
for a next publication. This is work in progress and NOT the EEG-based model as described
in our manuscript. For that, see models/eeg.py

Core architecture (SpikeMILModel_v2):
  - Spike encoder:  SpikeEncoder_v2 (temporal 1D CNN + pairwise or MLP-based spatial mixing)
  - MIL pooling:    Mean pooling over spikes per patient
  - Head:           Per-channel prediction of underlying irritative zone.

Input: (batch, spikes, 21 channels, 128 samples)
Output: (batch, 21) sensitivity-field overlap probabilities (sum to 1).

TODO:
  - Time-resolved spatial mixing for propagation patterns?
    Consider adding skip connections between temporal and spatial mixing layers?
  - Explore possibility of predicting both per-channel "weight" as a "distance" output
    to capture stuff like morphology that says something about the depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, 16, k, padding=k // 2)
            for k in (3, 7, 15)
        ])
        self.pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, 16, 1)
        )
        self.mix = nn.Conv1d(64, out_ch, 1)

    def forward(self, x):
        h = [F.gelu(conv(x)) for conv in self.convs]
        h.append(F.gelu(self.pool(x)))
        return F.gelu(self.mix(torch.cat(h, dim=1)))


class TemporalEncoder_v2(nn.Module):
    """Shared per-channel 1D CNN; collapses time to one 32D vector per channel."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 7, padding=3), nn.GELU())
        self.ms1 = MultiScaleBlock(32, 64)
        self.ms2 = MultiScaleBlock(64, 32)
        self.down = nn.MaxPool1d(2)

    def forward(self, x):
        # x: (batch*spikes, 21, 128)
        batch, channels, samples = x.shape
        h = self.stem(x.reshape(batch * channels, 1, samples))
        h = self.down(self.ms1(h))  # 128 -> 64
        h = self.down(self.ms2(h))  # 64 -> 32
        h = h.mean(dim=-1)          # Global temporal average pooling
        h = h.reshape(batch, channels, 32)
        return h


class PairwiseSpatialMixer(nn.Module):
    """Learn pairwise channel interactions from time-collapsed features. No time-resolved interactions yet..."""

    def __init__(self):
        super().__init__()
        self.left = nn.Linear(32, 12)
        self.right = nn.Linear(32, 12)
        self.channel_embed = nn.Parameter(torch.zeros(21, 12))
        self.pair = nn.Sequential(nn.Linear(144, 24), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(80, 32), nn.GELU(),
                                nn.Dropout(0.2))
        self.norm = nn.LayerNorm(32)

    def forward(self, h):
        # h: (batch*spikes, 21, 32)
        a = self.left(h) + self.channel_embed[None]
        b = self.right(h) + self.channel_embed[None]
        pairs = torch.einsum('bid,bje->bijde', a, b).flatten(-2)
        pairs = self.pair(pairs)  # (batch*spikes, 21, 21, 16)
        row = pairs.mean(dim=2)
        col = pairs.mean(dim=1)
        return self.norm(h + self.mix(torch.cat([h, row, col], dim=-1)))


class MLPSpatialMixer(nn.Module):
    """Shallow global channel mixing, matched in capacity to pairwise mixer."""

    def __init__(self):
        super().__init__()
        self.channel_mlp = nn.Sequential(nn.Linear(21, 64), nn.GELU(),
                                         nn.Linear(64, 21))
        self.feature_mlp = nn.Sequential(nn.Linear(32, 32), nn.GELU(),
                                         nn.Dropout(0.2))
        self.norm = nn.LayerNorm(32)

    def forward(self, h):
        # h: (batch*spikes, 21, 32)
        mixed = self.channel_mlp(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h + self.feature_mlp(mixed))


class SpikeEncoder_v2(nn.Module):
    """Encode each spike into channel-aligned 32D feature vectors."""

    def __init__(self, spatial_mixing='pairwise'):
        super().__init__()
        mixers = {'pairwise': PairwiseSpatialMixer, 'mlp': MLPSpatialMixer}
        if spatial_mixing not in mixers:
            raise ValueError(f'Unknown spatial mixer: {spatial_mixing}')
        self.temporal = TemporalEncoder_v2()
        self.spatial = mixers[spatial_mixing]()

    def forward(self, x):
        return self.spatial(self.temporal(x))


class SpikeMILModel_v2(nn.Module):
    """Masked-mean MIL with a shared per-channel prediction head."""

    def __init__(self, spatial_mixing='pairwise'):
        super().__init__()
        self.encoder = SpikeEncoder_v2(spatial_mixing=spatial_mixing)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x, mask=None):
        batch, spikes, channels, samples = x.shape
        h = self.encoder(x.reshape(batch * spikes, channels, samples))
        h = h.reshape(batch, spikes, channels, 32)
        if mask is None:
            h = h.mean(dim=1)
        else:
            valid = mask[:, :, None, None].to(h.dtype)
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        logits = self.head(h).squeeze(-1)
        return torch.softmax(logits, dim=-1)