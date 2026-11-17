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
New:
Input: (batch, spikes, 21 channels, 128 samples)
Output: (batch, 21, 3) joint probabilities, summing to one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, 16, k, padding=k // 2) for k in (3, 7, 15)
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
    """Shared per-channel 1D CNN; one 32D feature vector per channel."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 7, padding=3), nn.GELU())
        self.ms1 = MultiScaleBlock(32, 64)
        self.ms2 = MultiScaleBlock(64, 32)
        self.down = nn.MaxPool1d(2)

    def forward(self, x):
        batch, channels, samples = x.shape
        h = self.stem(x.reshape(batch * channels, 1, samples))
        h = self.down(self.ms1(h))
        h = self.down(self.ms2(h))
        h = h.mean(dim=-1)
        return h.reshape(batch, channels, 32)


class BipolarTemporalEncoder(nn.Module):
    """Encode all directed channel differences, then aggregate by electrode."""

    def __init__(self, pair_chunk_size=32):
        super().__init__()
        self.pair_chunk_size = pair_chunk_size
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
        )
        self.pair_projection = nn.Linear(16, 32)
        self.channel_projection = nn.Linear(21 * 64, 32)

    def _encode_traces(self, traces):
        features = self.temporal(traces)
        return self.pair_projection(features.mean(dim=-1))

    def forward(self, x):
        batch, channels, samples = x.shape
        if channels != 21:
            raise ValueError(f'Expected 21 EEG channels, got {channels}.')

        # Flattened pair indices preserve all 21 x 21 directed differences.
        pair_i = torch.arange(channels, device=x.device).repeat_interleave(channels)
        pair_j = torch.arange(channels, device=x.device).repeat(channels)
        pair_chunks = []
        for start in range(0, channels * channels, self.pair_chunk_size):
            end = min(start + self.pair_chunk_size, channels * channels)
            traces = (x[:, pair_i[start:end], :] - x[:, pair_j[start:end], :])
            traces = traces.reshape(batch * (end - start), 1, samples)
            if self.training and torch.is_grad_enabled():
                encoded = checkpoint(self._encode_traces, traces, use_reentrant=False)
            else:
                encoded = self._encode_traces(traces)
            pair_chunks.append(encoded.reshape(batch, end - start, 32))

        pair_features = torch.cat(pair_chunks, dim=1).reshape(
            batch, channels, channels, 32)
        pairs = torch.cat((pair_features, pair_features.transpose(1, 2)), dim=-1)
        diagonal = torch.eye(channels, dtype=torch.bool, device=x.device)
        pairs = pairs.masked_fill(diagonal[None, :, :, None], 0.0)
        return self.channel_projection(pairs.flatten(2))


class PairwiseSpatialMixer(nn.Module):
    """Learn pairwise channel interactions from time-collapsed features."""

    def __init__(self):
        super().__init__()
        self.left = nn.Linear(32, 12)
        self.right = nn.Linear(32, 12)
        self.channel_embed = nn.Parameter(torch.zeros(21, 12))
        self.pair = nn.Sequential(nn.Linear(144, 24), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(80, 32), nn.GELU(), nn.Dropout(0.2))
        self.norm = nn.LayerNorm(32)

    def forward(self, h):
        a = self.left(h) + self.channel_embed[None]
        b = self.right(h) + self.channel_embed[None]
        pairs = torch.einsum('bid,bje->bijde', a, b).flatten(-2)
        pairs = self.pair(pairs)
        row = pairs.mean(dim=2)
        col = pairs.mean(dim=1)
        return self.norm(h + self.mix(torch.cat([h, row, col], dim=-1)))


class MLPSpatialMixer(nn.Module):
    """Shallow global channel mixing, matched to the pairwise mixer."""

    def __init__(self):
        super().__init__()
        self.channel_mlp = nn.Sequential(nn.Linear(21, 64), nn.GELU(),
                                         nn.Linear(64, 21))
        self.feature_mlp = nn.Sequential(nn.Linear(32, 32), nn.GELU(),
                                         nn.Dropout(0.2))
        self.norm = nn.LayerNorm(32)

    def forward(self, h):
        mixed = self.channel_mlp(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h + self.feature_mlp(mixed))


class SpikeEncoder_v2(nn.Module):
    """Encode each spike into channel-aligned 32D feature vectors."""

    def __init__(self, spatial_mixing='pairwise', temporal_mode='raw'):
        super().__init__()
        mixers = {'pairwise': PairwiseSpatialMixer, 'mlp': MLPSpatialMixer}
        if spatial_mixing not in mixers:
            raise ValueError(f'Unknown spatial mixer: {spatial_mixing}')
        if temporal_mode not in {'raw', 'raw_bipolar'}:
            raise ValueError(f'Unknown temporal mode: {temporal_mode}')
        self.temporal_mode = temporal_mode
        self.temporal = TemporalEncoder_v2()
        if temporal_mode == 'raw_bipolar':
            self.bipolar_temporal = BipolarTemporalEncoder()
            self.bipolar_fusion = nn.Linear(64, 32)
        self.spatial = mixers[spatial_mixing]()

    def forward(self, x):
        raw_features = self.temporal(x)
        if self.temporal_mode == 'raw':
            return self.spatial(raw_features)
        bipolar_features = self.bipolar_temporal(x)
        fused = self.bipolar_fusion(torch.cat((raw_features, bipolar_features), dim=-1))
        return self.spatial(fused)


class SpikeMILModel_v2(nn.Module):
    """Masked-mean MIL with joint or factorized channel/depth outputs."""

    def __init__(self, spatial_mixing='pairwise', num_bins=3, mode='joint',
                 temporal_mode='raw'):
        super().__init__()
        if mode not in {'joint', 'factorized_depth'}:
            raise ValueError(f'Unknown output mode: {mode}')
        if mode == 'factorized_depth' and num_bins != 3:
            raise ValueError('factorized_depth mode requires exactly three depth bins.')
        self.mode = mode
        self.num_bins = num_bins
        self.temporal_mode = temporal_mode
        self.encoder = SpikeEncoder_v2(spatial_mixing=spatial_mixing,
                           temporal_mode=temporal_mode)
        if mode == 'joint':
            self.head = nn.Sequential(nn.Linear(32, 32), nn.GELU(),
                                      nn.Dropout(0.2), nn.Linear(32, num_bins))
        else:
            self.channel_head = nn.Sequential(
                nn.Linear(32, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1)
            )
            self.depth_head = nn.Sequential(
                nn.Linear(21 * 32, 64), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(64, 3),
            )

    def forward(self, x, mask=None):
        batch, spikes, channels, samples = x.shape
        h = self.encoder(x.reshape(batch * spikes, channels, samples))
        h = h.reshape(batch, spikes, channels, 32)
        if mask is None:
            h = h.mean(dim=1)
        else:
            valid = mask[:, :, None, None].to(h.dtype)
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        if self.mode == 'joint':
            logits = self.head(h)  # (B, 21, num_bins)
            return logits.flatten(1).softmax(dim=1).reshape_as(logits)

        channel_logits = self.channel_head(h).squeeze(-1)
        depth_logits = self.depth_head(h.flatten(1))
        return channel_logits.softmax(dim=1), depth_logits.softmax(dim=1)
