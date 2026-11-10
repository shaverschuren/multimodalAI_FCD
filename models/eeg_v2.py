"""
models/eeg_v2.py

Multi-Instance Learning (MIL) model for EEG spike localization.
This is version 2 of the EEG MIL model, in which we update some architectural features
to improve localisation performance and hopefully enable feature level fusion
for a next publication. This is work in progress and NOT the EEG-based model as described
in our manuscript. For that, see models/eeg.py

Core architecture (SpikeMILModel_v2):
  - Spike encoder:  SpikeEncoder_v2 (temporal 1D CNN + 2D spatial mixing)
  - MIL pooling:    Attention or mean pooling over spikes per patient
  - Head:           Per-channel prediction of underlying irritative zone.

TODO:
  - Explore possibility of mixing BEFORE collapsing the temporal dimension
    to potentially capture stuff like propagation patterns better.
    Consider adding skip connections between temporal and spatial mixing layers?
  - Explore possibility of predicting both per-channel "weight" as a "distance" output
    to capture stuff like morphology that says something about the depth.
"""