"""
training/eeg_spike_mil_training_v2.py

Core EEG spike localization training script for SpikeMILModel_v2.
This is NOT the original EEG MIL model described in our manuscript; it is the updated version 2.
For the original EEG MIL model training, see training/eeg_spike_mil_mh_training.py.

Trains SpikeMILModel_v2 with:
  - Spike encoder:  SpikeEncoder_v2 (temporal 1D CNN + 2D spatial mixing)
  - MIL pooling:    Attention or mean pooling over spikes per patient
  - Head:           Per-channel prediction of underlying irritative zone.

After training, inference is run automatically via run_inference().
For standalone inference see inference/eeg_spike_mil_inference_v2.py.
"""