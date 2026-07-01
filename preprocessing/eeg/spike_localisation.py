import numpy as np
from scipy.ndimage import uniform_filter1d
import mkl

def pick_spike_index(sharp, steep, energy, mid_start, mid_end,
                     fs=256, weights=[2., 2., 1.]):

    # Extract mid window
    sharp_seg = sharp[:, mid_start:mid_end]    # (C, M)
    steep_seg = steep[:, mid_start:mid_end]    # (C, M)
    energy_seg = energy[:, mid_start:mid_end]  # (C, M)

    C, M = sharp_seg.shape

    # Window sizes (odd enforced later)
    win_sharp = max(1, int(round(10  /1000.0*fs)))
    win_steep = max(1, int(round(75  /1000.0*fs)))
    win_energy= max(1, int(round(100 /1000.0*fs)))

    def make_odd(k):
        return k if k % 2 == 1 else k+1

    win_sharp  = make_odd(win_sharp)
    win_steep  = make_odd(win_steep)
    win_energy = make_odd(win_energy)

    def smooth(x, k):
        if k <= 1:
            return x
        pad = k//2
        x_pad = np.pad(x, (pad, pad), mode="edge")
        return np.convolve(x_pad, np.ones(k), mode="valid")

    def z(v, axis=1):
        m = v.mean(axis=axis, keepdims=True)
        s = v.std(axis=axis, keepdims=True) + 1e-6
        return (v - m) / s

    # Smooth per channel
    sharp_s  = np.stack([smooth(sharp_seg[c],  win_sharp)  for c in range(C)])
    steep_s  = np.stack([smooth(steep_seg[c],  win_steep)  for c in range(C)])
    energy_s = np.stack([smooth(energy_seg[c], win_energy) for c in range(C)])

    # Z-score per channel
    sharp_z  = z(sharp_s)
    steep_z  = z(steep_s)
    energy_z = z(energy_s)

    # Channel-wise score
    score_ch = (
        weights[0]*sharp_z +
        weights[1]*steep_z +
        weights[2]*energy_z
    )    # shape (C, M)

    # 80-th percentile across channels = final spikiness curve
    score = np.percentile(score_ch, 80, axis=0)   # (M,)

    # Best index inside mid window
    best_offset = int(np.argmax(score))
    return mid_start + best_offset

def compute_spikiness(seg, fs=256):
    """
    Compute per-channel sharpness, steepness and energy curves.

    Returns:
        sharp : (C, L)
        steep : (C, L)
        energy: (C, L)
    """
    C, L = seg.shape

    # Remove DC
    x = seg - seg.mean(axis=1, keepdims=True)

    # 5 Hz high-pass via moving average subtraction
    hp_win = max(1, int(0.2 * fs))
    x_hp = x - uniform_filter1d(x, size=hp_win, axis=1, mode="nearest")

    # First derivative (steepness)
    d1 = np.diff(x_hp, n=1, axis=1)
    steep = np.abs(d1)
    steep = np.pad(steep, ((0,0),(0,1)))

    # Second derivative (sharpness)
    d2 = np.diff(x_hp, n=2, axis=1)
    sharp = np.abs(d2)
    sharp = np.pad(sharp, ((0,0),(1,1)))

    # Local energy
    win = 5
    kernel = np.ones(win) / win
    energy = np.array([
        np.convolve(x_hp[c]**2, kernel, mode="same")
        for c in range(C)
    ])

    return sharp.astype(np.float32), steep.astype(np.float32), energy.astype(np.float32)

def extract_aligned_window(seg, fs):
    """
    Extract 2-second window aligned on the most spiky point in the middle 1 second.
    Having to do this manually because Persyst outputs its detections in 1-second bins, so we'll
    center on the most spiky point within that second. This way, we can rely on clinically verified
    spike detections while still centering well on the actual spike.
    
    seg: (C, 3*fs) array (your extracted segment around detection)
    Returns: (C, 2*fs) aligned window.
    """

    # Set MKL threads to 1 for this thread
    mkl.set_num_threads(1)

    # Check shape
    C, L = seg.shape
    assert L >= int(3*fs)

    # Compute spikiness
    sharpness, steepness, energy = compute_spikiness(seg, fs)

    # Search only within the middle 1 second
    mid_start = int(1.0 * fs)
    mid_end   = int(2.0 * fs)

    spike_t = pick_spike_index(sharpness, steepness, energy, mid_start, mid_end)

    # Extract 2-second window centered on spike_t
    half = int(fs)  # 1s left, 1s right
    start = spike_t - half
    end   = spike_t + half

    # Pad if needed (edge handling)
    pad_left = max(0, -start)
    pad_right = max(0, end - L)

    start = max(0, start)
    end   = min(L, end)

    cropped = seg[:, start:end]

    if pad_left or pad_right:
        cropped = np.pad(
            cropped,
            ((0,0), (pad_left, pad_right)),
            mode="constant"
        )

    return cropped.astype(np.float32)