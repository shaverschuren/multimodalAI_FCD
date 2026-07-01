import json
import mne

montage = mne.channels.make_standard_montage("standard_1005")
ch_pos = montage.get_positions()["ch_pos"]

out = {
    "source": {
        "citation": "Oostenveld R, Praamstra P. The five percent electrode system for high-resolution EEG and ERP measurements. Clinical Neurophysiology. 2001;112:713-719.",
        "coordinate_source": "MNE-Python built-in standard_1005 montage",
        "coordinate_system": "template/head coordinates; based on 10-05 system",
        "unit": "mm",
        "note": "MNE returns meters; values below are converted to millimetres."
    },
    "data": {
        ch: {
            "mni": {
                "x": float(pos[0] * 1000),
                "y": float(pos[1] * 1000),
                "z": float(pos[2] * 1000),
            }
        }
        for ch, pos in ch_pos.items()
    }
}

with open("L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\src\\preprocessing\\eeg\\standard_1005_mni_coordinates.json", "w") as f:
    json.dump(out, f, indent=2)