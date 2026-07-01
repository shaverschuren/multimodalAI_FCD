"""
Batch conversion of Harmonie .STS/.SIG EEG recordings to EDF format.

Uses the Snooz HarmonieReader C++ extension (compiled as a Python .pyd) to read
proprietary Harmonie files, and MNE-Python to write standard EDF output. Check out
`https://github.com/SnoozToolbox/harmonie_reader.git` for the reader code and build instructions.
The included binary reader is built for ``Windows 11 64-bit`` and ``Python 3.14.5``, but you can recompile it for other platforms if needed.

- Uses temporary memory-mapped files for large recordings to avoid OOM stuff, but this is pretty slow
so it can be disabled using ``--no-memmap`` flag for small files or machines with a lot of RAM.
- Use the ``--copy-source-file-to-tmp`` flag to speed up reading if the source files are on a slow (network) drive and
the tmp directory is on a fast (local) drive.
- Warning: Overwrite is enabled by default, use the ``--no-overwrite`` flag to disable overwriting existing EDFs. 

Usage
-------------
Convert an entire folder tree of patient recordings::

    python sig2edf_batch.py \\
        --input-root  "D:/sig_eeg" \\
        --output-root "D:/edf_eeg" \\
        --harmonie-reader-root "C:/Users/me/harmonie_reader"

Make sure the .sig and .sts files are together and named identically in the input folders.
The script will find all .STS files (case-insensitive) and look for matching .SIG/.sig files next to them.

The output folder will mirror the input structure, but with .edf files instead of .STS/.SIG. For example:
    D:/sig_eeg/RESP0400/file.STS
    -> D:/edf_eeg/RESP0400/file.edf
        
Dependencies
------------
- Python 3.14 or later
- MNE-Python (``mne``) incl EDF export support (requires `edfio` installed)
- NumPy (``numpy``)
- Compiled HarmonieReader .pyd (from the Snooz harmonie_reader repository, pre-compiled Windows binary included)

Author
-------------
Sjors Verschuren
May 2026
"""

from __future__ import annotations

import re
import argparse
import importlib
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import mne
import numpy as np
from tqdm import tqdm


LOG_FILE_PATH: Optional[Path] = None
FAILED_SUBJECTS_FILE_PATH: Optional[Path] = None


def configure_log_file(log_file_path: Optional[str | Path]) -> None:
    """Configure optional file logging for all log() messages."""
    global LOG_FILE_PATH

    if log_file_path is None:
        LOG_FILE_PATH = None
        return

    path = Path(log_file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE_PATH = path


def configure_failed_subjects_file(failed_subjects_file_path: Optional[str | Path]) -> None:
    """Configure optional file for tracking failed subjects during runtime."""
    global FAILED_SUBJECTS_FILE_PATH

    if failed_subjects_file_path is None:
        FAILED_SUBJECTS_FILE_PATH = None
        return

    path = Path(failed_subjects_file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    FAILED_SUBJECTS_FILE_PATH = path

    # Start each run with a clean file that is updated incrementally.
    with FAILED_SUBJECTS_FILE_PATH.open("w", encoding="utf-8") as f:
        f.write("")


def append_failed_subject(subject_id: str) -> None:
    """Append a failed subject ID immediately so progress is preserved on interruption."""
    if FAILED_SUBJECTS_FILE_PATH is None:
        return

    with FAILED_SUBJECTS_FILE_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{subject_id}\n")


def log(message: str = ""):
    """Write output to console and optional log file without breaking tqdm bars."""
    tqdm.write(message)

    if LOG_FILE_PATH is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with LOG_FILE_PATH.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")


def import_harmonie_reader(harmonie_reader_root: Path):
    """
    Import the compiled Snooz HarmonieReader module from outside its repo.

    Expected Windows layout after CMake build:
        harmonie_reader_root/build/Release/HarmonieReader*.pyd

    Example:
        C:/Users/sversch6/Documents/harmonie_reader
    """
    # Find the compiled .pyd
    harmonie_reader_root = Path(harmonie_reader_root).resolve()
    candidate_paths = [
        harmonie_reader_root,
        harmonie_reader_root / "build",
        harmonie_reader_root / "build" / "Release",
        harmonie_reader_root / "build" / "Debug",
    ]

    for path in candidate_paths:
        if path.exists():
            sys.path.insert(0, str(path))

    # Works if sys.path contains build/Release directly
    try:
        return importlib.import_module("HarmonieReader")
    except ImportError:
        pass

    # Backup
    try:
        return importlib.import_module("build.Release.HarmonieReader")
    except ImportError as exc:
        raise ImportError(
            "Could not import HarmonieReader. Check that the compiled .pyd exists "
            "and that harmonie_reader_root points to the harmonie_reader repo folder.\n"
            f"Given root: {harmonie_reader_root}"
        ) from exc


def scale_factor_to_volts(unit: str | None) -> float:
    """
    MNE RawArray expects data in volts.

    Harmonie may report units as uV, µV, mV, V, etc.
    Unknown/empty units are treated as 1e-6.
    """
    if unit is None or type(unit) is not str:
        return 1e-6

    u = unit.strip().lower()

    if u in {"uv", "µv", "μv", "microv", "microvolt", "microvolts"}:
        return 1e-6
    if u in {"mv", "milliv", "millivolt", "millivolts"}:
        return 1e-3
    if u in {"v", "volt", "volts"}:
        return 1.0

    return 1e-6


def get_main_sample_rate(channels) -> float:
    """
    Return the most common sample rate among channels.
    Useful because MNE RawArray needs one sampling frequency.
    """
    rates = [float(ch.sample_rate) for ch in channels]
    counts = Counter(rates)
    return counts.most_common(1)[0][0]


def infer_channel_types(channel_names: list[str]) -> list[str]:
    """
    Infer MNE channel types from channel names.
    
    Maps channel names to their physiological types:
    - EEG channels (Fp, Cz, Pz, etc.)
    - EMG channels (contain 'EMG')
    - ECG channels (contain 'ECG')
    - EOG channels (contain 'EOG')
    - Other channels default to 'misc'
    """
    channel_types = []
    
    for name in channel_names:
        name_upper = name.upper()
        
        if "EMG" in name_upper:
            channel_types.append("emg")
        elif "ECG" in name_upper:
            channel_types.append("ecg")
        elif "EOG" in name_upper:
            channel_types.append("eog")
        elif any(eeg_pattern in name_upper for eeg_pattern in 
                 ["FP", "F", "C", "P", "T", "O", "A", "M", "Z"]):
            # Common EEG electrode names
            channel_types.append("eeg")
        else:
            channel_types.append("misc")
    
    return channel_types


def section_model_to_array(signal_model, selected_names: list[str], section_idx: int) -> np.ndarray:
    """
    Return section samples ordered exactly like selected_names.

    HarmonieReader may return channel objects without a populated name field, even when
    channel names were explicitly requested. When all missing names can be matched 1-to-1
    with the unnamed channels by position, a positional fallback is applied and a warning
    is logged. This preserves the requested label order at the cost of trusting the reader's
    return order for those channels.
    """
    samples_by_name: dict[str, np.ndarray] = {}
    unnamed_samples: list[np.ndarray] = []
    duplicate_names: list[str] = []

    for channel in signal_model:
        channel_name = str(getattr(channel, "channel", "")).strip()
        channel_samples = np.asarray(channel.samples, dtype=np.float64)

        if not channel_name:
            unnamed_samples.append(channel_samples)
            continue

        if channel_name in samples_by_name:
            duplicate_names.append(channel_name)
            unnamed_samples.append(channel_samples)
            continue

        samples_by_name[channel_name] = channel_samples

    missing_names = [name for name in selected_names if name not in samples_by_name]
    if missing_names and len(unnamed_samples) == len(missing_names):
        if section_idx == 0:
            log(
                "WARNING: section channels include unnamed/duplicate labels; "
                "using requested channel order fallback for missing labels. "
                f"Missing={len(missing_names)}, unnamed_or_duplicate={len(unnamed_samples)}"
            )

        for name, samples in zip(missing_names, unnamed_samples):
            samples_by_name[name] = samples

    elif missing_names:
        present_names = sorted(samples_by_name.keys())
        raise RuntimeError(
            f"Section {section_idx} could not align channels. "
            f"Missing requested channels: {missing_names}. "
            f"Unnamed or duplicate channels: {len(unnamed_samples)}. "
            f"Duplicate names: {sorted(set(duplicate_names))}. "
            f"Present named channels: {present_names}"
        )

    ordered_samples = [samples_by_name[name] for name in selected_names]
    section_samples = np.asarray(ordered_samples, dtype=np.float64)

    if section_samples.ndim != 2:
        raise RuntimeError(
            f"Unexpected section shape at section {section_idx}: {section_samples.shape}"
        )

    if section_samples.shape[0] != len(selected_names):
        raise RuntimeError(
            f"Expected {len(selected_names)} channels, got "
            f"{section_samples.shape[0]} in section {section_idx}"
        )

    return section_samples


def convert_sts_to_edf(
    sts_path: str | Path,
    edf_path: str | Path,
    harmonie_reader_root: str | Path,
    tmp_dir: str | Path,
    montage_index: int = 0,
    channel_names: Optional[Iterable[str]] = None,
    keep_only_main_sample_rate: bool = True,
    overwrite: bool = True,
    use_memmap: bool = True,
    copy_source_file_to_tmp: bool = False,
    check_edf_validity: bool = True,
) -> Path:
    """
    Convert one Harmonie .STS/.SIG recording to EDF using Snooz HarmonieReader and MNE.

    Parameters
    ----------
    sts_path
        Path to the .STS file. The matching .SIG file should be next to it.
    edf_path
        Output .edf path.
    harmonie_reader_root
        Path to the harmonie_reader repository/build root.
    memmap_dir
        Directory used for the temporary disk-backed memmap file.
    montage_index
        Montage index to read. Your test code uses 0.
    channel_names
        Optional explicit channel list. If None, all channels are considered.
    keep_only_main_sample_rate
        If True, keep only channels with the most common sample rate.
        This avoids resampling/filtering. MNE RawArray requires a single sfreq.
        If False, raise an error when mixed sample rates are present.
    overwrite
        Whether to overwrite an existing EDF.
    copy_source_file_to_tmp
        Whether to copy the source .STS/.SIG files to the tmp dir before reading.
        Useful when source files are on a slow (network) drive and tmp_dir is on a fast local drive.
    use_memmap
        If True, use disk-backed memmap for samples (default, uses less RAM).
        If False, keep all samples in RAM (faster but uses more memory).
    check_edf_validity
        If True, try reading back the written EDF with pyEDFlib as a validity check.
        If False, skip pyEDFlib import and readback validation.

    Returns
    -------
    Path to written EDF file.
    """
    sts_path = Path(sts_path).resolve()
    edf_path = Path(edf_path).resolve()

    if edf_path.exists() and not overwrite:
        log(f"EDF already exists and overwrite=False, skipping: {edf_path}\n")
        return edf_path

    if sts_path.suffix.lower() != ".sts":
        raise ValueError(f"Expected an .STS file, got: {sts_path}")

    sig_path = sts_path.with_suffix(".SIG")
    sig_path_lower = sts_path.with_suffix(".sig")

    if not sig_path.exists() and not sig_path_lower.exists():
        raise FileNotFoundError(
            f"Could not find matching .SIG/.sig file next to:\n{sts_path}"
        )
    
    # Optionally copy source files to tmp directory for faster reading if the source is on a slow drive.
    if copy_source_file_to_tmp:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for source_path in [sts_path, sig_path, sig_path_lower]:
            if source_path.exists():
                dest_path = tmp_dir / source_path.name
                if not dest_path.exists():
                    dest_path.write_bytes(source_path.read_bytes())
                    log(f"Copied {source_path} to {dest_path} for faster reading.")
                else:
                    log(f"Tmp copy skipped, already exists: {dest_path}")

    # Setup reader
    HarmonieReader = import_harmonie_reader(Path(harmonie_reader_root))
    reader = HarmonieReader.HarmonieReader()
    memmap_path: Optional[Path] = None
    tmp_dir = Path(tmp_dir).resolve()

    # Use the local tmp copy if available, otherwise fall back to the original path.
    sts_open_path = tmp_dir / sts_path.name if (copy_source_file_to_tmp and (tmp_dir / sts_path.name).exists()) else sts_path

    log(f"Opening: {sts_open_path}")
    ok = reader.open_file(str(sts_open_path))

    if not ok:
        last_error = ""
        try:
            last_error = reader.get_last_error()
        except Exception:
            pass
        raise RuntimeError(f"Failed to open {sts_path}\n{last_error}")

    try:
        # Get sections and channels.
        section_count = reader.get_signal_section_count()
        channels = list(reader.get_channels(montage_index))

        if not channels:
            raise RuntimeError(f"No channels found in montage {montage_index}: {sts_path}")

        # Set wanted channels if appropriate, otherwise keep all.
        if channel_names is None:
            selected_channels = channels
        else:
            wanted = set(channel_names)
            selected_channels = [ch for ch in channels if ch.name in wanted]

        if not selected_channels:
            raise RuntimeError("No selected channels found.")

        # Handle sample-rate compatibility with MNE RawArray.
        all_rates = sorted({float(ch.sample_rate) for ch in selected_channels})

        if len(all_rates) > 1:
            if keep_only_main_sample_rate:
                main_sfreq = get_main_sample_rate(selected_channels)
                before = len(selected_channels)
                selected_channels = [
                    ch for ch in selected_channels
                    if float(ch.sample_rate) == main_sfreq
                ]
                after = len(selected_channels)
                log(
                    f"Mixed sample rates found {all_rates}. "
                    f"Keeping only main sample rate {main_sfreq} Hz: "
                    f"{after}/{before} channels."
                )
            else:
                raise RuntimeError(
                    "Selected channels have mixed sample rates, but "
                    "keep_only_main_sample_rate=False. Rates: "
                    f"{all_rates}"
                )

        # Setup channel-wise scale factors based on units, and log any issues.
        sfreq = float(selected_channels[0].sample_rate)
        selected_names = [ch.name for ch in selected_channels]
        units_by_name = {ch.name: ch.unit for ch in selected_channels}
        missing_unit_count = 0
        unknown_unit_counts: Counter[str] = Counter()
        scale_factors_list: list[float] = []

        for ch_name in selected_names:
            unit = units_by_name.get(ch_name)

            if unit is None or type(unit) is not str:
                missing_unit_count += 1
            else:
                unit_normalized = unit.strip().lower()
                known_units = {
                    "uv", "µv", "μv", "microv", "microvolt", "microvolts",
                    "mv", "milliv", "millivolt", "millivolts",
                    "v", "volt", "volts",
                }
                if unit_normalized not in known_units:
                    unknown_unit_counts[str(unit)] += 1

            scale_factors_list.append(scale_factor_to_volts(unit))

        scale_factors = np.asarray(scale_factors_list, dtype=np.float64)

        if missing_unit_count or unknown_unit_counts:
            warning_parts = []
            if missing_unit_count:
                warning_parts.append(f"missing/invalid unit: {missing_unit_count} channels")
            if unknown_unit_counts:
                unknown_summary = ", ".join(
                    f"{unit!r}: {count}" for unit, count in sorted(unknown_unit_counts.items())
                )
                warning_parts.append(f"unknown units ({unknown_summary})")

            log(
                "WARNING: unit fallback to uV used for "
                f"{sts_path.name}. "
                + "; ".join(warning_parts)
            )

        log(
            f"Reading {len(selected_names)} channels, "
            f"{section_count} signal sections, sfreq={sfreq} Hz"
        )

        # First pass: read each section to determine total sample count and check for issues.
        # We need to do this to allocate a memmap array because we run out of RAM otherwise, but it's slow.
        # Couldn't find a way to get the lengths without reading the samples, unfortunately.
        if use_memmap:
            section_lengths: list[int] = []
            for section_idx in range(section_count):
                signal_model = reader.get_signal_section(
                    montage_index,
                    selected_names,
                    section_idx,
                )

                section_samples = section_model_to_array(signal_model, selected_names, section_idx)

                section_lengths.append(int(section_samples.shape[1]))

                log(
                    f"  Scanned section {section_idx + 1}/{section_count}: "
                    f"{section_samples.shape[1]} samples"
                )

            total_samples = int(sum(section_lengths))
            if total_samples == 0:
                raise RuntimeError("No samples found across sections.")
        else:
            total_samples = None  # Not needed when not using memmap

        # Create memmap array for samples and do a second pass to fill it.
        edf_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        if use_memmap:
            with tempfile.NamedTemporaryFile(
                prefix="sig2edf_",
                suffix=".dat",
                dir=str(tmp_dir),
                delete=False,
            ) as tmp_file:
                memmap_path = Path(tmp_file.name)

            samples = np.memmap(
                memmap_path,
                mode="w+",
                dtype=np.float64,
                shape=(len(selected_names), total_samples),
            )
        else:
            memmap_path = None
            # No pre-allocation; sections are collected and concatenated after the loop.
            sections_buffer: list[np.ndarray] = []

        write_start = 0
        for section_idx in range(section_count):
            signal_model = reader.get_signal_section(
                montage_index,
                selected_names,
                section_idx,
            )

            section_samples = section_model_to_array(signal_model, selected_names, section_idx)
            section_samples *= scale_factors[:, np.newaxis]  # broadcast (n_channels,) over time axis

            if use_memmap:
                expected_len = section_lengths[section_idx]
                if int(section_samples.shape[1]) != expected_len:
                    raise RuntimeError(
                        f"Section length changed between scan and read at section {section_idx}: "
                        f"{section_samples.shape[1]} != {expected_len}"
                    )
                write_end = write_start + expected_len
                samples[:, write_start:write_end] = section_samples
                write_start = write_end
            else:
                sections_buffer.append(section_samples)

            log(
                f"  {'Wrote' if use_memmap else 'Read'} section {section_idx + 1}/{section_count}: "
                f"{section_samples.shape[1]} samples"
            )

        if use_memmap:
            samples.flush()
        else:
            samples = np.concatenate(sections_buffer, axis=1)

        recording_start_dt: Optional[datetime] = None
        try:
            recording_start_time = float(reader.get_recording_start_time())
            if np.isfinite(recording_start_time) and recording_start_time > 0:
                recording_start_dt = datetime.fromtimestamp(
                    recording_start_time,
                    tz=timezone.utc,
                )
            else:
                log(
                    "WARNING: invalid recording start timestamp from reader: "
                    f"{recording_start_time}"
                )
        except Exception as exc:
            log(f"WARNING: could not read recording start time: {exc}")

        # Create MNE RawArray.
        info = mne.create_info(
            ch_names=selected_names,
            sfreq=sfreq,
            ch_types=infer_channel_types(selected_names),
        )
        # Put datetime in the metadata if available. 
        # We omit patient data on purpose, but you can add it here if you want.
        raw = mne.io.RawArray(samples, info, verbose="ERROR")
        if recording_start_dt is not None:
            recording_start_dt = recording_start_dt.replace(microsecond=0, tzinfo=timezone.utc)
            raw.set_meas_date(recording_start_dt)
            log(
                "Set EDF recording start (UTC) from Harmonie metadata: "
                f"{recording_start_dt.isoformat()}"
            )

        # Transfer events as EDF annotations if available.
        try:
            events = reader.get_events()
            if events:
                onsets = []
                durations = []
                descriptions = []

                for ev in events:
                    try:
                        onset = float(ev.start_time)
                    except Exception:
                        continue

                    try:
                        duration = float(ev.duration)
                    except Exception:
                        duration = 0.0

                    name = str(getattr(ev, "name", "event"))
                    group = str(getattr(ev, "group", ""))

                    desc = f"{group}/{name}" if group else name

                    onsets.append(onset)
                    durations.append(duration)
                    descriptions.append(desc)

                if onsets:
                    raw.set_annotations(
                        mne.Annotations(
                            onset=onsets,
                            duration=durations,
                            description=descriptions,
                        )
                    )
                    log(f"Added {len(onsets)} annotations.")
        except Exception as exc:
            log(f"WARNING: could not transfer annotations: {exc}")

        # Strict EDF-safe check block
        # - Bad channels (warn for diagnostics if conversion fails, don't silently remove)
        # - Label sanitization and deduplication
        # ----------------------------------------------------------
        # As a safety, check for bad channels. 
        bad = []
        data = raw.get_data()
        for ch_name, x in zip(raw.ch_names, data):
            if (
                np.isnan(x).any()
                or np.isinf(x).any()
                or np.nanmin(x) == np.nanmax(x)
            ):
                bad.append(ch_name)
        if bad:
            log(f"WARNING: bad channels detected: {bad}")
            log("These may cause issues with EDF export, but we'll attempt anyways.")

        # Sanitize labels
        def clean_label(label, i):
            label = str(label).encode("ascii", errors="ignore").decode("ascii")
            label = re.sub(r"[^A-Za-z0-9_+\-\. ]", "_", label).strip()
            return label[:16] or f"CH{i:03d}"

        proposed = [clean_label(ch, i) for i, ch in enumerate(raw.ch_names, 1)]

        seen = {}
        mapping = {}
        for old, new in zip(raw.ch_names, proposed):
            base = new[:12]
            seen[base] = seen.get(base, 0) + 1
            if seen[base] > 1:
                new = f"{base}_{seen[base]:03d}"[:16]
            mapping[old] = new

        raw.rename_channels(mapping)

        # ----------------------------------------------------------

        # Export with MNE edfio backend
        log(f"Writing EDF: {edf_path}")
        raw.export(
            str(edf_path),
            fmt="edf",
            physical_range="auto",
            overwrite=overwrite,
            verbose="ERROR",
        )

        # Immediate readback check (optional, requires pyEDFlib).
        # This can catch issues with the written EDF file right away.
        # Exceptions are caught later so failed subjects are logged.
        if check_edf_validity:
            try:
                import pyedflib
            except ImportError:
                log("pyEDFlib not available, skipping readback check.")
            else:
                with pyedflib.EdfReader(str(edf_path)) as f:
                    log(f"pyEDFlib opened file successfully with {f.signals_in_file} signals.")
        else:
            log("Skipping EDF readback check (--no-check).")

    finally:
        try:
            reader.close_file()
        except Exception:
            pass
        if use_memmap and memmap_path is not None:
            try:
                if memmap_path.exists():
                    samples._mmap.close()  # type: ignore
                    del samples
                    memmap_path.unlink()
            except Exception as exc:
                log(f"WARNING: could not delete memmap file: {memmap_path}\n{exc}")
        if copy_source_file_to_tmp:
            for source_path in [sts_path, sig_path, sig_path_lower]:
                if source_path.exists():
                    tmp_copy = tmp_dir / source_path.name
                    if tmp_copy.exists():
                        try:
                            tmp_copy.unlink()
                            log(f"Deleted tmp copy: {tmp_copy}")
                        except Exception as exc:
                            log(f"WARNING: could not delete tmp copy: {tmp_copy}\n{exc}")

    log(f"Done: {edf_path}\n")
    return edf_path


def batch_convert_tree(
    input_root: str | Path,
    output_root: str | Path,
    harmonie_reader_root: str | Path,
    tmp_dir: str | Path,
    montage_index: int = 0,
    keep_only_main_sample_rate: bool = True,
    overwrite: bool = True,
    use_memmap: bool = True,
    copy_source_file_to_tmp: bool = False,
    check_edf_validity: bool = True,
) -> list[Path]:
    """
    Batch convert all .STS files under input_root to EDF.

    The output folder mirrors the input layout. For example:

        input_root/RESP0400/file.STS
        -> output_root/RESP0400/file.edf

    Filenames are normalized to:
        subjectID_date_time.edf

    based on the original .STS filename and parent folder.
    """
    # Set paths
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    tmp_dir = Path(tmp_dir).resolve()
    # Find all .STS files (case-insensitive)
    sts_files = sorted(
        list(input_root.rglob("*.STS")) +
        list(input_root.rglob("*.sts"))
    )
    # De-duplicate in case-insensitive filesystems
    sts_files = sorted(set(sts_files))

    if not sts_files:
        raise FileNotFoundError(f"No .STS files found under: {input_root}")

    written = []
    failed = []

    # Group files by subject ID, inferred as the first folder level under input_root.
    subject_to_files: dict[str, list[Path]] = {}
    for sts_path in sts_files:
        relative_parent = sts_path.parent.relative_to(input_root)
        subject_id = relative_parent.parts[0] if relative_parent.parts else sts_path.parent.name
        subject_to_files.setdefault(subject_id, []).append(sts_path)

    log(f"Found {len(sts_files)} .STS files across {len(subject_to_files)} subjects.")

    with tqdm(
        sorted(subject_to_files.keys()),
        total=len(subject_to_files),
        desc="Subjects",
        unit="subject",
    ) as subject_pbar:
        for subject_id in subject_pbar:
            subject_files = sorted(subject_to_files[subject_id])
            subject_pbar.set_postfix_str(f"{subject_id} ({len(subject_files)} files)")

            log("\n" + "=" * 80)
            log(f"Subject: {subject_id} ({len(subject_files)} files)")

            for sts_path in subject_files:
                try:
                    relative_parent = sts_path.parent.relative_to(input_root)
                    edf_name = sts_path.stem + ".edf"
                    edf_path = output_root / relative_parent / edf_name

                    log("-" * 80)
                    log(f"Input : {sts_path}")
                    log(f"Output: {edf_path}\n")

                    out = convert_sts_to_edf(
                        sts_path=sts_path,
                        edf_path=edf_path,
                        harmonie_reader_root=harmonie_reader_root,
                        tmp_dir=tmp_dir,
                        montage_index=montage_index,
                        keep_only_main_sample_rate=keep_only_main_sample_rate,
                        overwrite=overwrite,
                        use_memmap=use_memmap,
                        copy_source_file_to_tmp=copy_source_file_to_tmp,
                        check_edf_validity=check_edf_validity,
                    )

                    written.append(out)

                except Exception as exc:
                    log(f"FAILED: {sts_path}")
                    log(str(exc))
                    append_failed_subject(subject_id)
                    failed.append((sts_path, exc))

    log("\n" + "=" * 80)
    log(f"Finished. Written: {len(written)} EDF files. Failed: {len(failed)} files.")

    if failed:
        log("\nFailures:")
        for path, exc in failed:
            log(f"  {path}: {exc}")

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Convert Harmonie .STS/.SIG files to EDF using Snooz HarmonieReader and MNE."
    )

    parser.add_argument(
        "--input-root",
        required=True,
        help="Root folder containing patient subfolders with .STS/.SIG files.",
    )

    parser.add_argument(
        "--output-root",
        required=True,
        help="Root folder where EDF output tree will be written.",
    )

    parser.add_argument(
        "--harmonie-reader-root",
        default="..\\ext\\harmonie_reader",
        help="Path to the cloned/built harmonie_reader repository.",
    )

    parser.add_argument(
        "--tmp-dir",
        default="..\\data\\tmp\\sig2edf_tmp",
        help=(
            "Directory for temporary files used during conversion. "
            "Use a fast drive or local SSD to reduce I/O bottlenecks."
        ),
    )

    parser.add_argument(
        "--montage-index",
        type=int,
        default=0,
        help="Montage index to read. Default: 0. (This is the index corresponding to the ground-referenced montage in our files)",
    )

    parser.add_argument(
        "--all-sample-rates-error",
        action="store_true",
        help=(
            "Raise an error if channels have mixed sample rates. "
            "By default, only the most common sample-rate group is exported."
            "This is because MNE RawArray requires a single Fs, and sometimes the annotations are at a different rate. "
            "Set this flag to enforce that all channels must have the same sample rate."
        ),
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing EDF files.",
    )

    parser.add_argument(
        "--copy-source-file-to-tmp",
        action="store_true",
        help=(
            "Copy source .STS/.SIG files to the tmp directory before reading. "
            "This can speed up reading if the source files are on a slow drive and the tmp directory is on a fast drive."
        ),
    )

    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Optional path to a text log file. "
            "If omitted, defaults to <output-root>/sig2edf_logs_YYYYMMDD_HHMMSS.txt"
        ),
    )

    parser.add_argument(
        "--no-memmap",
        action="store_true",
        help="Keep all samples in RAM instead of using disk-backed memmap. Faster but uses more memory.",
    )

    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip post-export EDF readback validation with pyEDFlib.",
    )

    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    log_file = (
        Path(args.log_file).resolve()
        if args.log_file
        else output_root / f"sig2edf_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    configure_log_file(log_file)
    failed_subjects_file = output_root / "failed_subjects.txt"
    configure_failed_subjects_file(failed_subjects_file)
    log("\n============== Starting batch conversion of Harmonie .STS/.SIG to EDF ==============\n")
    log(f"Logging to: {log_file}")
    log(f"Failed subjects file: {failed_subjects_file}")

    batch_convert_tree(
        input_root=args.input_root,
        output_root=output_root,
        harmonie_reader_root=args.harmonie_reader_root,
        tmp_dir=args.tmp_dir,
        montage_index=args.montage_index,
        keep_only_main_sample_rate=not args.all_sample_rates_error,
        overwrite=not args.no_overwrite,
        use_memmap=not args.no_memmap,
        copy_source_file_to_tmp=args.copy_source_file_to_tmp,
        check_edf_validity=not args.no_check,
    )


if __name__ == "__main__":
    main()