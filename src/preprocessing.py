"""Preprocessing utilities for ECG heart disease detection.

This module implements the full signal preprocessing pipeline used for
both the PTB-XL and Chapman-Shaoxing ECG datasets. It includes helpers
for loading raw records, filtering, normalization, length normalization,
augmentation, and mapping dataset-specific labels into the unified
multi-label space defined in :mod:`src.config`.

Only step 3 of the overall project specification is implemented here.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import numpy as np
import pandas as pd
import wfdb
from scipy import io as sio
from scipy.signal import butter, filtfilt, resample

from .config import CONFIG, UNIFIED_LABELS


logger = logging.getLogger(__name__)


ArrayLike = Union[np.ndarray, Sequence[float]]
PathLike = Union[str, Path]


def _to_path(path: PathLike) -> Path:
    """Convert an arbitrary path-like object to :class:`pathlib.Path`.

    Parameters
    ----------
    path:
        Input path as string or :class:`~pathlib.Path`.

    Returns
    -------
    pathlib.Path
        Normalized absolute path.
    """

    p = Path(path)
    return p.expanduser().resolve()


def _ensure_2d(signal: np.ndarray) -> np.ndarray:
    """Ensure that a signal array is 2D (time, leads).

    If a 1D array is provided, it is interpreted as a single lead and
    reshaped to ``(T, 1)``.

    Parameters
    ----------
    signal:
        Input array of shape ``(T,)`` or ``(T, L)``.

    Returns
    -------
    numpy.ndarray
        2D array of shape ``(T, L)``.
    """

    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D signal, got shape {arr.shape}.")
    return arr


def pad_or_truncate(signal: ArrayLike, target_length: int = CONFIG.SIGNAL_LENGTH) -> np.ndarray:
    """Ensure all signals are exactly ``target_length`` samples.

    For shorter signals, pad with zeros at the end. For longer signals,
    perform a centered crop to the desired length.

    Parameters
    ----------
    signal:
        Input signal array of shape ``(T, L)`` or ``(T,)``.
    target_length:
        Desired number of time samples.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(target_length, L)``.
    """

    arr = _ensure_2d(np.asarray(signal, dtype=np.float32))
    num_samples, num_leads = arr.shape

    if num_samples == target_length:
        return arr

    if num_samples > target_length:
        start = (num_samples - target_length) // 2
        end = start + target_length
        return arr[start:end, :]

    # num_samples < target_length: pad with zeros at the end
    pad_width = target_length - num_samples
    padded = np.zeros((target_length, num_leads), dtype=np.float32)
    padded[:num_samples, :] = arr
    return padded


def bandpass_filter(signal: ArrayLike, lowcut: float = 0.5, highcut: float = 40.0, fs: float = CONFIG.SAMPLING_RATE) -> np.ndarray:
    """Apply a 4th-order Butterworth bandpass filter.

    The filter removes baseline wander and high-frequency noise. Filtering
    is applied independently along the time dimension for each lead.

    Parameters
    ----------
    signal:
        Input signal array of shape ``(T, L)`` or ``(T,)``.
    lowcut:
        Low cutoff frequency in Hz.
    highcut:
        High cutoff frequency in Hz.
    fs:
        Sampling frequency in Hz.

    Returns
    -------
    numpy.ndarray
        Filtered signal of the same shape as the input, with dtype
        ``float32``.
    """

    arr = _ensure_2d(np.asarray(signal, dtype=np.float32))

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if not (0.0 < low < high < 1.0):
        raise ValueError(
            f"Invalid bandpass frequencies: lowcut={lowcut}, highcut={highcut}, fs={fs}."
        )

    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, arr, axis=0).astype(np.float32)
    return filtered


def normalize_signal(signal: ArrayLike) -> np.ndarray:
    """Per-lead z-score normalization.

    For each lead, subtract the mean and divide by the standard deviation.
    If a lead has zero variance, its standard deviation is set to 1 to
    avoid division by zero.

    Parameters
    ----------
    signal:
        Input signal array of shape ``(T, L)`` or ``(T,)``.

    Returns
    -------
    numpy.ndarray
        Normalized signal of the same shape as the input, with dtype
        ``float32``.
    """

    arr = _ensure_2d(np.asarray(signal, dtype=np.float32))
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    std_safe = np.where(std == 0.0, 1.0, std)
    normalized = (arr - mean) / std_safe
    return normalized.astype(np.float32)


def augment_signal(signal: ArrayLike) -> np.ndarray:
    """Apply data augmentation to an ECG signal.

    Augmentations include:

    - Random Gaussian noise (standard deviation = 0.01)
    - Random amplitude scaling (uniform in [0.9, 1.1])
    - Random time shift (±50 samples, padded with zeros)
    - Random lead dropout: with 20% probability, zero out 1–2 random leads

    Parameters
    ----------
    signal:
        Input signal array of shape ``(T, L)`` or ``(T,)``.

    Returns
    -------
    numpy.ndarray
        Augmented signal of the same shape as the input, with dtype
        ``float32``.
    """

    rng = np.random.default_rng()
    arr = _ensure_2d(np.asarray(signal, dtype=np.float32))
    augmented = arr.copy()

    # Amplitude scaling
    scale = rng.uniform(0.9, 1.1)
    augmented *= scale

    # Additive Gaussian noise
    noise = rng.normal(loc=0.0, scale=0.01, size=augmented.shape).astype(np.float32)
    augmented += noise

    # Random time shift
    max_shift = 50
    shift = rng.integers(-max_shift, max_shift + 1)
    if shift != 0:
        shifted = np.zeros_like(augmented)
        if shift > 0:
            shifted[shift:, :] = augmented[:-shift, :]
        else:
            shifted[:shift, :] = augmented[-shift:, :]
        augmented = shifted

    # Random lead dropout
    if rng.random() < 0.2:
        num_leads = augmented.shape[1]
        num_drop = int(rng.integers(1, min(2, num_leads) + 1))
        drop_indices = rng.choice(num_leads, size=num_drop, replace=False)
        augmented[:, drop_indices] = 0.0

    return augmented.astype(np.float32)


def load_ptbxl_record(record_path: PathLike, sampling_rate: int = CONFIG.SAMPLING_RATE) -> np.ndarray:
    """Load a PTB-XL record using :func:`wfdb.rdrecord`.

    The function reads the underlying ``.hea``/``.dat`` files and returns
    the signal as a ``(5000, 12)`` array. If the original sampling rate
    differs from ``sampling_rate``, the signal is resampled accordingly.

    Parameters
    ----------
    record_path:
        Path to the PTB-XL record. This can be either the base record
        name, or a path to a ``.hea``/``.dat`` file.
    sampling_rate:
        Desired sampling rate in Hz (default: 500).

    Returns
    -------
    numpy.ndarray
        Signal array of shape ``(5000, 12)`` with dtype ``float32``.

    Raises
    ------
    IOError
        If the record cannot be read from disk.
    """

    path = _to_path(record_path)
    # wfdb.rdrecord expects the record name without extension.
    if path.suffix in {".hea", ".dat"}:
        base = path.with_suffix("")
    else:
        base = path

    try:
        record = wfdb.rdrecord(str(base))
    except Exception as exc:  # pragma: no cover - IO-dependent
        logger.error("Failed to load PTB-XL record from %s: %s", base, exc)
        raise IOError(f"Could not read PTB-XL record at {base}") from exc

    signal = np.asarray(record.p_signal, dtype=np.float32)

    # Resample if necessary
    original_fs = float(getattr(record, "fs", sampling_rate))
    if original_fs <= 0:
        original_fs = float(sampling_rate)

    if abs(original_fs - float(sampling_rate)) > 1e-3:
        num_samples = int(round(signal.shape[0] * float(sampling_rate) / original_fs))
        signal = resample(signal, num_samples, axis=0).astype(np.float32)

    # Ensure (5000, 12)
    signal = pad_or_truncate(signal, target_length=CONFIG.SIGNAL_LENGTH)

    # Guarantee exactly 12 leads by truncating or padding columns if needed
    signal = _ensure_2d(signal)
    if signal.shape[1] > CONFIG.NUM_LEADS:
        signal = signal[:, : CONFIG.NUM_LEADS]
    elif signal.shape[1] < CONFIG.NUM_LEADS:
        pad_cols = CONFIG.NUM_LEADS - signal.shape[1]
        padded = np.zeros((signal.shape[0], CONFIG.NUM_LEADS), dtype=np.float32)
        padded[:, : signal.shape[1]] = signal
        signal = padded

    return signal.astype(np.float32)


def load_chapman_record(record_path: PathLike) -> np.ndarray:
    """Load a Chapman-Shaoxing record from ``.hea``/``.mat`` files.

    The underlying waveform is stored in a MATLAB ``.mat`` file, usually
    under the key ``"val"`` with shape either ``(12, 5000)`` or
    ``(5000, 12)``. This function normalizes all variants to
    ``(5000, 12)``.

    Parameters
    ----------
    record_path:
        Path to a Chapman record header (``.hea``) or MATLAB file
        (``.mat``).

    Returns
    -------
    numpy.ndarray
        Signal array of shape ``(5000, 12)`` with dtype ``float32``.

    Raises
    ------
    IOError
        If the waveform file cannot be read or parsed.
    """

    path = _to_path(record_path)
    mat_path = path.with_suffix(".mat") if path.suffix == ".hea" else path

    try:
        mat_dict = sio.loadmat(str(mat_path))
    except Exception as exc:  # pragma: no cover - IO-dependent
        logger.error("Failed to load Chapman .mat file from %s: %s", mat_path, exc)
        raise IOError(f"Could not read Chapman MAT file at {mat_path}") from exc

    # Extract waveform array; commonly stored under key 'val'
    signal_array: Any
    if "val" in mat_dict:
        signal_array = mat_dict["val"]
    else:
        # Fall back to the first non-private array-like entry
        candidates = [
            v
            for k, v in mat_dict.items()
            if not k.startswith("__") and isinstance(v, np.ndarray)
        ]
        if not candidates:
            raise IOError(f"No waveform array found in MAT file {mat_path}")
        signal_array = candidates[0]

    signal = np.asarray(signal_array, dtype=np.float32)

    # Normalize shape to (T, L) then to (5000, 12)
    if signal.ndim == 1:
        signal = signal[:, None]
    elif signal.ndim == 2:
        # Possible shapes: (12, 5000) or (5000, 12)
        if signal.shape[0] == CONFIG.NUM_LEADS and signal.shape[1] != CONFIG.SIGNAL_LENGTH:
            signal = signal.T
        elif signal.shape[1] == CONFIG.NUM_LEADS:
            # Already (T, 12)
            pass
        elif signal.shape[0] == CONFIG.SIGNAL_LENGTH and signal.shape[1] != CONFIG.NUM_LEADS:
            signal = signal.T
    else:
        # Flatten higher dimensions into (T, L)
        signal = signal.reshape(signal.shape[0], -1)

    signal = _ensure_2d(signal)
    signal = pad_or_truncate(signal, target_length=CONFIG.SIGNAL_LENGTH)

    # Ensure exactly 12 leads
    if signal.shape[1] > CONFIG.NUM_LEADS:
        signal = signal[:, : CONFIG.NUM_LEADS]
    elif signal.shape[1] < CONFIG.NUM_LEADS:
        pad_cols = CONFIG.NUM_LEADS - signal.shape[1]
        padded = np.zeros((signal.shape[0], CONFIG.NUM_LEADS), dtype=np.float32)
        padded[:, : signal.shape[1]] = signal
        signal = padded

    return signal.astype(np.float32)


def _init_label_vector() -> np.ndarray:
    """Return a zero-initialized binary label vector.

    Returns
    -------
    numpy.ndarray
        Binary vector of shape ``(len(UNIFIED_LABELS),)``.
    """

    return np.zeros(len(UNIFIED_LABELS), dtype=np.float32)


def _set_label(vec: np.ndarray, label: str) -> None:
    """Safely set a label entry in a binary vector to 1.

    Parameters
    ----------
    vec:
        Binary label vector.
    label:
        Label name from :data:`UNIFIED_LABELS`.
    """

    try:
        idx = UNIFIED_LABELS.index(label)
    except ValueError:
        return
    vec[idx] = 1.0


def _map_text_to_labels(text: str, vec: np.ndarray) -> None:
    """Update a label vector based on diagnostic text.

    The mapping is heuristic and relies on keyword matching in the
    provided text. It is designed to work with both PTB-XL SCP
    statements and Chapman header comments.

    Parameters
    ----------
    text:
        Diagnostic or rhythm description text.
    vec:
        Binary label vector to update in-place.
    """

    t = text.lower()

    # Normal rhythm
    if "normal" in t and "sinus" in t:
        _set_label(vec, "NORM")

    # Atrial fibrillation
    if "atrial fibrillation" in t or "afib" in t or "a-fib" in t:
        _set_label(vec, "AFIB")

    # Myocardial infarction / STEMI / ischemia
    if "stemi" in t or "st elevation" in t or "st-elevation" in t:
        _set_label(vec, "STEMI")
        _set_label(vec, "ISCHEMIA")
    if "myocardial infarction" in t or "mi" in t:
        _set_label(vec, "STEMI")
        _set_label(vec, "ISCHEMIA")
    if "ischemia" in t or "ischemic" in t or "st-t" in t or "st-t change" in t:
        _set_label(vec, "ISCHEMIA")

    # Bundle branch blocks
    if "left bundle branch" in t or "lbbb" in t:
        _set_label(vec, "LBBB")
    if "right bundle branch" in t or "rbbb" in t:
        _set_label(vec, "RBBB")

    # AV block
    if "1st degree av" in t or "first degree av" in t or "1 av block" in t or "first-degree av" in t:
        _set_label(vec, "1AVB")

    # Hypertrophy / LVH
    if "left ventricular hypertrophy" in t or "lvh" in t or "hypertrophy" in t:
        _set_label(vec, "LVHYP")

    # Premature beats
    if "premature atrial" in t or "pac" in t:
        _set_label(vec, "PAC")
    if "premature ventricular" in t or "pvc" in t or "ventricular ectopic" in t:
        _set_label(vec, "PVC")

    # Rate-related
    if "bradycardia" in t or "brady-arrhythmia" in t:
        _set_label(vec, "BRADYCARDIA")
    if "tachycardia" in t or "tachy-arrhythmia" in t:
        _set_label(vec, "TACHYCARDIA")

    # Heart failure
    if "heart failure" in t or "congestive heart" in t:
        _set_label(vec, "HEART_FAILURE")

    # Conduction disorders more generally
    if "conduction" in t or "block" in t:
        # If no specific conduction label was set, mark OTHER
        if not any(
            vec[UNIFIED_LABELS.index(lbl)] == 1.0
            for lbl in ("LBBB", "RBBB", "1AVB")
        ):
            _set_label(vec, "OTHER")


def map_ptbxl_labels(
    scp_codes: Union[str, Mapping[str, Any], Sequence[str]],
    scp_statements_df: pd.DataFrame,
) -> np.ndarray:
    """Map PTB-XL SCP codes to the unified label set.

    The PTB-XL ``scp_codes`` field is typically a dictionary mapping SCP
    code strings to importance scores. This function converts that
    dictionary into a binary vector over :data:`UNIFIED_LABELS` using the
    diagnostic class and textual descriptions from ``scp_statements_df``.

    Parameters
    ----------
    scp_codes:
        SCP codes for a single record. This can be a dictionary,
        a sequence of code strings, or a string representation of a
        dictionary as stored in ``ptbxl_database.csv``.
    scp_statements_df:
        DataFrame loaded from ``scp_statements.csv`` containing at least
        the ``scp_code`` column and preferably ``diagnostic_class`` and
        textual description columns (e.g. ``statement``).

    Returns
    -------
    numpy.ndarray
        Binary label vector of length ``len(UNIFIED_LABELS)``.
    """

    # Normalize scp_codes to a list of code strings
    codes: Sequence[str]
    if isinstance(scp_codes, Mapping):
        codes = list(scp_codes.keys())
    elif isinstance(scp_codes, str):
        try:
            parsed = ast.literal_eval(scp_codes)
            if isinstance(parsed, Mapping):
                codes = list(parsed.keys())
            elif isinstance(parsed, Sequence):
                codes = [str(c) for c in parsed]
            else:
                codes = []
        except (ValueError, SyntaxError):
            codes = []
    elif isinstance(scp_codes, Sequence):
        codes = [str(c) for c in scp_codes]
    else:
        codes = []

    labels = _init_label_vector()
    if not codes:
        _set_label(labels, "OTHER")
        return labels

    # Prepare lookup table keyed by SCP code. In the official PTB-XL
    # release the code is stored in the first column (often named
    # "scp_code" or left unnamed, which pandas reads as "Unnamed: 0").
    # We therefore fall back to the first column when "scp_code" is not
    # explicitly present.
    if "scp_code" in scp_statements_df.columns:
        code_col = "scp_code"
    else:
        code_col = scp_statements_df.columns[0]

    lookup = scp_statements_df.set_index(code_col, drop=False)

    for code in codes:
        if code not in lookup.index:
            # Some codes may contain trailing spaces or different
            # capitalisation; normalise once before giving up.
            code_norm = str(code).strip()
            if code_norm not in lookup.index:
                continue
            row = lookup.loc[code_norm]
        else:
            row = lookup.loc[code]

        # Diagnostic class mapping (e.g. NORM, MI, STTC, HYP, CD)
        diag_class = str(row.get("diagnostic_class", "")).strip().lower()
        if diag_class == "norm":
            _set_label(labels, "NORM")
        elif diag_class == "mi":
            _set_label(labels, "STEMI")
            _set_label(labels, "ISCHEMIA")
        elif diag_class == "sttc":
            _set_label(labels, "ISCHEMIA")
        elif diag_class == "hyp":
            _set_label(labels, "LVHYP")
        elif diag_class == "cd":
            # generic conduction disorder; more specific parsing below
            _set_label(labels, "OTHER")

        # Use textual description fields for finer-grained mapping
        text_parts = []
        for col in ("diagnostic_subclass", "statement", "description"):
            if col in lookup.columns:
                value = row.get(col, "")
                if isinstance(value, str):
                    text_parts.append(value)

        if text_parts:
            _map_text_to_labels(" ".join(text_parts), labels)

    if not labels.any():
        _set_label(labels, "OTHER")

    return labels


def map_chapman_labels(header_comments: Union[str, Iterable[str]]) -> np.ndarray:
    """Map Chapman header comments to the unified label set.

    The Chapman-Shaoxing dataset stores rhythm and diagnosis information
    in the ``.hea`` header file. Typical lines include ``Rhythm`` and
    ``Dx`` fields. This function parses those lines and maps them to the
    unified multi-label vector.

    Parameters
    ----------
    header_comments:
        Either the entire header content as a single string or an
        iterable of header lines.

    Returns
    -------
    numpy.ndarray
        Binary label vector of length ``len(UNIFIED_LABELS)``.
    """

    if isinstance(header_comments, str):
        lines = header_comments.splitlines()
    else:
        lines = list(header_comments)

    relevant_parts = []
    for line in lines:
        stripped = line.lstrip("#").strip()
        lower = stripped.lower()
        if "rhythm" in lower or "dx" in lower or "diagnosis" in lower:
            relevant_parts.append(stripped)

    if not relevant_parts:
        # Fallback: use the entire header text
        relevant_parts = [line.lstrip("#").strip() for line in lines]

    text = " ".join(relevant_parts)
    labels = _init_label_vector()
    _map_text_to_labels(text, labels)

    if not labels.any():
        _set_label(labels, "OTHER")

    return labels
