"""Prediction utilities for ECG heart disease detection.

This module implements the single-record inference pipeline for the
ECG heart disease classifier. It loads a trained model, runs the full
signal preprocessing pipeline on a single PTB-XL or Chapman-Shaoxing
ECG file, performs test-time augmentation, and returns a rich
prediction dictionary suitable for use by the Flask frontend.

Implements step 8 of the project specification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import numpy as np
import torch

from .config import CONFIG, UNIFIED_LABELS
from .model import build_ecg_resnet
from .preprocessing import (
	augment_signal,
	bandpass_filter,
	load_chapman_record,
	load_ptbxl_record,
	normalize_signal,
	pad_or_truncate,
)


logger = logging.getLogger(__name__)

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
		Absolute, expanded path.
	"""

	return Path(path).expanduser().resolve()


def _load_ecg_signal(file_path: Path) -> np.ndarray:
	"""Load a single ECG recording from disk.

	This helper infers whether the input file belongs to PTB-XL or the
	Chapman-Shaoxing dataset based on its extension and neighbouring
	files, and then delegates loading to the appropriate function in
	:mod:`src.preprocessing`.

	The returned array has shape ``(CONFIG.SIGNAL_LENGTH, CONFIG.NUM_LEADS)``
	and dtype ``float32``.

	Parameters
	----------
	file_path:
		Path to a ``.hea``, ``.mat`` or ``.dat`` file.

	Returns
	-------
	numpy.ndarray
		Time-major ECG signal.

	Raises
	------
	FileNotFoundError
		If the ECG file does not exist.
	ValueError
		If the file extension is unsupported.
	"""

	if not file_path.exists():
		raise FileNotFoundError(f"ECG file not found: {file_path}")

	suffix = file_path.suffix.lower()
	parents_lower = {p.name.lower() for p in file_path.parents}

	try:
		if suffix == ".hea":
			# Decide between PTB-XL (.hea + .dat) and Chapman (.hea + .mat)
			mat_sibling = file_path.with_suffix(".mat")
			dat_sibling = file_path.with_suffix(".dat")

			if mat_sibling.exists() or "chapman" in parents_lower:
				# Chapman-Shaoxing
				signal = load_chapman_record(file_path)
			elif dat_sibling.exists() or "ptbxl" in parents_lower:
				# PTB-XL
				signal = load_ptbxl_record(file_path)
			else:
				# Heuristic: try Chapman first, then PTB-XL
				try:
					signal = load_chapman_record(file_path)
				except Exception:
					signal = load_ptbxl_record(file_path)
		elif suffix == ".mat":
			# Chapman-Shaoxing MAT file
			signal = load_chapman_record(file_path)
		elif suffix == ".dat":
			# PTB-XL DAT file
			signal = load_ptbxl_record(file_path)
		else:
			raise ValueError(
				f"Unsupported ECG file extension '{suffix}'. Expected .hea, .mat, or .dat."
			)
	except Exception as exc:  # pragma: no cover - IO-dependent
		logger.error("Failed to load ECG signal from %s: %s", file_path, exc)
		raise

	# All loader functions already enforce (T, L) and correct lengths/leads.
	return signal.astype(np.float32)


def _preprocess_for_model(signal: np.ndarray) -> np.ndarray:
	"""Run the full preprocessing pipeline prior to model inference.

	The pipeline consists of:

	1. Length normalization via :func:`pad_or_truncate`.
	2. Bandpass filtering with :func:`bandpass_filter`.
	3. Per-lead z-score normalization with :func:`normalize_signal`.

	The final array is returned in channel-first format
	``(CONFIG.NUM_LEADS, CONFIG.SIGNAL_LENGTH)`` suitable for feeding
	into the 1D CNN model.

	Parameters
	----------
	signal:
		Raw ECG signal of shape ``(T, L)`` or ``(T,)``.

	Returns
	-------
	numpy.ndarray
		Preprocessed signal in channel-first layout.
	"""

	arr = np.asarray(signal, dtype=np.float32)
	if arr.ndim == 1:
		arr = arr[:, None]

	arr = pad_or_truncate(arr, target_length=CONFIG.SIGNAL_LENGTH)
	arr = bandpass_filter(arr, fs=CONFIG.SAMPLING_RATE)
	arr = normalize_signal(arr)

	# Model expects (batch, 12, 5000) → here we return (12, 5000)
	return arr.astype(np.float32).T


def _load_model(model_path: Path) -> torch.nn.Module:
	"""Load the trained ECGResNet model from disk.

	Parameters
	----------
	model_path:
		Path to the saved model checkpoint (``.pth``).

	Returns
	-------
	torch.nn.Module
		ECGResNet model in evaluation mode on the configured device.

	Raises
	------
	FileNotFoundError
		If the model file does not exist.
	"""

	if not model_path.exists():
		raise FileNotFoundError(f"Model file not found: {model_path}")

	device = torch.device(CONFIG.DEVICE)
	model = build_ecg_resnet()

	try:
		state = torch.load(model_path, map_location=device)
		# Allow for checkpoints that store a dict with a 'state_dict' key
		if isinstance(state, dict) and "state_dict" in state:
			state_dict = state["state_dict"]
		else:
			state_dict = state
		model.load_state_dict(state_dict)
	except Exception as exc:  # pragma: no cover - IO-dependent
		logger.error("Failed to load model from %s: %s", model_path, exc)
		raise

	model.to(device)
	model.eval()
	return model


def _ensure_thresholds(thresholds: Union[Sequence[float], np.ndarray]) -> np.ndarray:
	"""Normalize thresholds into an array of shape ``(num_labels,)``.

	Parameters
	----------
	thresholds:
		Scalar or sequence of per-label thresholds.

	Returns
	-------
	numpy.ndarray
		Array of length ``len(UNIFIED_LABELS)``.
	"""

	arr = np.asarray(thresholds, dtype=np.float32)
	if arr.ndim == 0:
		arr = np.full(len(UNIFIED_LABELS), float(arr), dtype=np.float32)
	if arr.shape[0] != len(UNIFIED_LABELS):
		raise ValueError(
			f"Expected thresholds of length {len(UNIFIED_LABELS)}, got shape {arr.shape}."
		)
	return arr


def _compute_risk_and_detected(probs: np.ndarray, thresholds: np.ndarray) -> Dict[str, Any]:
	"""Derive detected conditions, risk level, and confidence.

	Risk stratification is heuristic and based on the maximum predicted
	probability as well as whether any high-risk labels are strongly
	positive.

	Parameters
	----------
	probs:
		Per-label probabilities, array of shape ``(num_labels,)``.
	thresholds:
		Per-label decision thresholds.

	Returns
	-------
	dict
		Dictionary with keys ``detected_conditions``, ``risk_level`` and
		``confidence``.
	"""

	detected_mask = probs >= thresholds
	detected_indices = np.where(detected_mask)[0]
	detected_conditions = [UNIFIED_LABELS[i] for i in detected_indices]

	max_prob = float(probs.max()) if probs.size else 0.0

	high_risk_labels = {
		"AFIB",
		"STEMI",
		"HEART_FAILURE",
		"LBBB",
		"RBBB",
		"1AVB",
		"ISCHEMIA",
	}

	high_risk = False
	for label in high_risk_labels:
		if label in UNIFIED_LABELS:
			idx = UNIFIED_LABELS.index(label)
			if probs[idx] >= 0.7:
				high_risk = True
				break

	medium_risk = False
	for label, prob in zip(UNIFIED_LABELS, probs):
		if label != "NORM" and prob >= 0.5:
			medium_risk = True
			break

	if high_risk:
		risk_level = "HIGH"
	elif medium_risk:
		risk_level = "MEDIUM"
	else:
		risk_level = "LOW"

	return {
		"detected_conditions": detected_conditions,
		"risk_level": risk_level,
		"confidence": max_prob,
	}


def predict_single(
	file_path: PathLike,
	model_path: PathLike,
	thresholds: Union[Sequence[float], np.ndarray],
	tta_runs: int = 5,
) -> Dict[str, Any]:
	"""Run inference on a single ECG file.

	This function accepts a single ``.hea``/``.mat`` or ``.hea``/``.dat``
	file path, runs the full preprocessing pipeline, loads the trained
	model, and returns a rich prediction dictionary suitable for the
	web dashboard.

	Parameters
	----------
	file_path:
		Path to the ECG file (header or waveform file).
	model_path:
		Path to the saved model checkpoint (e.g. ``models/best_model.pth``).
	thresholds:
		Per-label decision thresholds (length = ``len(UNIFIED_LABELS)``).
	tta_runs:
		Number of test-time augmentation runs to average. A value of
		1 disables augmentation. Default is 5.

	Returns
	-------
	dict
		Dictionary with the following structure::

		    {
		      "predictions": {"AFIB": 0.87, "NORM": 0.12, ...},
		      "detected_conditions": ["AFIB"],
		      "risk_level": "HIGH" / "MEDIUM" / "LOW",
		      "confidence": 0.87,
		      "ecg_plot_data": [[lead1_values], [lead2_values], ...]
		    }
	"""

	ecg_path = _to_path(file_path)
	model_path = _to_path(model_path)
	logger.info("Running prediction for ECG file: %s", ecg_path)

	# ------------------------------------------------------------------
	# Load and preprocess signal
	# ------------------------------------------------------------------
	try:
		raw_signal = _load_ecg_signal(ecg_path)
	except Exception as exc:  # pragma: no cover - IO-dependent
		raise ValueError(f"Failed to load ECG from {ecg_path}: {exc}") from exc

	try:
		processed = _preprocess_for_model(raw_signal)
	except Exception as exc:  # pragma: no cover - signal-dependent
		raise ValueError(f"Failed to preprocess ECG from {ecg_path}: {exc}") from exc

	# ------------------------------------------------------------------
	# Load model
	# ------------------------------------------------------------------
	model = _load_model(model_path)
	device = torch.device(CONFIG.DEVICE)

	# ------------------------------------------------------------------
	# Test-time augmentation: average predictions over multiple passes
	# ------------------------------------------------------------------
	tta_runs = max(1, int(tta_runs))
	probs_accum = np.zeros(len(UNIFIED_LABELS), dtype=np.float32)

	with torch.no_grad():
		for _ in range(tta_runs):
			if tta_runs > 1:
				# Augment in time-major form, then convert back to (C, T)
				aug_time_major = augment_signal(processed.T).astype(np.float32)
				input_arr = pad_or_truncate(aug_time_major, target_length=CONFIG.SIGNAL_LENGTH).T
			else:
				input_arr = processed

			# Model expects (batch, 12, 5000)
			input_tensor = torch.from_numpy(input_arr[None, ...]).to(device)
			outputs = model(input_tensor, return_logits=False)
			batch_probs = outputs.cpu().numpy()[0].astype(np.float32)
			probs_accum += batch_probs

	probs_mean = probs_accum / float(tta_runs)
	thresholds_arr = _ensure_thresholds(thresholds)

	# Build per-label prediction dictionary
	predictions: Dict[str, float] = {
		label: float(prob) for label, prob in zip(UNIFIED_LABELS, probs_mean)
	}

	risk_info = _compute_risk_and_detected(probs_mean, thresholds_arr)

	# Prepare ECG plot data for the frontend: 12 lists of time-series
	ecg_plot_data = processed.tolist()  # (12, T)

	result: Dict[str, Any] = {
		"predictions": predictions,
		"detected_conditions": risk_info["detected_conditions"],
		"risk_level": risk_info["risk_level"],
		"confidence": float(risk_info["confidence"]),
		"ecg_plot_data": ecg_plot_data,
	}

	return result

