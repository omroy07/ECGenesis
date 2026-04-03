"""Global configuration for the ECG heart disease project.

This module defines a single :class:`Config` class that centralizes
all hyperparameters, paths, and label definitions used across the
training, evaluation, and inference pipelines.

Only step 2 of the overall project is implemented here: model
configuration and path management. All paths are expressed using
``pathlib.Path`` to ensure cross-platform compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


try:
	import torch
except Exception:  # pragma: no cover - torch may not be installed yet
	torch = None  # type: ignore


# Unified label set (map both datasets to these).
UNIFIED_LABELS: List[str] = [
	"NORM",
	"AFIB",
	"STEMI",
	"LBBB",
	"RBBB",
	"1AVB",
	"LVHYP",
	"ISCHEMIA",
	"PAC",
	"PVC",
	"BRADYCARDIA",
	"TACHYCARDIA",
	"HEART_FAILURE",
	"OTHER",
]


@dataclass(frozen=True)
class Config:
	"""Central configuration for ECG heart disease detection.

	This class groups together all key hyperparameters and filesystem
	locations needed by the rest of the project. The design follows the
	specification provided in the project description while ensuring
	robust, cross-platform path handling via :class:`pathlib.Path`.
	"""

	# ------------------------------------------------------------------
	# Base paths
	# ------------------------------------------------------------------
	# Root directory for the ECG project (one level above src/).
	BASE_DIR: Path = Path(__file__).resolve().parent.parent

	# Dataset directories.
	PTBXL_PATH: Path = BASE_DIR / "data" / "ptbxl"
	CHAPMAN_PATH: Path = BASE_DIR / "data" / "chapman"

	# Model and log locations.
	MODEL_SAVE_PATH: Path = BASE_DIR / "models" / "best_model.pth"
	LOG_PATH: Path = BASE_DIR / "logs" / "training_log.csv"

	# ------------------------------------------------------------------
	# ECG signal configuration
	# ------------------------------------------------------------------
	SAMPLING_RATE: int = 500  # Hz
	SIGNAL_LENGTH: int = 5000  # samples (10 seconds at 500 Hz)
	NUM_LEADS: int = 12

	# ------------------------------------------------------------------
	# Training hyperparameters
	# ------------------------------------------------------------------
	BATCH_SIZE: int = 64
	EPOCHS: int = 5
	LEARNING_RATE: float = 1e-3
	WEIGHT_DECAY: float = 1e-4
	PATIENCE: int = 15  # early stopping
	NUM_WORKERS: int = 4

	# Gradient accumulation steps (used in later training stages).
	GRADIENT_ACCUMULATION_STEPS: int = 4

	# Number of unified labels.
	NUM_CLASSES: int = len(UNIFIED_LABELS)

	# Device selection is computed dynamically so that the project can
	# run on CPU-only setups as well as GPU-enabled machines.
	@property
	def DEVICE(self) -> str:
		"""Return preferred computation device as a string.

		If PyTorch is available and a CUDA-capable GPU is detected,
		``"cuda"`` is returned. Otherwise, ``"cpu"`` is used.
		"""

		if torch is not None:
			try:
				return "cuda" if torch.cuda.is_available() else "cpu"
			except Exception:
				return "cpu"
		return "cpu"


# A convenient singleton-style instance that can be imported directly
# (e.g. ``from src.config import CONFIG``) if desired.
CONFIG = Config()

