"""Unified ECG dataset and dataloaders.

This module implements step 4 of the project specification:

- A single :class:`ECGDataset` that transparently combines the PTB-XL
  and Chapman-Shaoxing datasets into one multi-label dataset.
- A :func:`get_dataloaders` helper that creates stratified
  train/validation/test splits using ``MultilabelStratifiedKFold`` from
  :mod:`iterstrat.ml_stratifiers` and a :class:`WeightedRandomSampler`
  to mitigate class imbalance in the training set.

All signals are preprocessed to a common shape of ``(12, 5000)`` using
functions from :mod:`src.preprocessing`, and all labels are mapped into
:data:`src.config.UNIFIED_LABELS`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import CONFIG, UNIFIED_LABELS
from .preprocessing import (
    augment_signal,
    bandpass_filter,
    load_chapman_record,
    load_ptbxl_record,
    map_chapman_labels,
    map_ptbxl_labels,
    normalize_signal,
    pad_or_truncate,
)

try:  # iterstrat is required for multi-label stratification
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except ImportError as exc:  # pragma: no cover - dependency error
    MultilabelStratifiedKFold = None  # type: ignore[misc]
    _iterstrat_import_error = exc
else:
    _iterstrat_import_error = None


logger = logging.getLogger(__name__)


@dataclass
class RecordInfo:
    """Container describing a single ECG record.

    Attributes
    ----------
    file_path:
        Path to the record header (``.hea``) or base record name
        (PTB-XL). For Chapman records this is typically the ``.hea``
        file; for PTB-XL it may be the base name used with WFDB.
    dataset_source:
        String identifier for the dataset source (``"ptbxl"`` or
        ``"chapman"``).
    labels_vector:
        Binary numpy array of shape ``(len(UNIFIED_LABELS),)``.
    """

    file_path: Path
    dataset_source: str
    labels_vector: np.ndarray


class ECGDataset(Dataset):
    """Unified PyTorch dataset for PTB-XL and Chapman ECG records.

    The dataset scans both the PTB-XL and Chapman folders, builds a
    master :class:`pandas.DataFrame` with columns
    ``[file_path, dataset_source, labels_vector]`` and serves
    preprocessed, optionally augmented ECG signals suitable for model
    training.

    Notes
    -----
    - All signals are returned as ``torch.FloatTensor`` of shape
      ``(12, 5000)`` (channels-first) using the configuration from
      :mod:`src.config`.
    - Labels are returned as ``torch.FloatTensor`` binary vectors of
      shape ``(len(UNIFIED_LABELS),)``.
    - When ``augment=True``, data augmentation is applied on-the-fly
      using :func:`augment_signal` **only** for the training dataset.
    """

    def __init__(
        self,
        ptbxl_path: Optional[Path] = None,
        chapman_path: Optional[Path] = None,
        augment: bool = False,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Create a new unified ECG dataset.

        Parameters
        ----------
        ptbxl_path:
            Base directory of the PTB-XL dataset. Defaults to
            :data:`CONFIG.PTBXL_PATH`.
        chapman_path:
            Base directory of the Chapman-Shaoxing dataset. Defaults to
            :data:`CONFIG.CHAPMAN_PATH`.
        augment:
            If ``True``, apply data augmentation when returning
            samples. Intended for the training split.
        metadata_df:
            Optional pre-built metadata frame with the required
            columns. If provided, directory scanning is skipped and the
            given metadata is used directly.
        """

        super().__init__()

        self.augment = augment
        self.ptbxl_path = (ptbxl_path or CONFIG.PTBXL_PATH).expanduser().resolve()
        # Use the configured Chapman dataset path by default.
        self.chapman_path = (chapman_path or CONFIG.CHAPMAN_PATH).expanduser().resolve()

        if metadata_df is not None:
            self.records = metadata_df.reset_index(drop=True)
        else:
            ptb_df = self._scan_ptbxl(self.ptbxl_path)
            chap_df = self._scan_chapman(self.chapman_path)

            if ptb_df is None and chap_df is None:
                raise RuntimeError(
                    "No ECG records found in PTB-XL or Chapman directories. "
                    f"Checked PTB-XL at {self.ptbxl_path} and Chapman at {self.chapman_path}."
                )

            frames = [df for df in (ptb_df, chap_df) if df is not None]
            self.records = pd.concat(frames, ignore_index=True)

        if self.records.empty:
            raise RuntimeError("ECGDataset metadata is empty; no samples available.")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        """Return the total number of ECG samples in the dataset."""

        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Return a single preprocessed ECG sample and its labels.

        The processing pipeline is:

        1. Load raw signal based on ``dataset_source``.
        2. Apply :func:`bandpass_filter`.
        3. Apply :func:`normalize_signal`.
        4. Apply :func:`pad_or_truncate`.
        5. Optionally apply :func:`augment_signal` if
           :attr:`augment` is ``True``.
        6. Convert to ``torch.FloatTensor`` of shape ``(12, 5000)``.
        7. Return ``(signal_tensor, label_tensor)``.
        """

        row = self.records.iloc[int(index)]
        file_path: Path = Path(row["file_path"])
        dataset_source: str = str(row["dataset_source"]).lower()
        labels_vector: np.ndarray = np.asarray(row["labels_vector"], dtype=np.float32)

        # Load signal according to dataset source
        if dataset_source == "ptbxl":
            signal = load_ptbxl_record(file_path)
        elif dataset_source == "chapman":
            signal = load_chapman_record(file_path)
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown dataset_source '{dataset_source}' for file {file_path}")

        # Apply preprocessing pipeline
        signal = bandpass_filter(signal, fs=CONFIG.SAMPLING_RATE)
        signal = normalize_signal(signal)
        signal = pad_or_truncate(signal, target_length=CONFIG.SIGNAL_LENGTH)

        if self.augment:
            signal = augment_signal(signal)

        # Convert to (channels, time) tensor (12, 5000)
        signal_arr = np.asarray(signal, dtype=np.float32)
        if signal_arr.ndim != 2:
            raise ValueError(f"Expected 2D signal after preprocessing, got shape {signal_arr.shape}.")

        # Ensure (T, L) then transpose to (L, T)
        if signal_arr.shape[1] == CONFIG.NUM_LEADS:
            # Already (T, 12)
            pass
        elif signal_arr.shape[0] == CONFIG.NUM_LEADS:
            signal_arr = signal_arr.T
        else:
            # Fallback: treat last dimension as leads
            if signal_arr.shape[-1] != CONFIG.NUM_LEADS:
                raise ValueError(
                    "Signal does not have expected number of leads "
                    f"({CONFIG.NUM_LEADS}); got shape {signal_arr.shape}."
                )

        signal_tensor = torch.from_numpy(signal_arr.T)  # (12, 5000)
        label_tensor = torch.from_numpy(labels_vector.astype(np.float32))

        return signal_tensor, label_tensor

    # ------------------------------------------------------------------
    # Internal helpers to scan datasets
    # ------------------------------------------------------------------
    @staticmethod
    def _scan_ptbxl(root: Path) -> Optional[pd.DataFrame]:
        """Scan the PTB-XL directory and build metadata.

        Parameters
        ----------
        root:
            Base directory containing ``ptbxl_database.csv`` and
            ``scp_statements.csv`` as well as the WFDB record files.

        Returns
        -------
        pandas.DataFrame or None
            Metadata frame with columns
            ``[file_path, dataset_source, labels_vector]`` or ``None``
            if the directory does not exist or required files are
            missing.
        """

        if not root.exists():
            logger.warning("PTB-XL directory %s does not exist; skipping.", root)
            return None

        db_path = root / "ptbxl_database.csv"
        scp_path = root / "scp_statements.csv"

        if not db_path.exists() or not scp_path.exists():
            logger.warning(
                "PTB-XL metadata files not found at %s and %s; skipping PTB-XL.",
                db_path,
                scp_path,
            )
            return None

        try:
            db_df = pd.read_csv(db_path)
            scp_df = pd.read_csv(scp_path)
        except Exception as exc:  # pragma: no cover - IO-dependent
            logger.error("Failed to read PTB-XL metadata CSVs: %s", exc)
            return None

        if db_df.empty:
            logger.warning("PTB-XL database CSV %s is empty; skipping.", db_path)
            return None

        # Prefer 500 Hz records if column available
        if "sampling_frequency" in db_df.columns:
            db_df = db_df[db_df["sampling_frequency"] == CONFIG.SAMPLING_RATE]

        entries: List[RecordInfo] = []

        for _, row in db_df.iterrows():
            try:
                scp_codes = row["scp_codes"]
            except KeyError:
                continue

            labels_vec = map_ptbxl_labels(scp_codes, scp_df)

            # Determine relative path to WFDB record (500 Hz high-res)
            rel_path: Optional[str] = None
            for col in ("filename_hr", "filename", "filepath", "filename_lr"):
                if col in db_df.columns and isinstance(row.get(col), str):
                    rel_path = row.get(col)
                    break

            if not rel_path:
                continue

            file_path = (root / rel_path).with_suffix("")  # base name for WFDB
            entries.append(
                RecordInfo(
                    file_path=file_path,
                    dataset_source="ptbxl",
                    labels_vector=labels_vec,
                )
            )

        if not entries:
            logger.warning("No PTB-XL records discovered in %s.", root)
            return None

        data = {
            "file_path": [e.file_path for e in entries],
            "dataset_source": [e.dataset_source for e in entries],
            "labels_vector": [e.labels_vector for e in entries],
        }
        return pd.DataFrame(data)

    @staticmethod
    def _scan_chapman(root: Path) -> Optional[pd.DataFrame]:
        """Scan the Chapman directory and build metadata.

        Parameters
        ----------
        root:
            Base directory containing Chapman ``.hea`` and ``.mat``
            files.

        Returns
        -------
        pandas.DataFrame or None
            Metadata frame with columns
            ``[file_path, dataset_source, labels_vector]`` or ``None``
            if the directory does not exist or contains no valid
            records.
        """

        if not root.exists():
            logger.warning("Chapman directory %s does not exist; skipping.", root)
            return None

        hea_files = sorted(root.rglob("*.hea"))
        if not hea_files:
            logger.warning("No Chapman .hea files found under %s; skipping.", root)
            return None

        entries: List[RecordInfo] = []

        for hea_path in hea_files:
            try:
                header_text = hea_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:  # pragma: no cover - IO-dependent
                logger.error("Failed to read Chapman header %s: %s", hea_path, exc)
                continue

            labels_vec = map_chapman_labels(header_text)
            entries.append(
                RecordInfo(
                    file_path=hea_path,
                    dataset_source="chapman",
                    labels_vector=labels_vec,
                )
            )

        if not entries:
            logger.warning("No valid Chapman records discovered under %s.", root)
            return None

        data = {
            "file_path": [e.file_path for e in entries],
            "dataset_source": [e.dataset_source for e in entries],
            "labels_vector": [e.labels_vector for e in entries],
        }
        return pd.DataFrame(data)


def _check_iterstrat_available() -> None:
    """Raise a helpful error if ``iterstrat`` is not installed."""

    if MultilabelStratifiedKFold is None:  # type: ignore[truthy-function]
        raise ImportError(
            "MultilabelStratifiedKFold is required for stratified splitting. "
            "Please install the 'iterstrat' package, for example with:\n"
            "  pip install iterstrat"
        ) from _iterstrat_import_error


def _compute_stratified_splits(
    labels: np.ndarray,
    val_split: float,
    test_split: float,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute stratified train/val/test splits for multi-label data.

    This function uses :class:`MultilabelStratifiedKFold` in two stages
    to approximate the desired validation and test fractions while
    maintaining label distribution across splits.

    Parameters
    ----------
    labels:
        Binary label matrix of shape ``(N, C)``.
    val_split:
        Desired fraction of samples in the validation split.
    test_split:
        Desired fraction of samples in the test split.
    random_state:
        Random seed for reproducible splits.

    Returns
    -------
    (train_idx, val_idx, test_idx): tuple of numpy.ndarray
        Index arrays into the original dataset.
    """

    _check_iterstrat_available()

    num_samples = labels.shape[0]
    indices = np.arange(num_samples)

    # -------------------- Test split --------------------
    # Approximate the desired test fraction by choosing an integer
    # number of folds.
    test_folds = max(2, int(round(1.0 / max(test_split, 1e-6))))
    mskf_test = MultilabelStratifiedKFold(
        n_splits=test_folds, shuffle=True, random_state=random_state
    )

    # Take the first fold as test
    train_val_idx, test_idx = next(mskf_test.split(indices, labels))

    # -------------------- Validation split --------------------
    remaining_fraction = 1.0 - float(len(test_idx)) / float(num_samples)
    desired_val_fraction_within_train_val = val_split / max(remaining_fraction, 1e-6)
    val_folds = max(2, int(round(1.0 / max(desired_val_fraction_within_train_val, 1e-6))))

    mskf_val = MultilabelStratifiedKFold(
        n_splits=val_folds, shuffle=True, random_state=random_state + 1
    )

    train_idx_rel, val_idx_rel = next(
        mskf_val.split(train_val_idx, labels[train_val_idx])
    )

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    return train_idx, val_idx, test_idx


def _create_weighted_sampler(labels: np.ndarray, indices: np.ndarray) -> WeightedRandomSampler:
    """Create a :class:`WeightedRandomSampler` for the training set.

    Sample weights are computed from inverse class frequencies so that
    under-represented labels are sampled more often.

    Parameters
    ----------
    labels:
        Binary label matrix of shape ``(N, C)`` for the *full* dataset.
    indices:
        Index array specifying which rows belong to the training set.

    Returns
    -------
    torch.utils.data.WeightedRandomSampler
        Sampler producing indices into the training subset.
    """

    train_labels = labels[indices]
    # Class frequencies and inverse-frequency weights
    class_counts = train_labels.sum(axis=0) + 1e-6
    class_weights = 1.0 / class_counts

    # Per-sample weight is the sum of its positive class weights
    sample_weights = (train_labels * class_weights).sum(axis=1)
    # Ensure no sample has zero weight (e.g. all-zero labels)
    max_weight = float(class_weights.max())
    sample_weights = np.where(sample_weights > 0.0, sample_weights, max_weight)

    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(indices), replacement=True)


def get_dataloaders(
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create stratified train/validation/test dataloaders.

    The function performs the following steps:

    - Instantiate :class:`ECGDataset` over the full combined dataset.
    - Use :class:`MultilabelStratifiedKFold` (via
      :func:`_compute_stratified_splits`) to obtain stratified
      train/validation/test indices.
    - Create three :class:`ECGDataset` instances using the same
      underlying metadata, with ``augment=True`` **only** for the
      training split.
    - Use :class:`WeightedRandomSampler` on the training split to
      mitigate class imbalance.

    Parameters
    ----------
    val_split:
        Approximate fraction of samples to allocate to the validation
        split (default 0.15).
    test_split:
        Approximate fraction of samples to allocate to the test split
        (default 0.15).

    Returns
    -------
    (train_loader, val_loader, test_loader): tuple of DataLoader
        DataLoaders for training, validation and testing.
    """

    if val_split <= 0.0 or test_split <= 0.0 or val_split + test_split >= 0.9:
        raise ValueError(
            "val_split and test_split must be positive and leave at least 10% "
            "of the data for training."
        )

    full_dataset = ECGDataset(augment=False)
    metadata_df = full_dataset.records

    # Build label matrix for stratification
    label_matrix = np.stack(
        [np.asarray(v, dtype=np.float32) for v in metadata_df["labels_vector"].to_list()]
    )

    train_idx, val_idx, test_idx = _compute_stratified_splits(
        labels=label_matrix, val_split=val_split, test_split=test_split
    )

    # Create per-split datasets using the same metadata
    train_metadata = metadata_df.iloc[train_idx].reset_index(drop=True)
    val_metadata = metadata_df.iloc[val_idx].reset_index(drop=True)
    test_metadata = metadata_df.iloc[test_idx].reset_index(drop=True)

    train_dataset = ECGDataset(augment=True, metadata_df=train_metadata)
    val_dataset = ECGDataset(augment=False, metadata_df=val_metadata)
    test_dataset = ECGDataset(augment=False, metadata_df=test_metadata)

    # Weighted sampler for training
    sampler = _create_weighted_sampler(label_matrix, train_idx)

    pin_memory = CONFIG.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        sampler=sampler,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=False,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=False,
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        "Created dataloaders: %d train, %d val, %d test samples.",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    return train_loader, val_loader, test_loader
