"""Evaluation utilities for ECG heart disease detection.

This module implements step 7 of the project specification:

- :func:`evaluate` to compute rich multi-label metrics on a given
  dataloader and persist them to disk.
- :func:`find_optimal_threshold` to search per-label decision
  thresholds that maximise F1-score on a validation set.

The implementation is designed for multi-label ECG classification and
is compatible with the rest of the project modules.
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    roc_auc_score,
)
from torch import nn

from .config import CONFIG, UNIFIED_LABELS
from .preprocessing import augment_signal

logger = logging.getLogger(__name__)


TensorOrArray = Union[torch.Tensor, np.ndarray]
ThresholdLike = Union[float, Sequence[float], np.ndarray]


def _to_numpy(x: TensorOrArray) -> np.ndarray:
    """Convert a tensor or array to a detached CPU ``numpy.ndarray``.

    Parameters
    ----------
    x:
        Input tensor or array.

    Returns
    -------
    numpy.ndarray
        Detached CPU array.
    """

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_thresholds(threshold: ThresholdLike, num_classes: int) -> np.ndarray:
    """Return a per-class threshold vector of length ``num_classes``.

    A scalar threshold is broadcast to all classes; a sequence is
    converted to a numpy array and truncated/padded as needed.
    """

    if isinstance(threshold, (float, int)):
        return np.full(num_classes, float(threshold), dtype=float)

    arr = np.asarray(list(threshold), dtype=float)
    if arr.size < num_classes:
        pad = np.full(num_classes - arr.size, float(arr[0] if arr.size > 0 else 0.5))
        arr = np.concatenate([arr, pad])
    elif arr.size > num_classes:
        arr = arr[:num_classes]
    return arr


def _compute_per_class_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, List[List[int]]]]:
    """Compute per-class AUC, F1, precision, recall and confusion matrices.

    Parameters
    ----------
    y_true:
        Binary ground-truth matrix of shape ``(N, C)``.
    y_score:
        Probabilistic predictions in ``[0, 1]`` of shape ``(N, C)``.
    thresholds:
        Per-class decision thresholds of shape ``(C,)``.

    Returns
    -------
    (auc, f1, precision, recall): dicts
        Per-class metric dictionaries keyed by label name.
    confusion_mats: dict
        Confusion matrices for each label as 2×2 integer lists.
    """

    num_classes = y_true.shape[1]

    auc_per_class: Dict[str, float] = {}
    f1_per_class: Dict[str, float] = {}
    prec_per_class: Dict[str, float] = {}
    rec_per_class: Dict[str, float] = {}
    conf_mats: Dict[str, List[List[int]]] = {}

    for idx in range(num_classes):
        label = UNIFIED_LABELS[idx] if idx < len(UNIFIED_LABELS) else f"label_{idx}"
        y_t = y_true[:, idx]
        y_s = y_score[:, idx]
        thr = thresholds[idx]

        # Predicted labels for this class
        y_p = (y_s >= thr).astype(int)

        # AUC-ROC (skip if only one class present)
        if np.unique(y_t).size < 2:
            auc = float("nan")
        else:
            try:
                auc = float(roc_auc_score(y_t, y_s))
            except ValueError:
                auc = float("nan")
        auc_per_class[label] = auc

        # F1 score
        if y_t.sum() == 0:
            f1_val = float("nan")
        else:
            f1_val = float(f1_score(y_t, y_p, zero_division=0))
        f1_per_class[label] = f1_val

        # Precision and recall from confusion matrix
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec_per_class[label] = prec
        rec_per_class[label] = rec
        conf_mats[label] = cm.astype(int).tolist()

    return auc_per_class, f1_per_class, prec_per_class, rec_per_class, conf_mats


def _macro_from_per_class(metrics: Mapping[str, float]) -> float:
    """Compute the macro-average from a per-class metric mapping.

    ``NaN`` values are ignored when computing the mean.
    """

    values = np.asarray(list(metrics.values()), dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    threshold: ThresholdLike = 0.5,
    tta_runs: int = 5,
) -> Dict[str, Any]:
    """Evaluate a trained model on a dataloader.

    Compute and return:

    - Per-class AUC-ROC
    - Macro-average AUC-ROC
    - Per-class F1 score
    - Macro F1
    - Hamming loss
    - Confusion matrix per label
    - Classification report string

    The metrics are also saved to ``logs/evaluation_results.json``.

    Parameters
    ----------
    model:
        Trained multi-label model (e.g. :class:`ECGResNet`).
    dataloader:
        Dataloader providing ``(inputs, targets)`` batches.
    threshold:
        Scalar or sequence specifying the decision threshold(s) to use
        when binarising probabilities.
    tta_runs:
        Number of test-time augmentation runs. If greater than 1, each
        batch is evaluated multiple times with random augmentations and
        predictions are averaged.

    Returns
    -------
    dict
        Dictionary containing the computed metrics.
    """

    device = torch.device(CONFIG.DEVICE)
    model = model.to(device)
    model.eval()

    amp_enabled = device.type == "cuda"
    autocast_cm = torch.cuda.amp.autocast if amp_enabled else nullcontext

    all_targets: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            batch_size = inputs.size(0)

            # Base CPU tensors from dataloader
            inputs_cpu = inputs
            targets_cpu = targets

            if tta_runs <= 1:
                with autocast_cm():
                    logits = model(inputs_cpu.to(device), return_logits=True)
                    probs = torch.sigmoid(logits)
                all_probs.append(_to_numpy(probs))
            else:
                # Test-time augmentation: average predictions across runs
                probs_accum: Optional[np.ndarray] = None
                for _ in range(tta_runs):
                    # Apply augmentation per sample on CPU
                    augmented_samples: List[torch.Tensor] = []
                    for i in range(batch_size):
                        # Convert (C, T) -> (T, C) for augmentation
                        sample_np = inputs_cpu[i].cpu().numpy().T  # (T, C)
                        aug_np = augment_signal(sample_np)
                        # Back to (C, T)
                        aug_tensor = torch.from_numpy(aug_np.T.astype(np.float32))
                        augmented_samples.append(aug_tensor)

                    aug_batch = torch.stack(augmented_samples, dim=0)
                    with autocast_cm():
                        logits = model(aug_batch.to(device), return_logits=True)
                        probs = torch.sigmoid(logits)
                    probs_np = _to_numpy(probs)
                    probs_accum = probs_np if probs_accum is None else probs_accum + probs_np

                assert probs_accum is not None
                all_probs.append(probs_accum / float(tta_runs))

            all_targets.append(_to_numpy(targets_cpu))

    if not all_targets:
        raise RuntimeError("Evaluation dataloader produced no batches.")

    y_true = np.concatenate(all_targets, axis=0)
    y_score = np.concatenate(all_probs, axis=0)

    if y_true.ndim != 2 or y_score.ndim != 2:
        raise ValueError(
            f"Expected 2D label and score matrices, got shapes {y_true.shape} and {y_score.shape}."
        )

    num_classes = y_true.shape[1]
    thresholds = _normalize_thresholds(threshold, num_classes)

    # Per-class metrics
    auc_per_class, f1_per_class, prec_per_class, rec_per_class, conf_mats = _compute_per_class_metrics(
        y_true, y_score, thresholds
    )

    macro_auc = _macro_from_per_class(auc_per_class)
    macro_f1 = _macro_from_per_class(f1_per_class)

    # Binarised predictions for global metrics
    y_pred = (y_score >= thresholds[None, :]).astype(int)

    # Hamming loss (fraction of misclassified labels)
    hamming = float(hamming_loss(y_true, y_pred))

    # Subset accuracy over multi-label outputs
    overall_accuracy = float(accuracy_score(y_true, y_pred))

    # Classification report
    try:
        clf_report_str = classification_report(
            y_true,
            y_pred,
            target_names=UNIFIED_LABELS[:num_classes],
            zero_division=0,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to compute classification report: %s", exc)
        clf_report_str = f"Classification report computation failed: {exc}"

    metrics: Dict[str, Any] = {
        "labels": UNIFIED_LABELS[:num_classes],
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(num_classes),
        "thresholds": thresholds.tolist(),
        "per_class_auc": auc_per_class,
        "macro_auc": macro_auc,
        "per_class_f1": f1_per_class,
        "macro_f1": macro_f1,
        "per_class_precision": prec_per_class,
        "per_class_recall": rec_per_class,
        "overall_accuracy": overall_accuracy,
        "hamming_loss": hamming,
        "confusion_matrices": conf_mats,
        "classification_report": clf_report_str,
    }

    # Persist metrics to logs/evaluation_results.json
    logs_dir = CONFIG.LOG_PATH.parent
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_path = logs_dir / "evaluation_results.json"

    try:
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except OSError as exc:  # pragma: no cover - IO-dependent
        logger.error("Failed to write evaluation results to %s: %s", eval_path, exc)

    logger.info(
        "Evaluation complete: macro AUC=%.4f, macro F1=%.4f, accuracy=%.4f, hamming=%.4f",
        macro_auc,
        macro_f1,
        overall_accuracy,
        hamming,
    )

    return metrics


def find_optimal_threshold(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    search_range: Tuple[float, float] = (0.1, 0.9),
    step: float = 0.05,
) -> np.ndarray:
    """Find per-label thresholds that maximise F1 on a validation set.

    Per-label threshold optimisation using the validation set. For each
    label independently, thresholds are scanned in the given range and
    the one yielding the highest F1 score is selected.

    Parameters
    ----------
    model:
        Trained multi-label model.
    val_loader:
        Validation dataloader.
    search_range:
        Tuple ``(low, high)`` describing the inclusive search range.
    step:
        Step size for threshold search.

    Returns
    -------
    numpy.ndarray
        Array of optimal thresholds of length ``num_classes``.
    """

    device = torch.device(CONFIG.DEVICE)
    model = model.to(device)
    model.eval()

    amp_enabled = device.type == "cuda"
    autocast_cm = torch.cuda.amp.autocast if amp_enabled else nullcontext

    all_targets: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            with autocast_cm():
                logits = model(inputs.to(device), return_logits=True)
                probs = torch.sigmoid(logits)
            all_scores.append(_to_numpy(probs))
            all_targets.append(_to_numpy(targets))

    if not all_targets:
        raise RuntimeError("Validation dataloader produced no batches.")

    y_true = np.concatenate(all_targets, axis=0)
    y_score = np.concatenate(all_scores, axis=0)

    if y_true.ndim != 2 or y_score.ndim != 2:
        raise ValueError(
            f"Expected 2D label and score matrices, got shapes {y_true.shape} and {y_score.shape}."
        )

    num_classes = y_true.shape[1]
    thresholds = np.full(num_classes, 0.5, dtype=float)

    low, high = search_range
    thr_values = np.arange(low, high + 1e-8, step, dtype=float)

    for idx in range(num_classes):
        y_t = y_true[:, idx]
        y_s = y_score[:, idx]

        if np.unique(y_t).size < 2:
            # If only one class present, keep default threshold
            continue

        best_thr = 0.5
        best_f1 = -1.0

        for thr in thr_values:
            y_p = (y_s >= thr).astype(int)
            f1_val = float(f1_score(y_t, y_p, zero_division=0))
            if f1_val > best_f1:
                best_f1 = f1_val
                best_thr = thr

        thresholds[idx] = best_thr

    logger.info("Optimised thresholds per label: %s", thresholds)
    return thresholds
