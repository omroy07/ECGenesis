"""Training loop for ECG heart disease detection.

This module implements step 6 of the project specification:

- Load dataloaders from :mod:`src.dataset`.
- Initialise the :class:`ECGResNet` model from :mod:`src.model`.
- Use a BCE-with-logits loss with class-balanced ``pos_weight`` and
  label smoothing as a robust alternative to AsymmetricLoss.
- Optimise with AdamW and a CosineAnnealingWarmRestarts scheduler with
  an initial warm-up phase.
- Apply gradient clipping, gradient accumulation, mixup augmentation,
  and (indirectly via the dataset) data augmentation.
- Log training and validation metrics to ``logs/training_log.csv``.
- Save the best model checkpoint based on validation AUC.
- After training, run a full evaluation on the test set using
  :func:`src.evaluate.evaluate` and print per-class metrics.
"""

from __future__ import annotations

import csv
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CONFIG, UNIFIED_LABELS
from .dataset import ECGDataset, get_dataloaders
from .evaluate import evaluate, find_optimal_threshold
from .model import build_ecg_resnet


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
	"""Detach a tensor and move it to CPU as a ``numpy.ndarray``."""

	return x.detach().cpu().numpy()


def _compute_pos_weight_from_metadata(records_df: pd.DataFrame) -> torch.Tensor:
	"""Compute ``pos_weight`` for BCE loss from dataset label frequencies.

	The positive class weight for each label is defined as

	.. math::

		w_c = \frac{N - P_c}{P_c}

	where :math:`P_c` is the number of positive samples for class
	:math:`c` and :math:`N` is the total number of samples. Classes with
	no positive samples receive a neutral weight of 1.0.
	"""

	label_matrix = np.stack(
		[np.asarray(v, dtype=np.float32) for v in records_df["labels_vector"].to_list()]
	)
	num_samples, num_classes = label_matrix.shape

	pos_counts = label_matrix.sum(axis=0)
	neg_counts = num_samples - pos_counts

	# Avoid division by zero for rare classes
	pos_weight = np.where(pos_counts > 0.0, neg_counts / (pos_counts + 1e-6), 1.0)
	return torch.as_tensor(pos_weight, dtype=torch.float32)


def _smooth_labels(targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
	"""Apply label smoothing for multi-label targets.

	With smoothing factor :math:`\alpha`, targets are mapped as

	.. math::

		y_\text{smooth} = y * (1 - \alpha) + 0.5 * \alpha

	which moves positive labels towards 0.95 and negatives towards 0.05
	for :math:`\alpha = 0.1`.
	"""

	if smoothing <= 0.0:
		return targets
	return targets * (1.0 - smoothing) + 0.5 * smoothing


def _compute_bce_loss(
	logits: torch.Tensor,
	targets: torch.Tensor,
	criterion: nn.BCEWithLogitsLoss,
	smoothing: float = 0.1,
) -> torch.Tensor:
	"""Compute BCE-with-logits loss with optional label smoothing."""

	targets_smooth = _smooth_labels(targets, smoothing)
	return criterion(logits, targets_smooth)


def _mixup_data(
	x: torch.Tensor,
	y: torch.Tensor,
	alpha: float = 0.2,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Apply mixup augmentation to a batch.

	Parameters
	----------
	x:
		Input batch of shape ``(B, C, T)``.
	y:
		Multi-label targets of shape ``(B, num_classes)``.
	alpha:
		Mixup Beta distribution parameter. If ``alpha <= 0`` or the
		batch contains a single sample, mixup is disabled.
	"""

	batch_size = x.size(0)
	if alpha <= 0.0 or batch_size <= 1:
		return 1.0, x, y, y

	lam = np.random.beta(alpha, alpha)
	# Use the larger lambda to preserve signal strength
	lam = float(max(lam, 1.0 - lam))

	index = torch.randperm(batch_size, device=x.device)
	mixed_x = lam * x + (1.0 - lam) * x[index]
	y_a, y_b = y, y[index]
	return lam, mixed_x, y_a, y_b


def _compute_macro_auc_f1(
	y_true: np.ndarray,
	y_score: np.ndarray,
	threshold: float = 0.5,
) -> Tuple[float, float]:
	"""Compute macro-averaged AUC and F1 for multi-label data."""

	num_classes = y_true.shape[1]
	aucs: List[float] = []
	f1s: List[float] = []

	for idx in range(num_classes):
		y_t = y_true[:, idx]
		y_s = y_score[:, idx]

		# AUC per class
		if np.unique(y_t).size < 2:
			auc_val = float("nan")
		else:
			try:
				auc_val = float(roc_auc_score(y_t, y_s))
			except ValueError:
				auc_val = float("nan")
		aucs.append(auc_val)

		# F1 per class
		y_p = (y_s >= threshold).astype(int)
		if y_t.sum() == 0:
			f1_val = float("nan")
		else:
			f1_val = float(f1_score(y_t, y_p, zero_division=0))
		f1s.append(f1_val)

	macro_auc = float(np.nanmean(aucs))
	macro_f1 = float(np.nanmean(f1s))
	return macro_auc, macro_f1


def _init_training_log(log_path: Path) -> None:
	"""Ensure the training log CSV exists with a header row."""

	log_path.parent.mkdir(parents=True, exist_ok=True)
	if log_path.exists():
		return

	with log_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["epoch", "train_loss", "val_loss", "val_auc", "val_f1", "lr", "time_sec"])


def _append_training_log(
	log_path: Path,
	epoch: int,
	train_loss: float,
	val_loss: float,
	val_auc: float,
	val_f1: float,
	lr: float,
	time_sec: float,
) -> None:
	"""Append a single epoch's metrics to the training log CSV."""

	try:
		with log_path.open("a", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(
				[
					int(epoch),
					float(train_loss),
					float(val_loss),
					float(val_auc),
					float(val_f1),
					float(lr),
					float(time_sec),
				]
			)
	except OSError as exc:  # pragma: no cover - IO-dependent
		logger.error("Failed to append to training log %s: %s", log_path, exc)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(num_epochs: int = CONFIG.EPOCHS) -> Dict[str, Any]:
	"""Train the ECGResNet model on the combined ECG datasets.

	The training procedure performs the following steps:

	1. Load stratified train/validation/test dataloaders.
	2. Initialise :class:`ECGResNet` and move it to the configured
	   device (CPU or GPU).
	3. Construct a BCE-with-logits loss with ``pos_weight`` computed
	   from the training label distribution and apply label smoothing
	   (factor 0.1).
	4. Optimise with AdamW (``lr=1e-3``, ``weight_decay=1e-4``) and a
	   CosineAnnealingWarmRestarts scheduler (``T_0=10``, ``T_mult=2``)
	   with a 5-epoch linear warm-up.
	5. Use gradient accumulation (``GRADIENT_ACCUMULATION_STEPS``),
	   gradient clipping (max-norm 1.0), and mixup augmentation
	   (``alpha=0.2``) during training.
	6. After each epoch, evaluate on the validation set, compute macro
	   AUC/F1, log metrics to ``logs/training_log.csv``, and save the
	   best model based on validation AUC with early stopping
	   (patience 15 epochs).
	7. After training completes, reload the best model, optimise
	   per-label thresholds on the validation set, and run a final
	   evaluation on the test set, printing summary metrics.

	Parameters
	----------
	num_epochs:
		Maximum number of training epochs (defaults to
		:data:`CONFIG.EPOCHS`).

	Returns
	-------
	dict
		Metrics dictionary returned by :func:`src.evaluate.evaluate`
		for the final test-set evaluation.
	"""

	device = torch.device(CONFIG.DEVICE)
	logger.info("Starting training on device %s", device)
	print(f"[train] Using device: {device}")

	# ------------------------------------------------------------------
	# Data
	# ------------------------------------------------------------------
	train_loader, val_loader, test_loader = get_dataloaders()

	if not isinstance(train_loader.dataset, ECGDataset):
		raise RuntimeError("Expected train_loader.dataset to be an ECGDataset instance.")

	train_records = train_loader.dataset.records  # type: ignore[attr-defined]
	pos_weight = _compute_pos_weight_from_metadata(train_records).to(device)

	# ------------------------------------------------------------------
	# Model, loss, optimiser, scheduler
	# ------------------------------------------------------------------
	model = build_ecg_resnet(num_classes=CONFIG.NUM_CLASSES).to(device)

	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
		optimizer, T_0=10, T_mult=2
	)

	amp_enabled = device.type == "cuda"
	scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
	autocast_cm = torch.cuda.amp.autocast if amp_enabled else nullcontext

	accum_steps = max(1, CONFIG.GRADIENT_ACCUMULATION_STEPS)
	warmup_epochs = 5

	# Logging and checkpoint paths
	_init_training_log(CONFIG.LOG_PATH)
	CONFIG.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

	best_val_auc = 0.0
	best_epoch = -1
	epochs_no_improve = 0

	global_start_time = time.time()

	for epoch in range(num_epochs):
		epoch_start = time.time()

		# -------------------- Training phase --------------------
		model.train()

		# Warm-up learning rate for the first few epochs
		if epoch < warmup_epochs:
			warmup_factor = float(epoch + 1) / float(warmup_epochs)
			base_lr = CONFIG.LEARNING_RATE
			for pg in optimizer.param_groups:
				pg["lr"] = base_lr * warmup_factor
		else:
			# Cosine annealing schedule after warm-up
			scheduler.step(epoch - warmup_epochs)

		train_loss_sum = 0.0
		num_train_samples = 0

		optimizer.zero_grad(set_to_none=True)

		with tqdm(
			train_loader,
			desc=f"Epoch {epoch + 1}/{num_epochs} [train]",
			unit="batch",
			ncols=100,
		) as pbar:
			for step, (inputs, targets) in enumerate(pbar):
				inputs = inputs.to(device, non_blocking=True)
				targets = targets.to(device, non_blocking=True)

				batch_size = inputs.size(0)
				num_train_samples += batch_size

				# Mixup augmentation in feature space
				lam, mixed_x, y_a, y_b = _mixup_data(inputs, targets, alpha=0.2)

				with autocast_cm():
					logits = model(mixed_x, return_logits=True)
					if lam == 1.0:
						raw_loss = _compute_bce_loss(logits, y_a, criterion, smoothing=0.1)
					else:
						loss_a = _compute_bce_loss(logits, y_a, criterion, smoothing=0.1)
						loss_b = _compute_bce_loss(logits, y_b, criterion, smoothing=0.1)
						raw_loss = lam * loss_a + (1.0 - lam) * loss_b

				# Gradient accumulation
				loss = raw_loss / float(accum_steps)
				scaler.scale(loss).backward()

				train_loss_sum += raw_loss.detach().item() * batch_size

				if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
					if amp_enabled:
						scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
					scaler.step(optimizer)
					scaler.update()
					optimizer.zero_grad(set_to_none=True)

				current_lr = optimizer.param_groups[0]["lr"]
				avg_train_loss = train_loss_sum / max(1, num_train_samples)
				pbar.set_postfix(loss=f"{avg_train_loss:.4f}", lr=f"{current_lr:.2e}")

		train_loss = train_loss_sum / max(1, num_train_samples)

		# -------------------- Validation phase --------------------
		model.eval()
		val_loss_sum = 0.0
		num_val_samples = 0
		all_val_targets: List[np.ndarray] = []
		all_val_scores: List[np.ndarray] = []

		with torch.no_grad():
			for inputs, targets in val_loader:
				inputs = inputs.to(device, non_blocking=True)
				targets = targets.to(device, non_blocking=True)

				batch_size = inputs.size(0)
				num_val_samples += batch_size

				with autocast_cm():
					logits = model(inputs, return_logits=True)
					raw_loss = _compute_bce_loss(logits, targets, criterion, smoothing=0.1)

				val_loss_sum += raw_loss.detach().item() * batch_size

				probs = torch.sigmoid(logits)
				all_val_scores.append(_tensor_to_numpy(probs))
				all_val_targets.append(_tensor_to_numpy(targets))

		val_loss = val_loss_sum / max(1, num_val_samples)

		if all_val_targets:
			y_true_val = np.concatenate(all_val_targets, axis=0)
			y_score_val = np.concatenate(all_val_scores, axis=0)
			val_auc, val_f1 = _compute_macro_auc_f1(y_true_val, y_score_val, threshold=0.5)
		else:  # pragma: no cover - defensive
			val_auc, val_f1 = float("nan"), float("nan")

		epoch_time = time.time() - epoch_start
		current_lr = optimizer.param_groups[0]["lr"]

		logger.info(
			"Epoch %d/%d - train_loss=%.4f, val_loss=%.4f, val_auc=%.4f, val_f1=%.4f, lr=%.2e, time=%.1fs",
			epoch + 1,
			num_epochs,
			train_loss,
			val_loss,
			val_auc,
			val_f1,
			current_lr,
			epoch_time,
		)
		print(
			f"[epoch {epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
			f"val_auc={val_auc:.4f} val_f1={val_f1:.4f} lr={current_lr:.2e} time={epoch_time:.1f}s"
		)

		_append_training_log(
			CONFIG.LOG_PATH,
			epoch=epoch + 1,
			train_loss=train_loss,
			val_loss=val_loss,
			val_auc=val_auc,
			val_f1=val_f1,
			lr=current_lr,
			time_sec=epoch_time,
		)

		# -------------------- Checkpointing & early stopping --------------------
		if np.isnan(val_auc):
			# If AUC cannot be computed, treat it as 0.0 for the purpose
			# of early-stopping so that we still save at least one model
			# checkpoint from the first epoch.
			val_auc_for_cmp = 0.0
		else:
			val_auc_for_cmp = val_auc

		if val_auc_for_cmp > best_val_auc:
			best_val_auc = val_auc_for_cmp
			best_epoch = epoch + 1
			epochs_no_improve = 0
			try:
				torch.save(model.state_dict(), str(CONFIG.MODEL_SAVE_PATH))
				logger.info("Saved new best model to %s", CONFIG.MODEL_SAVE_PATH)
			except OSError as exc:  # pragma: no cover - IO-dependent
				logger.error("Failed to save model checkpoint to %s: %s", CONFIG.MODEL_SAVE_PATH, exc)
		else:
			epochs_no_improve += 1

		if epochs_no_improve >= CONFIG.PATIENCE:
			logger.info(
				"Early stopping triggered after %d epochs without improvement in validation AUC.",
				CONFIG.PATIENCE,
			)
			break

	total_time = time.time() - global_start_time
	logger.info(
		"Training finished in %.1f minutes. Best epoch=%d, best val AUC=%.4f",
		total_time / 60.0,
		best_epoch,
		best_val_auc,
	)

	# ------------------------------------------------------------------
	# Load best model and run final evaluation
	# ------------------------------------------------------------------
	# As a safety net, ensure we have a checkpoint on disk even if
	# validation AUC was ``NaN`` for all epochs. If no checkpoint is
	# present yet, save the final-epoch weights.
	if not CONFIG.MODEL_SAVE_PATH.exists():  # type: ignore[arg-type]
		try:
			torch.save(model.state_dict(), str(CONFIG.MODEL_SAVE_PATH))
			logger.info(
				"No best-model checkpoint found; saved last-epoch weights to %s",
				CONFIG.MODEL_SAVE_PATH,
			)
		except OSError as exc:  # pragma: no cover - IO-dependent
			logger.error("Failed to save fallback model checkpoint to %s: %s", CONFIG.MODEL_SAVE_PATH, exc)

	try:
		state_dict = torch.load(CONFIG.MODEL_SAVE_PATH, map_location=device)
		model.load_state_dict(state_dict)
		logger.info("Loaded best model from %s (epoch %d)", CONFIG.MODEL_SAVE_PATH, best_epoch)
	except Exception as exc:  # pragma: no cover - IO-dependent
		logger.error(
			"Failed to load model from %s even after fallback saving: %s.",
			CONFIG.MODEL_SAVE_PATH,
			exc,
		)

	# Optimise per-label thresholds on the validation set
	thresholds = find_optimal_threshold(model, val_loader)

	# Final evaluation on the test set (includes test-time augmentation)
	metrics = evaluate(model, test_loader, threshold=thresholds)

	print("\nFinal evaluation on test set:")
	print(
		f"Macro AUC: {metrics['macro_auc']:.4f}\n"
		f"Macro F1: {metrics['macro_f1']:.4f}\n"
		f"Accuracy: {metrics['overall_accuracy']:.4f}\n"
		f"Hamming loss: {metrics['hamming_loss']:.4f}"
	)

	labels = metrics.get("labels", UNIFIED_LABELS)
	per_class_auc = metrics.get("per_class_auc", {})
	per_class_f1 = metrics.get("per_class_f1", {})
	per_class_precision = metrics.get("per_class_precision", {})
	per_class_recall = metrics.get("per_class_recall", {})

	print("\nPer-class metrics:")
	for label in labels:
		auc_val = float(per_class_auc.get(label, float("nan")))
		f1_val = float(per_class_f1.get(label, float("nan")))
		prec_val = float(per_class_precision.get(label, float("nan")))
		rec_val = float(per_class_recall.get(label, float("nan")))
		print(
			f"  {label:>14s} | AUC {auc_val:.4f} | F1 {f1_val:.4f} "
			f"| P {prec_val:.4f} | R {rec_val:.4f}"
		)

	return metrics


if __name__ == "__main__":  # pragma: no cover - manual invocation
	logging.basicConfig(level=logging.INFO)
	train()

