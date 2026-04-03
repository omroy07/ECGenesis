"""Model definitions for ECG heart disease classification.

This module implements a 1D ResNet with channel-wise Squeeze-and-Excitation
attention and temporal self-attention for multi-label ECG classification.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

try:
	# Local config provides the unified label set
	from .config import UNIFIED_LABELS
except Exception:  # pragma: no cover - fallback if config is unavailable
	UNIFIED_LABELS = [
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


class SEBlock1D(nn.Module):
	"""Squeeze-and-Excitation (SE) block for 1D feature maps.

	This block applies channel-wise attention using global average pooling
	followed by a small two-layer fully connected network and a sigmoid
	activation to re-weight the input feature channels.
	"""

	def __init__(self, channels: int, reduction: int = 16) -> None:
		super().__init__()
		hidden_dim = max(1, channels // reduction)
		self.avg_pool = nn.AdaptiveAvgPool1d(1)
		self.fc = nn.Sequential(
			nn.Linear(channels, hidden_dim, bias=True),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, channels, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, x: Tensor) -> Tensor:
		"""Forward pass.

		Args:
			x: Input tensor of shape (batch, channels, length).

		Returns:
			Tensor of the same shape as ``x`` with re-weighted channels.
		"""

		b, c, _ = x.shape
		# Global average pooling over temporal dimension
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1)
		return x * y


class ResidualBlock1D(nn.Module):
	"""1D residual block with SE attention.

	Architecture:
	- Conv1d → BatchNorm1d → ReLU → Dropout(0.2)
	- Conv1d → BatchNorm1d
	- SEBlock1D for channel attention
	- Skip connection (with optional 1×1 Conv for shape matching)
	- Final ReLU activation
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		stride: int = 1,
		reduction: int = 16,
		dropout: float = 0.2,
	) -> None:
		super().__init__()

		self.conv1 = nn.Conv1d(
			in_channels,
			out_channels,
			kernel_size=3,
			stride=stride,
			padding=1,
			bias=False,
		)
		self.bn1 = nn.BatchNorm1d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.dropout = nn.Dropout(dropout)

		self.conv2 = nn.Conv1d(
			out_channels,
			out_channels,
			kernel_size=3,
			stride=1,
			padding=1,
			bias=False,
		)
		self.bn2 = nn.BatchNorm1d(out_channels)

		# Channel-wise attention
		self.se = SEBlock1D(out_channels, reduction=reduction)

		# Optional projection for the skip connection
		if stride != 1 or in_channels != out_channels:
			self.downsample: Optional[nn.Module] = nn.Sequential(
				nn.Conv1d(
					in_channels,
					out_channels,
					kernel_size=1,
					stride=stride,
					bias=False,
				),
				nn.BatchNorm1d(out_channels),
			)
		else:
			self.downsample = None

	def forward(self, x: Tensor) -> Tensor:
		"""Forward pass for the residual block.

		Args:
			x: Input tensor of shape (batch, in_channels, length).

		Returns:
			Tensor of shape (batch, out_channels, new_length).
		"""

		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out = self.se(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out = out + identity
		out = self.relu(out)
		return out


class ECGResNet(nn.Module):
	"""ResNet-style 1D CNN with SE and temporal attention for ECG.

	Overall architecture (input: (batch, 12, 5000)):

	- Stem: Conv1d(12, 64, kernel=15, stride=2, padding=7)
	  → BatchNorm1d → ReLU → MaxPool1d(kernel=3, stride=2)
	- Stage 1: 2 × ResidualBlock1D(64, 64,  stride=1)
	- Stage 2: 2 × ResidualBlock1D(64, 128, stride=2)
	- Stage 3: 2 × ResidualBlock1D(128, 256, stride=2)
	- Stage 4: 2 × ResidualBlock1D(256, 512, stride=2)

	After the convolutional backbone, a temporal self-attention layer operates
	over the sequence dimension, followed by dual pooling (adaptive average
	and max pooling) and a fully connected classifier head with dropout.
	"""

	def __init__(self, num_classes: Optional[int] = None) -> None:
		super().__init__()

		if num_classes is None:
			num_classes = len(UNIFIED_LABELS)

		self.num_classes = num_classes

		# Stem
		self.conv1 = nn.Conv1d(
			in_channels=12,
			out_channels=64,
			kernel_size=15,
			stride=2,
			padding=7,
			bias=False,
		)
		self.bn1 = nn.BatchNorm1d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

		# ResNet stages
		self.stage1 = self._make_stage(64, 64, blocks=2, stride=1)
		self.stage2 = self._make_stage(64, 128, blocks=2, stride=2)
		self.stage3 = self._make_stage(128, 256, blocks=2, stride=2)
		self.stage4 = self._make_stage(256, 512, blocks=2, stride=2)

		# Temporal self-attention over sequence dimension
		self.attention_norm = nn.LayerNorm(512)
		self.self_attention = nn.MultiheadAttention(
			embed_dim=512,
			num_heads=8,
			batch_first=True,
		)

		# Dual pooling
		self.avg_pool = nn.AdaptiveAvgPool1d(1)
		self.max_pool = nn.AdaptiveMaxPool1d(1)

		# Classifier head: 1024 -> 256 -> num_classes
		self.fc1 = nn.Linear(512 * 2, 256)
		self.fc1_bn = nn.BatchNorm1d(256)
		self.fc1_drop = nn.Dropout(0.5)
		self.fc2 = nn.Linear(256, num_classes)

	@staticmethod
	def _make_stage(
		in_channels: int,
		out_channels: int,
		blocks: int,
		stride: int,
	) -> nn.Sequential:
		"""Create a ResNet stage consisting of multiple residual blocks.

		The first block may change the number of channels and/or apply
		downsampling via ``stride``; subsequent blocks keep ``out_channels``
		and use stride 1.
		"""

		layers = [ResidualBlock1D(in_channels, out_channels, stride=stride)]
		for _ in range(1, blocks):
			layers.append(ResidualBlock1D(out_channels, out_channels, stride=1))
		return nn.Sequential(*layers)

	def forward(self, x: Tensor, return_logits: bool = False) -> Tensor:
		"""Forward pass through the ECGResNet.

		Args:
			x: Input tensor of shape (batch, 12, 5000).
			return_logits: If ``True``, return raw logits before sigmoid.

		Returns:
			Tensor of shape (batch, num_classes). If ``return_logits`` is
			``False`` (default), the output is passed through a sigmoid
			activation and can be interpreted as per-class probabilities.
		"""

		# Stem
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		# ResNet backbone
		x = self.stage1(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)

		# x shape: (batch, channels=512, length)
		# Temporal self-attention expects (batch, length, channels)
		x_seq = x.permute(0, 2, 1)  # (B, L, C)
		x_seq = self.attention_norm(x_seq)
		attn_out, _ = self.self_attention(x_seq, x_seq, x_seq)

		# Back to (B, C, L)
		x_attn = attn_out.permute(0, 2, 1)

		# Dual pooling: adaptive average and max pooling
		avg = self.avg_pool(x_attn).squeeze(-1)
		max_ = self.max_pool(x_attn).squeeze(-1)
		feats = torch.cat([avg, max_], dim=1)  # (B, 1024)

		# Classifier head
		feats = self.fc1(feats)
		feats = self.fc1_bn(feats)
		feats = self.relu(feats)
		feats = self.fc1_drop(feats)
		logits = self.fc2(feats)

		if return_logits:
			return logits
		return torch.sigmoid(logits)


def build_ecg_resnet(num_classes: Optional[int] = None) -> ECGResNet:
	"""Factory function to create an ECGResNet model.

	Args:
		num_classes: Number of output labels. If ``None``, uses the length
			of ``UNIFIED_LABELS`` from the configuration.

	Returns:
		An instance of :class:`ECGResNet`.
	"""

	return ECGResNet(num_classes=num_classes)

