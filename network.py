import torch
import torch.nn as nn
from torch import Tensor
from encoding import (
	INPUT_SIZE, ACTION_TYPE_SIZE, PLAY_START_SIZE, PLAY_END_SIZE, SCOUT_INSERT_SIZE,
)

CONDITIONING_SIZE = ACTION_TYPE_SIZE + PLAY_START_SIZE # 29

class ResidualBlock(nn.Module):
	"""Skip connection for same-width consecutive layers."""
	def __init__(self, size: int):
		super().__init__()
		self.linear = nn.Linear(size, size)
		self.relu = nn.ReLU()
	def forward(self, x: Tensor) -> Tensor:
		return self.relu(self.linear(x) + x)

class ScoutNetwork(nn.Module):
	def __init__(self, input_size: int = INPUT_SIZE, layer_sizes: list[int] | None = None,
				 hidden_size: int = 128, first_hidden_size: int = 256):
		super().__init__()
		# Backward compat: convert old format
		if layer_sizes is None:
			layer_sizes = [first_hidden_size, hidden_size]
		self.layer_sizes = layer_sizes
		layers = []
		prev_size = input_size
		for size in layer_sizes:
			if size == prev_size:
				layers.append(ResidualBlock(size))
			else:
				layers.append(nn.Linear(prev_size, size))
				layers.append(nn.ReLU())
			prev_size = size
		self.shared = nn.Sequential(*layers)
		output_size = layer_sizes[-1]
		self.value_head = nn.Linear(output_size, 1)
		self.action_type_head = nn.Linear(output_size + CONDITIONING_SIZE, ACTION_TYPE_SIZE)
		self.play_start_head = nn.Linear(output_size + CONDITIONING_SIZE, PLAY_START_SIZE)
		self.play_end_head = nn.Linear(output_size + CONDITIONING_SIZE, PLAY_END_SIZE)
		self.scout_insert_head = nn.Linear(output_size + CONDITIONING_SIZE, SCOUT_INSERT_SIZE)

	def _build_conditioning(self, hidden: Tensor, action_type_idx: int | None, start_idx: int | None) -> Tensor:
		"""Concatenate hidden state with one-hot conditioning vectors."""
		unbatched = hidden.ndim == 1
		if unbatched:
			hidden = hidden.unsqueeze(0)
		batch_size = hidden.shape[0]
		device = hidden.device
		# Build action type one-hot
		if action_type_idx is not None and action_type_idx >= 0:
			action_oh = torch.zeros(batch_size, ACTION_TYPE_SIZE, device=device)
			action_oh[:, action_type_idx] = 1.0
		else:
			action_oh = torch.zeros(batch_size, ACTION_TYPE_SIZE, device=device)
		# Build start index one-hot
		if start_idx is not None and start_idx >= 0:
			start_oh = torch.zeros(batch_size, PLAY_START_SIZE, device=device)
			start_oh[:, start_idx] = 1.0
		else:
			start_oh = torch.zeros(batch_size, PLAY_START_SIZE, device=device)
		conditioned = torch.cat([hidden, action_oh, start_oh], dim=1)
		if unbatched:
			conditioned = conditioned.squeeze(0)
		return conditioned

	def forward(self, x: Tensor) -> Tensor:
		"""Run input through shared layers, return hidden state."""
		return self.shared(x)

	def value(self, hidden: Tensor) -> Tensor:
		"""Value estimate from hidden state. No conditioning."""
		return self.value_head(hidden)

	def action_type_logits(self, hidden: Tensor) -> Tensor:
		"""Step 1: no prior decisions, conditioning is all zeros."""
		conditioned = self._build_conditioning(hidden, None, None)
		return self.action_type_head(conditioned)

	def play_start_logits(self, hidden: Tensor, action_type: int) -> Tensor:
		"""Step 2: action type known, first_index unknown."""
		conditioned = self._build_conditioning(hidden, action_type, None)
		return self.play_start_head(conditioned)

	def play_end_logits(self, hidden: Tensor, action_type: int, start: int) -> Tensor:
		"""Step 3: both action type and start index known."""
		conditioned = self._build_conditioning(hidden, action_type, start)
		return self.play_end_head(conditioned)

	def scout_insert_logits(self, hidden: Tensor, action_type: int) -> Tensor:
		"""Step 2 alt: action type known, first_index unknown."""
		conditioned = self._build_conditioning(hidden, action_type, None)
		return self.scout_insert_head(conditioned)

class RandomBot:
	"""Bot that plays uniformly random legal actions. For evaluation baseline.
	Duck-types ScoutNetwork: zero logits → uniform distribution after masking."""

	def __call__(self, x: Tensor) -> Tensor:
		return torch.zeros(1)

	def eval(self):
		return self

	def value(self, hidden: Tensor) -> Tensor:
		return torch.tensor(0.0)

	def action_type_logits(self, hidden: Tensor) -> Tensor:
		return torch.zeros(ACTION_TYPE_SIZE)

	def play_start_logits(self, hidden: Tensor, action_type: int) -> Tensor:
		return torch.zeros(PLAY_START_SIZE)

	def play_end_logits(self, hidden: Tensor, action_type: int, start: int) -> Tensor:
		return torch.zeros(PLAY_END_SIZE)

	def scout_insert_logits(self, hidden: Tensor, action_type: int) -> Tensor:
		return torch.zeros(SCOUT_INSERT_SIZE)

def masked_sample(logits: Tensor, mask: Tensor) -> tuple[int, None]:
	"""Sample from masked logits using Gumbel-max trick. Returns (sampled_index, None).
	Log prob is not computed since no caller uses it."""
	u = torch.rand_like(logits).clamp(1e-10, 1.0)
	noisy = logits - torch.log(-torch.log(u))
	noisy = noisy.masked_fill(~mask, float('-inf'))
	return noisy.argmax().item(), None

def masked_log_prob(logits: Tensor, mask: Tensor, action: int) -> Tensor:
	"""Compute log probability of a specific action under masked logits."""
	masked_logits = logits.clone()
	masked_logits[~mask] = float('-inf')
	return torch.log_softmax(masked_logits, dim=-1)[action]
