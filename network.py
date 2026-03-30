import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from encoding import (
	INPUT_SIZE, ACTION_TYPE_SIZE, PLAY_START_SIZE, PLAY_END_SIZE, SCOUT_INSERT_SIZE,
	CNN_CHANNELS_V4, HAND_SLOTS_V4, FLAT_SIZE_V4, CNN_FLAT_SIZE_V4, INPUT_SIZE_V4,
	PLAY_START_SIZE_V4, PLAY_END_SIZE_V4, SCOUT_INSERT_SIZE_V4,
	FLAT_ACTION_SIZE,
	GLOBAL_START_V6, GLOBAL_DIM_V6, NUM_ENTITIES_V6,
)

def build_conditioning(hidden: Tensor, action_type_idx: int | None, start_idx: int | None,
					   play_start_size: int) -> Tensor:
	"""Concatenate hidden state with one-hot conditioning vectors for action heads."""
	unbatched = hidden.ndim == 1
	if unbatched:
		hidden = hidden.unsqueeze(0)
	batch_size = hidden.shape[0]
	device = hidden.device
	if action_type_idx is not None and action_type_idx >= 0:
		action_oh = torch.zeros(batch_size, ACTION_TYPE_SIZE, device=device)
		action_oh[:, action_type_idx] = 1.0
	else:
		action_oh = torch.zeros(batch_size, ACTION_TYPE_SIZE, device=device)
	if start_idx is not None and start_idx >= 0:
		start_oh = torch.zeros(batch_size, play_start_size, device=device)
		start_oh[:, start_idx] = 1.0
	else:
		start_oh = torch.zeros(batch_size, play_start_size, device=device)
	conditioned = torch.cat([hidden, action_oh, start_oh], dim=1)
	if unbatched:
		conditioned = conditioned.squeeze(0)
	return conditioned

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
				 hidden_size: int = 128, first_hidden_size: int = 256,
				 play_start_size: int = PLAY_START_SIZE, play_end_size: int = PLAY_END_SIZE,
				 scout_insert_size: int = SCOUT_INSERT_SIZE, encoding_version: int = 1):
		super().__init__()
		# Backward compat: convert old format
		if layer_sizes is None:
			layer_sizes = [first_hidden_size, hidden_size]
		self.layer_sizes = layer_sizes
		self.encoding_version = encoding_version
		self.play_start_size = play_start_size
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
		conditioning_size = ACTION_TYPE_SIZE + play_start_size
		self.value_head = nn.Linear(output_size, 1)
		self.action_type_head = nn.Linear(output_size + conditioning_size, ACTION_TYPE_SIZE)
		self.play_start_head = nn.Linear(output_size + conditioning_size, play_start_size)
		self.play_end_head = nn.Linear(output_size + conditioning_size, play_end_size)
		self.scout_insert_head = nn.Linear(output_size + conditioning_size, scout_insert_size)

	def forward(self, x: Tensor) -> Tensor:
		"""Run input through shared layers, return hidden state."""
		return self.shared(x)

	def value(self, hidden: Tensor) -> Tensor:
		"""Value estimate from hidden state. No conditioning."""
		return self.value_head(hidden)

	def action_type_logits(self, hidden: Tensor) -> Tensor:
		"""Step 1: no prior decisions, conditioning is all zeros."""
		conditioned = build_conditioning(hidden, None, None, self.play_start_size)
		return self.action_type_head(conditioned)

	def play_start_logits(self, hidden: Tensor, action_type: int) -> Tensor:
		"""Step 2: action type known, first_index unknown."""
		conditioned = build_conditioning(hidden, action_type, None, self.play_start_size)
		return self.play_start_head(conditioned)

	def play_end_logits(self, hidden: Tensor, action_type: int, start: int) -> Tensor:
		"""Step 3: both action type and start index known."""
		conditioned = build_conditioning(hidden, action_type, start, self.play_start_size)
		return self.play_end_head(conditioned)

	def scout_insert_logits(self, hidden: Tensor, action_type: int) -> Tensor:
		"""Step 2 alt: action type known, first_index unknown."""
		conditioned = build_conditioning(hidden, action_type, None, self.play_start_size)
		return self.scout_insert_head(conditioned)

class CircularCNNScoutNetwork(nn.Module):
	"""V4 network: circular CNN hand encoder + flat scalars → FC trunk → action heads."""

	def __init__(self, num_filters: int = 32, num_conv_layers: int = 3,
				 layer_sizes: list[int] | None = None,
				 play_start_size: int = PLAY_START_SIZE_V4,
				 play_end_size: int = PLAY_END_SIZE_V4,
				 scout_insert_size: int = SCOUT_INSERT_SIZE_V4,
				 encoding_version: int = 4):
		super().__init__()
		if layer_sizes is None:
			layer_sizes = [512, 256, 256, 128, 128, 128]
		self.layer_sizes = layer_sizes
		self.encoding_version = encoding_version
		self.play_start_size = play_start_size
		self.num_filters = num_filters
		self.num_conv_layers = num_conv_layers

		# Circular CNN for hand
		conv_layers = []
		in_channels = CNN_CHANNELS_V4  # 11
		for i in range(num_conv_layers):
			conv_layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=HAND_SLOTS_V4, bias=True))
			in_channels = num_filters
		self.conv_layers = nn.ModuleList(conv_layers)

		# FC trunk
		cnn_flat_size = num_filters * HAND_SLOTS_V4
		trunk_input = cnn_flat_size + FLAT_SIZE_V4
		layers = []
		prev_size = trunk_input
		for size in layer_sizes:
			if size == prev_size:
				layers.append(ResidualBlock(size))
			else:
				layers.append(nn.Linear(prev_size, size))
				layers.append(nn.ReLU())
			prev_size = size
		self.trunk = nn.Sequential(*layers)

		# Action heads (same interface as ScoutNetwork)
		output_size = layer_sizes[-1]
		conditioning_size = ACTION_TYPE_SIZE + play_start_size
		self.value_head = nn.Linear(output_size, 1)
		self.action_type_head = nn.Linear(output_size + conditioning_size, ACTION_TYPE_SIZE)
		self.play_start_head = nn.Linear(output_size + conditioning_size, play_start_size)
		self.play_end_head = nn.Linear(output_size + conditioning_size, play_end_size)
		self.scout_insert_head = nn.Linear(output_size + conditioning_size, scout_insert_size)

	def _circular_conv(self, x: Tensor) -> Tensor:
		"""Apply circular conv layers. x: (batch, channels, 15)."""
		for conv in self.conv_layers:
			# Circular padding: kernel=15 on length=15 needs 14 wrapped elements
			# so Conv1d sees length 29, producing 29-15+1=15 output positions.
			pad = HAND_SLOTS_V4 - 1  # 14
			x = torch.cat([x[:, :, -pad:], x], dim=2)
			x = F.relu(conv(x))
		return x

	def forward(self, x: Tensor) -> Tensor:
		"""Run packed state through CNN + trunk, return hidden state.
		x: (batch, 279) or (279,) — layout [CNN_flat (165) | flat (114)]."""
		unbatched = x.ndim == 1
		if unbatched:
			x = x.unsqueeze(0)
		# Split and reshape
		cnn_input = x[:, :CNN_FLAT_SIZE_V4].reshape(-1, CNN_CHANNELS_V4, HAND_SLOTS_V4)
		flat_input = x[:, CNN_FLAT_SIZE_V4:]
		# CNN path
		cnn_out = self._circular_conv(cnn_input)  # (batch, F, 15)
		cnn_flat = cnn_out.flatten(1)  # (batch, F*15)
		# Concat and run trunk
		combined = torch.cat([cnn_flat, flat_input], dim=1)
		hidden = self.trunk(combined)
		if unbatched:
			hidden = hidden.squeeze(0)
		return hidden

	def value(self, hidden: Tensor) -> Tensor:
		return self.value_head(hidden)

	def action_type_logits(self, hidden: Tensor) -> Tensor:
		conditioned = build_conditioning(hidden, None, None, self.play_start_size)
		return self.action_type_head(conditioned)

	def play_start_logits(self, hidden: Tensor, action_type: int) -> Tensor:
		conditioned = build_conditioning(hidden, action_type, None, self.play_start_size)
		return self.play_start_head(conditioned)

	def play_end_logits(self, hidden: Tensor, action_type: int, start: int) -> Tensor:
		conditioned = build_conditioning(hidden, action_type, start, self.play_start_size)
		return self.play_end_head(conditioned)

	def scout_insert_logits(self, hidden: Tensor, action_type: int) -> Tensor:
		conditioned = build_conditioning(hidden, action_type, None, self.play_start_size)
		return self.scout_insert_head(conditioned)

class FlatScoutNetwork(nn.Module):
	"""V6 network: optional self-attention over per-card entities → FC trunk → flat policy head."""

	def __init__(self, input_size: int, layer_sizes: list[int] | None = None,
				 encoding_version: int = 6, attention: dict | None = None):
		super().__init__()
		if layer_sizes is None:
			layer_sizes = [512, 256, 256, 128, 128, 128]
		self.layer_sizes = layer_sizes
		self.encoding_version = encoding_version
		self.attention_cfg = attention
		self._use_attention = attention is not None

		if self._use_attention:
			d_model = attention["dim"]
			num_heads = attention["heads"]
			num_layers = attention["layers"]
			assert d_model % num_heads == 0, \
				f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
			# Precomputed indices for gathering each entity's 13 features from flat input
			entity_indices = torch.zeros(NUM_ENTITIES_V6, 13, dtype=torch.long)
			for i in range(16):  # hand entities
				for d in range(10):
					entity_indices[i, d] = i * 12 + d       # one-hot value
				entity_indices[i, 10] = i * 12 + 10          # empty flag
				entity_indices[i, 11] = i * 12 + 11          # top scalar
				entity_indices[i, 12] = 192 + i              # bottom scalar
			for j in range(4):  # scout entities
				for d in range(10):
					entity_indices[16 + j, d] = 208 + j * 11 + d  # one-hot value
				entity_indices[16 + j, 10] = 208 + j * 11 + 10    # absent flag
				entity_indices[16 + j, 11] = 252 + j               # top scalar
				entity_indices[16 + j, 12] = 256 + j               # bottom scalar
			self.register_buffer('entity_indices', entity_indices)
			self.register_buffer('position_onehots',
				torch.eye(NUM_ENTITIES_V6).unsqueeze(0))
			self.entity_proj = nn.Linear(13 + NUM_ENTITIES_V6, d_model)
			self.attn_layers = nn.ModuleList()
			for _ in range(num_layers):
				self.attn_layers.append(nn.ModuleDict({
					'norm': nn.LayerNorm(d_model),
					'attn': nn.MultiheadAttention(d_model, num_heads, batch_first=True),
				}))
			trunk_input = NUM_ENTITIES_V6 * d_model + GLOBAL_DIM_V6
		else:
			trunk_input = input_size

		# FC trunk
		layers = []
		prev_size = trunk_input
		for size in layer_sizes:
			if size == prev_size:
				layers.append(ResidualBlock(size))
			else:
				layers.append(nn.Linear(prev_size, size))
				layers.append(nn.LayerNorm(size))
				layers.append(nn.GELU())
			prev_size = size
		self.shared = nn.Sequential(*layers)

		output_size = layer_sizes[-1]
		self.policy_head = nn.Linear(output_size, FLAT_ACTION_SIZE)

		# Value head: detached multi-scale features (metadata + all trunk layer activations)
		# No policy gradient flows into the value head; it reads trunk features passively
		metadata_dim = GLOBAL_DIM_V6 if self._use_attention else input_size
		value_input_dim = metadata_dim + sum(layer_sizes)
		self.value_head = nn.Sequential(
			nn.Linear(value_input_dim, 256),
			nn.GELU(),
			nn.Linear(256, 1),
		)

	def _run_trunk(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
		"""Run input through trunk, return (final_hidden, per_layer_activations)."""
		h = x
		intermediates = []
		for module in self.shared:
			h = module(h)
			if isinstance(module, (nn.GELU, nn.ReLU, ResidualBlock)):
				intermediates.append(h)
		return h, intermediates

	def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
		"""Returns (hidden, value_context) tuple. Both are passed opaquely to value()/policy_logits()."""
		if self._use_attention:
			unbatched = x.ndim == 1
			if unbatched:
				x = x.unsqueeze(0)
			# Gather per-entity features and concat positional encoding
			entities = x[:, self.entity_indices]  # [batch, 20, 13]
			pos = self.position_onehots.expand(x.shape[0], -1, -1)
			entities = torch.cat([entities, pos], dim=2)  # [batch, 20, 33]
			entities = self.entity_proj(entities)  # [batch, 20, d_model]
			# Pre-norm self-attention with residual connections
			for layer in self.attn_layers:
				residual = entities
				entities = layer['norm'](entities)
				entities, _ = layer['attn'](entities, entities, entities)
				entities = residual + entities
			# Flatten attention output, concat global features, run FC trunk
			combined = torch.cat([entities.flatten(1),
				x[:, GLOBAL_START_V6:]], dim=1)
			hidden, intermediates = self._run_trunk(combined)
			# Value context: raw metadata + detached trunk activations
			metadata = x[:, GLOBAL_START_V6:]
			value_ctx = torch.cat(
				[metadata.detach()] + [a.detach() for a in intermediates], dim=-1)
			if unbatched:
				hidden = hidden.squeeze(0)
				value_ctx = value_ctx.squeeze(0)
			return hidden, value_ctx
		# No-attention path
		hidden, intermediates = self._run_trunk(x)
		value_ctx = torch.cat(
			[x.detach()] + [a.detach() for a in intermediates], dim=-1)
		return hidden, value_ctx

	def value(self, h: Tensor | tuple) -> Tensor:
		_, value_ctx = h
		return self.value_head(value_ctx)

	def policy_logits(self, h: Tensor | tuple) -> Tensor:
		hidden, _ = h
		return self.policy_head(hidden)

class RandomBot:
	"""Bot that plays uniformly random legal actions. For evaluation baseline.
	Duck-types ScoutNetwork: zero logits → uniform distribution after masking."""

	def __init__(self, encoding_version: int = 1,
				 play_start_size: int = PLAY_START_SIZE, play_end_size: int = PLAY_END_SIZE,
				 scout_insert_size: int = SCOUT_INSERT_SIZE):
		self.encoding_version = encoding_version
		self._play_start_size = play_start_size
		self._play_end_size = play_end_size
		self._scout_insert_size = scout_insert_size

	def __call__(self, x: Tensor) -> Tensor:
		return torch.zeros(1)

	def eval(self):
		return self

	def value(self, hidden: Tensor) -> Tensor:
		return torch.tensor(0.0)

	def action_type_logits(self, hidden: Tensor) -> Tensor:
		return torch.zeros(ACTION_TYPE_SIZE)

	def play_start_logits(self, hidden: Tensor, action_type: int) -> Tensor:
		return torch.zeros(self._play_start_size)

	def play_end_logits(self, hidden: Tensor, action_type: int, start: int) -> Tensor:
		return torch.zeros(self._play_end_size)

	def scout_insert_logits(self, hidden: Tensor, action_type: int) -> Tensor:
		return torch.zeros(self._scout_insert_size)

	def policy_logits(self, hidden: Tensor) -> Tensor:
		return torch.zeros(FLAT_ACTION_SIZE)

def batched_masked_sample(logits: Tensor, mask: Tensor) -> Tensor:
	"""Gumbel-max sampling for a batch. logits/mask: [B, C] → [B] LongTensor."""
	u = torch.rand_like(logits).clamp(1e-10, 1.0)
	noisy = logits - torch.log(-torch.log(u))
	noisy = noisy.masked_fill(~mask, float('-inf'))
	return noisy.argmax(dim=1)

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
