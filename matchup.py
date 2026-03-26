import random
import os
import torch
from encoding import (
	INPUT_SIZE, INPUT_SIZE_V2, INPUT_SIZE_V6,
	PLAY_START_SIZE_V2, PLAY_END_SIZE_V2, SCOUT_INSERT_SIZE_V2,
)
from network import ScoutNetwork, FlatScoutNetwork, RandomBot
from training import play_eval_game

class Agent:
	"""Wrapper for a network-like object with a display name."""
	def __init__(self, name: str, network):
		self.name = name
		self.network = network

def load_agent(spec: str) -> Agent:
	"""Load an agent from a spec string.
	'random' → RandomBot, '*.pt' → load checkpoint."""
	if spec == "random":
		return Agent("random", RandomBot())
	if spec.endswith(".pt"):
		if not os.path.exists(spec):
			raise FileNotFoundError(f"Checkpoint not found: {spec}")
		checkpoint = torch.load(spec, weights_only=False)
		cfg = checkpoint.get("config", {})
		ev = cfg.get("encoding_version", 1)
		if "layer_sizes" in cfg:
			ls = cfg["layer_sizes"]
		else:
			ls = [cfg.get("first_hidden_size", 256), cfg.get("hidden_size", 128)]
		if ev == 6:
			network = FlatScoutNetwork(INPUT_SIZE_V6, ls, encoding_version=6)
		elif ev == 2:
			network = ScoutNetwork(INPUT_SIZE_V2, ls,
				play_start_size=PLAY_START_SIZE_V2, play_end_size=PLAY_END_SIZE_V2,
				scout_insert_size=SCOUT_INSERT_SIZE_V2, encoding_version=2)
		else:
			network = ScoutNetwork(input_size=INPUT_SIZE, layer_sizes=ls)
		network.load_state_dict(checkpoint["model_state"])
		network.eval()
		return Agent(os.path.basename(spec), network)
	raise ValueError(f"Unknown agent spec: {spec!r} (use 'random' or a .pt path)")

def run_matchup(agents: list[Agent], num_games: int):
	"""Play games between agents, shuffling seats each game for fairness."""
	num_players = len(agents)
	# Build display names (disambiguate duplicates)
	name_counts: dict[str, int] = {}
	for a in agents:
		name_counts[a.name] = name_counts.get(a.name, 0) + 1
	seen: dict[str, int] = {}
	display_names = []
	for a in agents:
		if name_counts[a.name] > 1:
			seen[a.name] = seen.get(a.name, 0) + 1
			display_names.append(f"{a.name} ({seen[a.name]})")
		else:
			display_names.append(a.name)

	wins = [0] * num_players
	total_score = [0] * num_players

	print(f"Matchup: {num_games} games, {num_players} players")
	for i, name in enumerate(display_names):
		print(f"  Agent {i}: {name}")
	print()

	indices = list(range(num_players))
	for g in range(num_games):
		# Shuffle seating for fairness
		random.shuffle(indices)
		networks = [agents[indices[seat]].network for seat in range(num_players)]
		scores = play_eval_game(networks, num_players)
		# Map scores back to agents
		max_score = max(scores)
		for seat in range(num_players):
			agent_idx = indices[seat]
			total_score[agent_idx] += scores[seat]
			if scores[seat] == max_score:
				wins[agent_idx] += 1
		# Progress
		if (g + 1) % 10 == 0 or g == num_games - 1:
			n = g + 1
			parts = [f"{display_names[i]}={wins[i]}" for i in range(num_players)]
			print(f"  [{n}/{num_games}] " + "  ".join(parts))

	# Final results
	print(f"\n{'Agent':<20} {'Wins':>6} {'Win%':>7} {'Avg Score':>10}")
	print("-" * 45)
	order = sorted(range(num_players), key=lambda i: wins[i], reverse=True)
	for i in order:
		wr = wins[i] / num_games * 100
		avg = total_score[i] / num_games
		print(f"{display_names[i]:<20} {wins[i]:>6} {wr:>6.1f}% {avg:>10.1f}")
