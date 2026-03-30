"""Profile rollout game performance to identify the dominant cost.

Runs play_games_with_rollouts_v6 on a small number of games under cProfile,
then prints a sorted breakdown of where time is spent.

Usage: python -u profile_rollouts.py [checkpoint_path]
Default checkpoint: bots/v7_3/latest.pt"""

import sys
import os
import cProfile
import pstats
import io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from network import FlatScoutNetwork
from encoding import INPUT_SIZE_V6
from training import play_games_with_rollouts_v6
import torch

NUM_GAMES = 5
NUM_PLAYERS = 4
ROLLOUTS_PER_STATE = 20

def load_network(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    layer_sizes = cfg.get("layer_sizes", [512, 256, 128])
    attention = cfg.get("attention", {})
    net = FlatScoutNetwork(input_size=INPUT_SIZE_V6, layer_sizes=layer_sizes, attention=attention)
    net.load_state_dict(checkpoint["model_state"])
    net.eval()
    return net

def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "bots/v7_3/latest.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    net = load_network(checkpoint_path)

    print(f"Profiling {NUM_GAMES} rollout games ({ROLLOUTS_PER_STATE} rollouts/state, {NUM_PLAYERS} players)...")

    pr = cProfile.Profile()
    pr.enable()
    play_games_with_rollouts_v6(net, NUM_GAMES, NUM_PLAYERS, rollouts_per_state=ROLLOUTS_PER_STATE)
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(40)
    print(buf.getvalue())

    print("\n--- Sort by tottime (self time, no children) ---")
    buf2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=buf2).sort_stats("tottime")
    ps2.print_stats(30)
    print(buf2.getvalue())

if __name__ == "__main__":
    main()
