print("start", flush=True)
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

print("1. stdlib...", flush=True)
import random
import numpy as np

print("2. torch...", flush=True)
import torch
import torch.nn.functional as F

print("3. matplotlib...", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("4. network...", flush=True)
from network import FlatScoutNetwork, masked_sample

print("5. encoding...", flush=True)
from encoding import (
    INPUT_SIZE_V6, HAND_SLOTS_V6, HAND_DIM_V6, SCOUT_CARDS_DIM_V6,
    GLOBAL_START_V6, GLOBAL_DIM_V6, METADATA_DIM_V6, PLAY_BUFFER_DIM_V6,
    encode_state_v6, get_flat_action_mask, encode_hand_both_orientations_v6,
    decode_flat_action, get_legal_plays,
)

print("6. game...", flush=True)
from game import Game, Phase

print("7. training...", flush=True)
from training import rollout_from_states_batched_v6

print("ALL OK", flush=True)
