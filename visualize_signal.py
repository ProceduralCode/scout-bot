"""Visualize the rollout signal pipeline for scout actions.

Shows raw data without assumptions about what's "good" or "bad."

1. ALL POSITIONS: For scoutable states, rollout value at every insert position.
   Shows the hand with showing values so you can see adjacencies (sets AND runs).
   Charts value by position at different rollout counts.

2. FULL GAME WALKTHROUGH: Every action in a complete game with rollout advantages.
   Chart of raw advantages colored by action type.

Usage: python scout-bot/visualize_signal.py [--checkpoint PATH] [--seed N] [--states N]
"""
import sys
import os
import copy
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from game import Game, Phase
from display import format_hand, format_card, format_play, format_showing_values
from encoding import (
    encode_state, encode_state_v2, encode_hand_both_orientations, encode_hand_both_orientations_v2,
    get_legal_plays, get_action_type_mask,
    get_play_start_mask, get_play_end_mask, get_scout_insert_mask,
    decode_action_type, decode_slot_to_hand_index,
    INPUT_SIZE, HAND_SLOTS, PLAY_SLOTS, SCOUT_INSERT_SIZE,
    PLAY_START_SIZE, PLAY_END_SIZE,
    INPUT_SIZE_V2, HAND_SLOTS_V2, SCOUT_INSERT_SIZE_V2,
    PLAY_START_SIZE_V2, PLAY_END_SIZE_V2,
)
from network import ScoutNetwork, masked_sample, batched_masked_sample
from training import rollout_from_states_batched

NUM_PLAYERS = 4


def load_checkpoint(path):
    checkpoint = torch.load(path, weights_only=False)
    cfg = checkpoint.get("config", {})
    layer_sizes = cfg.get("layer_sizes", [512, 256, 256, 128, 128, 128])
    ev = cfg.get("encoding_version", 1)
    if ev == 2:
        network = ScoutNetwork(INPUT_SIZE_V2, layer_sizes,
            play_start_size=PLAY_START_SIZE_V2, play_end_size=PLAY_END_SIZE_V2,
            scout_insert_size=SCOUT_INSERT_SIZE_V2, encoding_version=2)
    elif ev == 1:
        network = ScoutNetwork(INPUT_SIZE, layer_sizes)
    else:
        raise ValueError(f"Unsupported encoding version {ev}")
    network.load_state_dict(checkpoint["model_state"])
    network.eval()
    return network, ev


def compute_margins(scores_list, player, num_players):
    """Convert raw round scores to margins for a specific player."""
    margins = []
    for scores in scores_list:
        opp = [scores[j] for j in range(num_players) if j != player]
        margin = (scores[player] - sum(opp) / len(opp)) / 10.0
        margins.append(margin)
    return margins


def find_scoutable_state(max_attempts=200):
    """Find a game state where scouting is possible.
    Returns (game, player) or None."""
    for _ in range(max_attempts):
        game = Game(NUM_PLAYERS)
        game.start_round()
        for p in range(NUM_PLAYERS):
            game.submit_flip_decision(p, do_flip=random.random() < 0.5)
        # Play a few turns to get a mid-game state
        turns = random.randint(1, 8)
        for _ in range(turns):
            if game.phase != Phase.TURN:
                break
            p = game.current_player
            hand = game.players[p].hand
            legal_plays = get_legal_plays(hand, game.current_play)
            if legal_plays:
                start, end = random.choice(legal_plays)
                game.apply_play(start, end)
            elif game.current_play and game.current_play.cards:
                pos = random.randint(0, len(hand))
                game.apply_scout(True, False, pos)
        if game.phase == Phase.TURN and game.current_play is not None:
            return game, game.current_player
    return None


def val_char(v):
    return "T" if v == 10 else str(v)


def rollout_all_positions(game, player, network, n_rollouts):
    """Run rollouts for every possible insert position.
    Returns list of (position, mean_margin, se, individual_margins)."""
    hand = game.players[player].hand
    card = game.current_play.cards[0]  # left end
    num_positions = len(hand) + 1

    # Build all post-scout states
    position_games = []
    for pos in range(num_positions):
        g = copy.deepcopy(game)
        g.apply_scout(True, False, pos)
        position_games.append(g)

    # Run all rollouts in one batch
    expanded = [g for g in position_games for _ in range(n_rollouts)]
    all_scores = rollout_from_states_batched(expanded, network)

    results = []
    for pos in range(num_positions):
        base = pos * n_rollouts
        pos_scores = all_scores[base:base + n_rollouts]
        margins = compute_margins(pos_scores, player, NUM_PLAYERS)
        mean = sum(margins) / len(margins)
        se = (sum((x - mean) ** 2 for x in margins) / len(margins)) ** 0.5 / len(margins) ** 0.5
        results.append((pos, mean, se, margins))

    return results


def section_1_all_positions(network, num_states=3, output_dir="scout-bot"):
    """For each scoutable state, show rollout value at every position."""
    print("=" * 70)
    print("SECTION 1: ROLLOUT VALUE BY INSERT POSITION")
    print("=" * 70)

    for state_idx in range(num_states):
        result = find_scoutable_state()
        if result is None:
            print(f"\n  State {state_idx + 1}: Could not find scoutable state")
            continue
        game, player = result
        hand = game.players[player].hand
        card = game.current_play.cards[0]  # left end
        card_val = card[0]
        num_positions = len(hand) + 1

        print(f"\n--- State {state_idx + 1} ---")
        print(f"  Player {player}'s hand: {format_hand(hand)}")
        print(f"  Showing values:      {format_showing_values(hand)}")
        print(f"  Card to scout:       {format_card(card)} (showing: {val_char(card_val)})")
        if game.current_play:
            print(f"  Current play:        {format_play(game.current_play)}")

        # Position labels showing what's to the left and right
        print(f"\n  Position map (card {val_char(card_val)} inserted between):")
        for pos in range(num_positions):
            left_val = val_char(hand[pos - 1][0]) if pos > 0 else "_"
            right_val = val_char(hand[pos][0]) if pos < len(hand) else "_"
            print(f"    pos {pos:2d}: {left_val} [{val_char(card_val)}] {right_val}")

        # Run rollouts at different counts
        rollout_counts = [5, 10, 20, 50]
        max_rollouts = max(rollout_counts)
        print(f"\n  Running {max_rollouts} rollouts per position...")
        full_results = rollout_all_positions(game, player, network, max_rollouts)

        # Print table at each rollout count
        for n in rollout_counts:
            print(f"\n  {n} rollouts per position:")
            print(f"    {'Pos':>4}  {'Left':>5}  [card]  {'Right':>5}  {'Mean':>9}  {'SE':>8}")
            for pos, _, _, margins in full_results:
                sub = margins[:n]
                mean = sum(sub) / n
                se = (sum((x - mean) ** 2 for x in sub) / n) ** 0.5 / n ** 0.5
                left_val = val_char(hand[pos - 1][0]) if pos > 0 else "_"
                right_val = val_char(hand[pos][0]) if pos < len(hand) else "_"
                bar = "#" * max(0, int((mean + 1) * 20))  # rough visual
                print(f"    {pos:>4}  {left_val:>5}  [{val_char(card_val)}]  {right_val:>5}  {mean:>+9.4f}  {se:>8.4f}  {bar}")

        # Chart: bar chart of mean value by position at 50 rollouts
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"State {state_idx + 1}: Scouting {val_char(card_val)} into hand [{format_showing_values(hand)}]",
            fontsize=13
        )

        for ax_idx, n in enumerate(rollout_counts):
            ax = axes[ax_idx // 2][ax_idx % 2]
            means = []
            ses = []
            for pos, _, _, margins in full_results:
                sub = margins[:n]
                m = sum(sub) / n
                s = (sum((x - m) ** 2 for x in sub) / n) ** 0.5 / n ** 0.5
                means.append(m)
                ses.append(s)

            positions = list(range(num_positions))
            # X-axis labels: show what's on each side of the insert position
            xlabels = []
            for pos in positions:
                left = val_char(hand[pos - 1][0]) if pos > 0 else "_"
                right = val_char(hand[pos][0]) if pos < len(hand) else "_"
                xlabels.append(f"{left}|{right}")

            bars = ax.bar(positions, means, yerr=ses, capsize=3,
                         color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xticks(positions)
            ax.set_xticklabels(xlabels, fontsize=8, rotation=45 if num_positions > 8 else 0)
            ax.set_xlabel(f"Insert position (left|right neighbor showing values)")
            ax.set_ylabel("Mean margin (rollout)")
            ax.set_title(f"{n} rollouts/position")
            ax.axhline(y=sum(means) / len(means), color='red', linestyle='--', alpha=0.5, label='mean')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, f"signal_positions_state{state_idx + 1}.png")
        plt.savefig(chart_path, dpi=120)
        plt.close()
        print(f"\n  Chart saved: {chart_path}")


def section_2_full_game(network, output_dir="scout-bot"):
    """Play a full game and show advantages for every action."""
    print("\n\n" + "=" * 70)
    print("SECTION 2: FULL GAME WALKTHROUGH")
    print("=" * 70)

    ev = getattr(network, 'encoding_version', 1)
    v2 = ev == 2
    _hs = HAND_SLOTS_V2 if v2 else HAND_SLOTS
    _sis = SCOUT_INSERT_SIZE_V2 if v2 else SCOUT_INSERT_SIZE
    _pss = PLAY_START_SIZE_V2 if v2 else PLAY_START_SIZE

    game = Game(NUM_PLAYERS)
    if v2:
        game.starting_player = random.randint(0, NUM_PLAYERS - 1)
        game.total_rounds = 1
    game.start_round()

    # Flip decisions
    for p in range(NUM_PLAYERS):
        if v2:
            ho = random.randint(0, HAND_SLOTS_V2 - 1)
            t_n, t_f = encode_hand_both_orientations_v2(game, p, ho)
        else:
            ho = random.randint(0, HAND_SLOTS - 1)
            po = random.randint(0, PLAY_SLOTS - 1)
            t_n, t_f = encode_hand_both_orientations(game, p, ho, po)
        with torch.no_grad():
            h_n = network(t_n)
            h_f = network(t_f)
            v_n = network.value(h_n).item()
            v_f = network.value(h_f).item()
        game.submit_flip_decision(p, do_flip=v_f > v_n)

    print(f"\n  Starting hands:")
    for p in range(NUM_PLAYERS):
        print(f"    P{p}: {format_hand(game.players[p].hand)}  (showing: {format_showing_values(game.players[p].hand)})")

    # Play through, recording actions and snapshots
    actions = []  # (player, action_type_str, description, snap_before_idx, nn_value)
    snapshots = [copy.deepcopy(game)]
    rollouts_per = 20

    action_num = 0
    while game.phase in (Phase.TURN, Phase.SNS_PLAY):
        p = game.current_player
        hand = game.players[p].hand
        legal_plays = get_legal_plays(hand, game.current_play)
        ho = random.randint(0, _hs - 1)

        if v2:
            state = encode_state_v2(game, p, ho)
        else:
            po = random.randint(0, PLAY_SLOTS - 1)
            state = encode_state(game, p, ho, po)

        with torch.no_grad():
            hidden = network(state)
            value = network.value(hidden).item()
            at_logits = network.action_type_logits(hidden)
            at_mask_np = get_action_type_mask(game, legal_plays, max_hand=_hs)
            if not at_mask_np.any():
                game._advance_turn()
                continue
            at_mask = torch.from_numpy(at_mask_np)
            action_type, _ = masked_sample(at_logits, at_mask)
            action_info = decode_action_type(action_type)

            if action_info["type"] == "play":
                ps_logits = network.play_start_logits(hidden, action_type)
                ps_mask_np = get_play_start_mask(legal_plays, ho, num_slots=_hs)
                ps_mask = torch.from_numpy(ps_mask_np)
                start_slot, _ = masked_sample(ps_logits, ps_mask)
                start_idx = decode_slot_to_hand_index(start_slot, ho, num_slots=_hs)
                pe_logits = network.play_end_logits(hidden, action_type, start_slot)
                pe_mask_np = get_play_end_mask(legal_plays, start_idx, ho, num_slots=_hs)
                pe_mask = torch.from_numpy(pe_mask_np)
                end_slot, _ = masked_sample(pe_logits, pe_mask)
                end_idx = decode_slot_to_hand_index(end_slot, ho, num_slots=_hs)
                cards_played = hand[start_idx:end_idx + 1]
                play_len = end_idx - start_idx + 1
                desc = f"plays {format_hand(cards_played)} ({play_len})"
                game.apply_play(start_idx, end_idx)
                actions.append((p, "PLAY", desc, len(snapshots) - 1, value))

            elif action_info["type"] == "scout":
                si_logits = network.scout_insert_logits(hidden, action_type)
                si_mask_np = get_scout_insert_mask(game, ho, num_slots=_sis)
                si_mask = torch.from_numpy(si_mask_np)
                insert_slot, _ = masked_sample(si_logits, si_mask)
                insert_pos = (insert_slot - ho) % _sis
                card = game.current_play.cards[0 if action_info["left_end"] else -1]
                if action_info["flip"]:
                    card = (card[1], card[0])
                end_str = "L" if action_info["left_end"] else "R"
                flip_str = "F" if action_info["flip"] else ""
                desc = f"scouts {format_card(card)}({end_str}{flip_str}) -> pos {insert_pos}"
                game.apply_scout(action_info["left_end"], action_info["flip"], insert_pos)
                actions.append((p, "SCOUT", desc, len(snapshots) - 1, value))

            elif action_info["type"] == "sns":
                si_logits = network.scout_insert_logits(hidden, action_type)
                si_mask_np = get_scout_insert_mask(game, ho, num_slots=_sis)
                si_mask = torch.from_numpy(si_mask_np)
                insert_slot, _ = masked_sample(si_logits, si_mask)
                insert_pos = (insert_slot - ho) % _sis
                card_info = game.current_play.cards[0 if action_info["left_end"] else -1]
                g_test = copy.deepcopy(game)
                try:
                    g_test.apply_sns_scout(action_info["left_end"], action_info["flip"], insert_pos)
                except (AssertionError, AssertionError):
                    desc = f"scouts {format_card(card_info)} -> pos {insert_pos} (S&S->scout)"
                    game.apply_scout(action_info["left_end"], action_info["flip"], insert_pos)
                    actions.append((p, "SCOUT", desc, len(snapshots) - 1, value))
                else:
                    desc = f"S&S scout {format_card(card_info)} -> pos {insert_pos}"
                    game.apply_sns_scout(action_info["left_end"], action_info["flip"], insert_pos)
                    actions.append((p, "S&S", desc, len(snapshots) - 1, value))
                    hand2 = game.players[p].hand
                    legal2 = get_legal_plays(hand2, game.current_play)
                    if legal2:
                        s2, e2 = random.choice(legal2)
                        cards2 = hand2[s2:e2 + 1]
                        desc2 = f"S&S plays {format_hand(cards2)} ({e2 - s2 + 1})"
                        game.apply_play(s2, e2)
                        actions.append((p, "S&S-PLAY", desc2, len(snapshots) - 1, value))

        snapshots.append(copy.deepcopy(game))
        action_num += 1
        if action_num > 100:
            break

    # Run rollouts from all snapshots
    print(f"\n  {len(actions)} actions, {len(snapshots)} snapshots")
    print(f"  Running {rollouts_per} rollouts per snapshot...")
    expanded = [snap for snap in snapshots for _ in range(rollouts_per)]
    all_scores = rollout_from_states_batched(expanded, network)

    # Per-snapshot values
    snapshot_values = []
    for si in range(len(snapshots)):
        player_margins = [0.0] * NUM_PLAYERS
        base = si * rollouts_per
        for r in range(rollouts_per):
            scores = all_scores[base + r]
            for p in range(NUM_PLAYERS):
                opp = [scores[j] for j in range(NUM_PLAYERS) if j != p]
                margin = (scores[p] - sum(opp) / len(opp)) / 10.0
                player_margins[p] += margin
        snapshot_values.append([m / rollouts_per for m in player_margins])

    # Raw advantages
    raw_advantages = []
    for i, (p, atype, desc, snap_before, nn_value) in enumerate(actions):
        v_before = snapshot_values[snap_before][p]
        v_after = snapshot_values[snap_before + 1][p]
        raw_advantages.append(v_after - v_before)

    # Normalize
    mean_adv = sum(raw_advantages) / len(raw_advantages) if raw_advantages else 0
    std_adv = (sum((a - mean_adv) ** 2 for a in raw_advantages) / len(raw_advantages)) ** 0.5 if raw_advantages else 1
    norm_advantages = [(a - mean_adv) / (std_adv + 1e-8) for a in raw_advantages]

    # Print table
    print(f"\n  {'#':>3}  {'P':>1}  {'Type':<10}  {'Description':<40}  {'V_before':>9}  {'V_after':>9}  {'Raw Adv':>9}  {'Norm Adv':>9}")
    print(f"  {'---':>3}  {'-':>1}  {'----':<10}  {'-'*40}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

    for i, (p, atype, desc, snap_before, nn_value) in enumerate(actions):
        v_before = snapshot_values[snap_before][p]
        v_after = snapshot_values[snap_before + 1][p]
        print(f"  {i+1:>3}  {p:>1}  {atype:<10}  {desc:<40}  {v_before:>+9.4f}  {v_after:>+9.4f}  {raw_advantages[i]:>+9.4f}  {norm_advantages[i]:>+9.4f}")

    # Summary stats
    play_raw = [raw_advantages[i] for i, (_, t, _, _, _) in enumerate(actions) if t == "PLAY"]
    scout_raw = [raw_advantages[i] for i, (_, t, _, _, _) in enumerate(actions) if "SCOUT" in t or "S&S" in t]
    print(f"\n  Play actions:  {len(play_raw)}", end="")
    if play_raw:
        print(f"  mean={np.mean(play_raw):+.4f}  std={np.std(play_raw):.4f}  range=[{min(play_raw):+.4f}, {max(play_raw):+.4f}]")
    else:
        print()
    print(f"  Scout actions: {len(scout_raw)}", end="")
    if scout_raw:
        print(f"  mean={np.mean(scout_raw):+.4f}  std={np.std(scout_raw):.4f}  range=[{min(scout_raw):+.4f}, {max(scout_raw):+.4f}]")
    else:
        print()
    print(f"  Normalization: shift={-mean_adv:+.4f}  scale=1/{std_adv:.4f}")

    # Chart: advantages by action, colored by type
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle("Full Game: Advantages by Action", fontsize=13)

    x = list(range(len(actions)))
    colors = []
    for _, atype, _, _, _ in actions:
        if atype == "PLAY":
            colors.append("steelblue")
        elif "S&S" in atype:
            colors.append("orange")
        else:
            colors.append("crimson")

    # Raw advantages
    ax1.bar(x, raw_advantages, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axhline(y=mean_adv, color='gray', linestyle='--', alpha=0.5, label=f'mean ({mean_adv:+.3f})')
    ax1.set_ylabel("Raw advantage (V_after - V_before)")
    ax1.set_title("Raw advantages")
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    # Label action types on x
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{a[1][0]}" for a in actions], fontsize=7)

    # Normalized advantages
    ax2.bar(x, norm_advantages, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel("Normalized advantage")
    ax2.set_xlabel("Action # (P=play, S=scout)")
    ax2.set_title("After normalization (what PPO sees)")
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{a[1][0]}" for a in actions], fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='Play'),
                       Patch(facecolor='crimson', label='Scout'),
                       Patch(facecolor='orange', label='S&S')]
    ax1.legend(handles=legend_elements, fontsize=8, loc='upper right')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "signal_game_advantages.png")
    plt.savefig(chart_path, dpi=120)
    plt.close()
    print(f"\n  Chart saved: {chart_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="v4_2/latest.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--states", type=int, default=3, help="Number of states for section 1")
    parser.add_argument("--section", type=int, nargs="*", default=[1, 2],
                        help="Which sections to run (1, 2)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Checkpoint: {args.checkpoint}")
    network, ev = load_checkpoint(args.checkpoint)
    print(f"Encoding: v{ev}\n")

    if 1 in args.section:
        section_1_all_positions(network, num_states=args.states)
    if 2 in args.section:
        section_2_full_game(network)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
