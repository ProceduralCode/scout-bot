"""When a triple is legal, what probability does the v7_12 policy assign to it?"""
import os, sys, random, torch
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from game import Game, Phase
from encoding import get_legal_plays, get_flat_action_mask, encode_state_v6, HAND_SLOTS_V6
from network import FlatScoutNetwork

def load_network(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    cfg = ckpt.get("config", {})
    net = FlatScoutNetwork(
        input_size=cfg.get("input_size", 309),
        layer_sizes=cfg.get("layer_sizes"),
        encoding_version=cfg.get("encoding_version", 6),
        attention=cfg.get("attention"),
    )
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net

def analyze(ckpt_path, n_games=5000, num_players=3):
    net = load_network(ckpt_path)
    H = HAND_SLOTS_V6  # 16

    # Stats for states where triples are legal
    triple_states = 0
    p_triple_sum = 0.0  # probability mass on triples
    p_single_sum = 0.0
    p_pair_sum = 0.0
    p_scout_sum = 0.0
    chose_triple = 0  # argmax is a triple
    chose_pair = 0
    chose_single = 0
    chose_scout = 0
    # Breakdown: when NOT choosing triple, what's the situation?
    non_triple_has_current_play = 0  # had to beat something
    non_triple_opening = 0  # no current play (opening move)

    # Stats for ALL play states (when pair is legal, for comparison with diag)
    pair_states = 0
    pair_p3plus_sum = 0.0

    for _ in range(n_games):
        g = Game(num_players)
        g.start_round()
        for p in range(num_players):
            g.submit_flip_decision(p, random.random() < 0.5)

        hand_offsets = [0] * num_players

        try:
            for _ in range(200):
                if g.phase == Phase.GAME_OVER:
                    break
                if g.phase == Phase.FLIP_DECISION:
                    for p in sorted(g.flips_remaining):
                        g.submit_flip_decision(p, random.random() < 0.5)
                    continue
                if g.phase not in (Phase.TURN, Phase.SNS_PLAY):
                    break

                cp = g.current_player
                hand = g.players[cp].hand

                if g.phase == Phase.SNS_PLAY:
                    legal = get_legal_plays(hand, g.current_play)
                    if legal:
                        s, e = random.choice(legal)
                        g.apply_play(s, e)
                    else:
                        break
                    continue

                # Phase.TURN
                legal = get_legal_plays(hand, g.current_play)

                if not legal:
                    if g.current_play is not None and len(hand) < 16:
                        g.apply_scout(random.random() < 0.5, random.random() < 0.5,
                                     random.randint(0, len(hand)))
                    else:
                        break
                    continue

                # Categorize legal plays by length
                by_len = {}
                for s, e in legal:
                    l = e - s + 1
                    by_len.setdefault(l, []).append((s, e))

                has_triple = any(l >= 3 for l in by_len)
                has_pair = any(l >= 2 for l in by_len)

                if has_triple or has_pair:
                    # Get network's policy
                    ho = hand_offsets[cp]
                    mask = get_flat_action_mask(g, cp, legal, ho)
                    state = encode_state_v6(g, cp, ho)
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        hidden = net(state_t)
                        logits = net.policy_logits(hidden).squeeze(0)

                    mask_bool = mask.bool()
                    masked_logits = logits.clone()
                    masked_logits[~mask_bool] = float('-inf')
                    probs = torch.softmax(masked_logits, dim=0)

                    # Sum probability by play length
                    p_by_len = {}
                    for a in range(256):
                        if mask_bool[a]:
                            s_slot = a // H
                            e_slot = a % H
                            length = (e_slot - s_slot) % H + 1
                            # Validate: only count if this matches actual legal play lengths
                            p_by_len[length] = p_by_len.get(length, 0) + probs[a].item()

                    p_scout = probs[256:].sum().item()

                    if has_pair:
                        pair_states += 1
                        pair_p3plus_sum += sum(v for k, v in p_by_len.items() if k >= 3)

                    if has_triple:
                        triple_states += 1
                        p_single_sum += p_by_len.get(1, 0)
                        p_pair_sum += p_by_len.get(2, 0)
                        p_triple = sum(v for k, v in p_by_len.items() if k >= 3)
                        p_triple_sum += p_triple
                        p_scout_sum += p_scout
                        # Check argmax
                        best_action = probs.argmax().item()
                        if best_action < 256:
                            s_slot = best_action // H
                            e_slot = best_action % H
                            best_len = (e_slot - s_slot) % H + 1
                            if best_len >= 3:
                                chose_triple += 1
                            elif best_len == 2:
                                chose_pair += 1
                                if g.current_play is not None:
                                    non_triple_has_current_play += 1
                                else:
                                    non_triple_opening += 1
                            else:
                                chose_single += 1
                                if g.current_play is not None:
                                    non_triple_has_current_play += 1
                                else:
                                    non_triple_opening += 1
                        else:
                            chose_scout += 1
                            if g.current_play is not None:
                                non_triple_has_current_play += 1
                            else:
                                non_triple_opening += 1

                # Take action (use network policy to get realistic states)
                masked_logits = logits.clone() if (has_triple or has_pair) else None
                if masked_logits is not None:
                    action = torch.multinomial(probs, 1).item()
                else:
                    # No pair/triple, just random
                    action = None

                # For simplicity, play random legal action to advance game
                if random.random() < 0.7 and legal:
                    s, e = random.choice(legal)
                    g.apply_play(s, e)
                elif g.current_play is not None and len(hand) < 16:
                    g.apply_scout(random.random() < 0.5, random.random() < 0.5,
                                 random.randint(0, len(hand)))
                elif legal:
                    s, e = random.choice(legal)
                    g.apply_play(s, e)
                else:
                    break

        except Exception:
            continue

    print(f"=== Triple Choice Analysis ({n_games} games, {num_players}p) ===\n")

    if triple_states > 0:
        print(f"States where triple+ is legal: {triple_states}")
        print(f"\nAvg probability mass (when triple is legal):")
        print(f"  P(single):  {p_single_sum/triple_states:.4f}")
        print(f"  P(pair):    {p_pair_sum/triple_states:.4f}")
        print(f"  P(triple+): {p_triple_sum/triple_states:.4f}")
        print(f"  P(scout):   {p_scout_sum/triple_states:.4f}")
        print(f"\nArgmax breakdown (when triple is legal):")
        print(f"  Triple+: {chose_triple:>5} ({100*chose_triple/triple_states:.1f}%)")
        print(f"  Pair:    {chose_pair:>5} ({100*chose_pair/triple_states:.1f}%)")
        print(f"  Single:  {chose_single:>5} ({100*chose_single/triple_states:.1f}%)")
        print(f"  Scout:   {chose_scout:>5} ({100*chose_scout/triple_states:.1f}%)")
        non_triple = chose_pair + chose_single + chose_scout
        if non_triple > 0:
            print(f"\nWhen NOT choosing triple ({non_triple} states):")
            print(f"  Has current play (must beat): {non_triple_has_current_play} ({100*non_triple_has_current_play/non_triple:.1f}%)")
            print(f"  Opening (no current play):    {non_triple_opening} ({100*non_triple_opening/non_triple:.1f}%)")
    else:
        print("No states with legal triples found!")

    if pair_states > 0:
        print(f"\n--- For comparison (when pair is legal): ---")
        print(f"  States: {pair_states}")
        print(f"  Avg P(3+): {pair_p3plus_sum/pair_states:.4f}")


if __name__ == "__main__":
    ckpt = os.path.join(SCRIPT_DIR, "bots", "v7_12", "iter_1104.pt")
    analyze(ckpt)
