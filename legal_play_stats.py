"""Diagnostic: what fraction of game states have legal plays of each length?"""
import os, sys, random
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from game import Game, Phase
from encoding import get_legal_plays

def _random_scout(g):
    """Apply a random legal scout action."""
    hand = g.players[g.current_player].hand
    left_end = random.random() < 0.5
    flip = random.random() < 0.5
    pos = random.randint(0, len(hand))
    g.apply_scout(left_end, flip, pos)

def analyze_games(n_games=10000, num_players=3):
    """Play random games, tracking legal play availability by length."""
    total_play_turns = 0
    has_legal_play = 0
    length_available = {}  # length -> count of turns where at least one play of this length exists
    length_chosen = {}

    # When triples ARE legal, what else is available?
    triple_available_detail = []

    for game_i in range(n_games):
        g = Game(num_players)
        g.start_round()
        for p in range(num_players):
            g.submit_flip_decision(p, random.random() < 0.5)

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
                total_play_turns += 1

                if legal:
                    has_legal_play += 1
                    lengths_present = set()
                    n_by_len = {}
                    for s, e in legal:
                        l = e - s + 1
                        lengths_present.add(l)
                        n_by_len[l] = n_by_len.get(l, 0) + 1

                    for l in lengths_present:
                        length_available[l] = length_available.get(l, 0) + 1

                    if any(l >= 3 for l in lengths_present):
                        triple_available_detail.append((
                            n_by_len.get(1, 0),
                            n_by_len.get(2, 0),
                            n_by_len.get(3, 0),
                            sum(v for k, v in n_by_len.items() if k >= 4)
                        ))

                    # Random action: play 70%, scout 30%
                    if random.random() < 0.7:
                        s, e = random.choice(legal)
                        l = e - s + 1
                        length_chosen[l] = length_chosen.get(l, 0) + 1
                        g.apply_play(s, e)
                    elif g.current_play is not None and len(hand) < 16:
                        _random_scout(g)
                    else:
                        s, e = random.choice(legal)
                        l = e - s + 1
                        length_chosen[l] = length_chosen.get(l, 0) + 1
                        g.apply_play(s, e)
                else:
                    # No legal play - must scout
                    if g.current_play is not None and len(hand) < 16:
                        _random_scout(g)
                    else:
                        break
        except (AssertionError, Exception):
            continue  # Skip broken games

    print(f"=== Legal Play Length Analysis ({n_games} games, {num_players}p) ===\n")
    print(f"Total play turns analyzed: {total_play_turns}")
    print(f"Turns with any legal play: {has_legal_play} ({100*has_legal_play/total_play_turns:.1f}%)\n")

    print("Fraction of turns where at least one play of length N is legal:")
    for l in sorted(length_available.keys()):
        count = length_available[l]
        pct = 100 * count / total_play_turns
        print(f"  Length {l}: {count:>8} turns ({pct:5.1f}%)")

    print(f"\nWhen random policy chooses to play, length distribution:")
    total_chosen = sum(length_chosen.values())
    for l in sorted(length_chosen.keys()):
        count = length_chosen[l]
        pct = 100 * count / total_chosen
        print(f"  Length {l}: {count:>8} plays ({pct:5.1f}%)")

    if triple_available_detail:
        print(f"\n--- When a triple+ IS legal ({len(triple_available_detail)} turns) ---")
        avg_singles = sum(d[0] for d in triple_available_detail) / len(triple_available_detail)
        avg_pairs = sum(d[1] for d in triple_available_detail) / len(triple_available_detail)
        avg_triples = sum(d[2] for d in triple_available_detail) / len(triple_available_detail)
        avg_4plus = sum(d[3] for d in triple_available_detail) / len(triple_available_detail)
        print(f"  Avg legal singles: {avg_singles:.1f}")
        print(f"  Avg legal pairs:   {avg_pairs:.1f}")
        print(f"  Avg legal triples: {avg_triples:.1f}")
        print(f"  Avg legal 4+:      {avg_4plus:.1f}")

        # What fraction of turns with triple available also have a pair available?
        has_pair_too = sum(1 for d in triple_available_detail if d[1] > 0)
        print(f"  Also have pair:    {100*has_pair_too/len(triple_available_detail):.1f}%")
        has_single_too = sum(1 for d in triple_available_detail if d[0] > 0)
        print(f"  Also have single:  {100*has_single_too/len(triple_available_detail):.1f}%")


if __name__ == "__main__":
    analyze_games(10000, 3)
