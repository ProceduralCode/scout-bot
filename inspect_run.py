"""Inspect metrics history from a training checkpoint.

Usage: python inspect_run.py <checkpoint_path> [--metric NAME] [--raw]

  --metric NAME  Show only these metrics (can be repeated)
  --raw          Print full per-iteration table instead of summary

Examples:
  python inspect_run.py v2_3/latest.pt
  python inspect_run.py v2_3/latest.pt --metric entropy_play_end --metric advantage_std
  python inspect_run.py v2_3/latest.pt --raw
"""

import argparse
import torch


def _sample_indices(n, count=5):
    """Pick evenly-spaced indices including first and last."""
    if n <= count:
        return list(range(n))
    step = (n - 1) / (count - 1)
    return [round(i * step) for i in range(count)]


def _trend_arrow(first, last):
    """Simple trend indicator."""
    if last > first * 1.05:
        return "^"
    elif last < first * 0.95:
        return "v"
    return "="


def main():
    parser = argparse.ArgumentParser(description="Inspect training run metrics")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--metric", action="append", dest="metrics",
                        help="Show only these metrics (repeatable)")
    parser.add_argument("--raw", action="store_true",
                        help="Print full per-iteration table")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    history = checkpoint.get("metrics_history", {})

    if not history:
        print("No metrics_history found in checkpoint.")
        return

    # Separate eval metrics (different length) from per-iteration metrics
    eval_keys = [k for k in history if k.startswith("eval_")]
    iter_keys = [k for k in history if not k.startswith("eval_")
                 and k != "iteration" and history[k]]

    if args.metrics:
        iter_keys = [k for k in iter_keys if k in args.metrics]
        eval_keys = [k for k in eval_keys if k in args.metrics]

    # Print config if present
    if "config" in checkpoint:
        cfg = checkpoint["config"]
        print("=== Config ===")
        for k, v in sorted(cfg.items()):
            print(f"  {k}: {v}")
        print()

    iterations = history.get("iteration", [])
    n = len(iterations)
    print(f"=== Run: {n} iterations ===\n")

    if args.raw and iter_keys:
        # Full table mode
        col_width = 10
        header = f"{'iter':>6}"
        for k in iter_keys:
            header += f"  {k:>{col_width}}"
        print(header)
        print("-" * len(header))
        for i, it in enumerate(iterations):
            row = f"{it:>6}"
            for k in iter_keys:
                vals = history[k]
                if i < len(vals):
                    v = vals[i]
                    if isinstance(v, float):
                        row += f"  {v:>{col_width}.4f}"
                    else:
                        row += f"  {v:>{col_width}}"
                else:
                    row += f"  {'':>{col_width}}"
            print(row)
        print()

    elif iter_keys:
        # Summary mode: trajectory snapshots + stats per metric
        sample_idx = _sample_indices(n)
        sample_iters = [iterations[i] for i in sample_idx]
        snap_header = "  ".join(f"i{it:>4}" for it in sample_iters)

        for k in iter_keys:
            vals = history[k]
            if not vals:
                continue
            first, last = vals[0], vals[-1]
            mn, mx = min(vals), max(vals)
            trend = _trend_arrow(first, last)

            snapshots = []
            for i in sample_idx:
                if i < len(vals):
                    snapshots.append(f"{vals[i]:>6.4f}")
                else:
                    snapshots.append(f"{'':>6}")

            print(f"  {k}  [{trend}]")
            print(f"    trajectory:  {snap_header}")
            print(f"                 {'  '.join(snapshots)}")
            print(f"    min={mn:.4f}  max={mx:.4f}  mean={sum(vals)/len(vals):.4f}")
            print()

    # Eval metrics
    eval_iters = history.get("eval_iteration", [])
    if eval_iters and eval_keys:
        margin_keys = [k for k in eval_keys if k != "eval_iteration"]
        if margin_keys:
            print("=== Eval ===\n")
            # Show first, last, and a few in between
            sample_idx = _sample_indices(len(eval_iters))
            for k in margin_keys:
                vals = history[k]
                if not vals:
                    continue
                points = []
                for i in sample_idx:
                    if i < len(vals):
                        points.append(f"i{eval_iters[i]}={vals[i]:+.1f}")
                print(f"  {k}: {', '.join(points)}")
            print()


if __name__ == "__main__":
    main()
