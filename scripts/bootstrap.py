"""
Bootstrap uncertainty for next-genre probe AP across experiment modes.

Compares all modes on the same validation users by resampling with replacement.
Default artifact stem is day06_mlp_<mode>_probe_{targets,scores}.npy
(use --prefix day06 for the older linear-head four-mode run).
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

MODES = ["zero", "status", "token", "vime"]


def load_probe(artifact: Path, prefix: str, mode: str):
    stem = f"{prefix}_{mode}"
    targets = np.load(artifact / f"{stem}_probe_targets.npy")
    scores = np.load(artifact / f"{stem}_probe_scores.npy")
    return targets, scores


def micro_ap(targets, scores):
    return average_precision_score(targets, scores, average="micro")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap probe-AP comparison across zero/status/token/vime."
    )
    parser.add_argument(
        "--prefix",
        default="day06_mlp",
        help="Artifact name prefix before _<mode>_probe_*.npy (default: day06_mlp).",
    )
    parser.add_argument(
        "--baseline",
        default="zero",
        choices=MODES,
        help="Mode used as baseline for pairwise differences (default: zero).",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    artifact = Path(__file__).resolve().parents[1] / "artifacts"
    data = {}
    baseline_y = None

    for mode in MODES:
        path = artifact / f"{args.prefix}_{mode}_probe_targets.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Available prefixes often include "
                f"'day06_mlp' or 'day06'."
            )
        y, s = load_probe(artifact, args.prefix, mode)
        if baseline_y is None:
            baseline_y = y
        elif not np.array_equal(baseline_y, y):
            raise ValueError(
                f"probe targets for mode={mode} differ from baseline; splits not aligned"
            )
        data[mode] = s

    y = baseline_y
    n = len(y)
    point = {mode: micro_ap(y, data[mode]) for mode in MODES}

    print(f"prefix={args.prefix}  samples={args.samples}  baseline={args.baseline}")
    print("point-estimate micro AP:")
    for mode in MODES:
        print(f"  {mode:8s}  {point[mode]:.4f}")

    rng = np.random.default_rng(args.seed)
    # diffs[mode] = list of AP(mode) - AP(baseline) over bootstrap draws
    diffs = {mode: [] for mode in MODES if mode != args.baseline}
    # wins[mode] = how often that mode has the highest AP on a bootstrap draw
    wins = {mode: 0 for mode in MODES}

    for _ in range(args.samples):
        index = rng.integers(0, n, size=n)
        y_b = y[index]
        aps = {mode: micro_ap(y_b, data[mode][index]) for mode in MODES}
        best = max(MODES, key=lambda m: aps[m])
        wins[best] += 1
        base = aps[args.baseline]
        for mode in diffs:
            diffs[mode].append(aps[mode] - base)

    print(f"\npairwise vs baseline={args.baseline} (treatment - baseline):")
    print(f"{'mode':8s}  {'mean':>8s}  {'95% lo':>8s}  {'95% hi':>8s}  {'P(>0)':>8s}")
    for mode in MODES:
        if mode == args.baseline:
            continue
        d = np.asarray(diffs[mode])
        lo, hi = np.quantile(d, [0.025, 0.975])
        print(
            f"{mode:8s}  {d.mean():8.4f}  {lo:8.4f}  {hi:8.4f}  {(d > 0).mean():8.3f}"
        )

    print("\nP(mode has highest AP on a bootstrap sample):")
    for mode in MODES:
        print(f"  {mode:8s}  {wins[mode] / args.samples:.3f}")


if __name__ == "__main__":
    main()
