#!/usr/bin/env python
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_RE = re.compile(r"iter (\d+), loss ([0-9.]+).*?(?:grad_norm ([0-9.]+))?")
EVAL_RE = re.compile(r"step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)")


def parse_log(path: Path):
    train = []
    evals = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if match := TRAIN_RE.search(line):
            train.append((int(match.group(1)), float(match.group(2))))
        if match := EVAL_RE.search(line):
            evals.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    return train, evals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/figures/loss_curves.png"))
    args = parser.parse_args()

    plt.figure(figsize=(9, 5))
    for log in args.logs:
        train, evals = parse_log(log)
        label = log.stem
        if train:
            xs, ys = zip(*train)
            plt.plot(xs, ys, alpha=0.35, label=f"{label} train batch")
        if evals:
            xs = [x[0] for x in evals]
            vals = [x[2] for x in evals]
            plt.plot(xs, vals, linewidth=2, label=f"{label} val")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=160)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

