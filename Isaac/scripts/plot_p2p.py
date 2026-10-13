"""Success and return per task vs env steps and wall-clock for Play2Perfect runs (``train_p2p.py`` metrics).

Each epoch's ``metrics.jsonl`` line holds, per task and per SAPG block, the sums over the episodes that ended in
that epoch (episodes cut by a Homo switch are not in them). Here they are summed into bins of env steps / hours
and divided: success = full successes / episodes. ``--block last`` uses only the last SAPG block (exploration
coefficient 0, the block rl_games' own reward log follows); ``all`` uses every env.

    python plot_p2p.py --out runs/p2p_analysis --block all \\
        --run Hetero tab:blue runs/p2p_train/0_p2p_hetero_n12288_s42 \\
        --run Homo tab:orange runs/p2p_train/0_p2p_homo_b40_n12288_s42

matplotlib only (no Isaac Sim).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(run_dir: Path):
    return [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]


def binned(metrics, task: str, block: str, x_of, edges):
    eps = np.zeros(len(edges) - 1)
    succ = np.zeros(len(edges) - 1)
    ret = np.zeros(len(edges) - 1)
    for m in metrics:
        d = m["episodes"].get(task, {})
        if block == "last":
            keys = [k for k in d if k != "all"]
            v = d[max(keys, key=int)] if keys else None
        else:
            v = d.get("all")
        if not v or v["episodes"] <= 0:
            continue
        i = np.searchsorted(edges, x_of(m), side="right") - 1
        if 0 <= i < len(eps):
            eps[i] += v["episodes"]
            succ[i] += v["success"]
            ret[i] += v["return"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return succ / eps, ret / eps, eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs=3, action="append", required=True, metavar=("LABEL", "COLOR", "RUN_DIR"))
    p.add_argument("--block", choices=["last", "all"], default="all")
    p.add_argument("--step_bin", type=float, default=50e6)
    p.add_argument("--time_bin_h", type=float, default=0.5)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [(label, color, load(Path(d))) for label, color, d in a.run]
    tasks = list(runs[0][2][0]["episodes"].keys())
    max_steps = max(m[-1]["env_steps"] for _, _, m in runs)
    max_h = max(m[-1]["wall_time"] / 3600 for _, _, m in runs)
    step_edges = np.arange(0, max_steps + a.step_bin, a.step_bin)
    time_edges = np.arange(0, max_h + a.time_bin_h, a.time_bin_h)
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for what, idx in (("success", 0), ("return", 1)):
        fig, axes = plt.subplots(2, len(tasks), figsize=(4 * len(tasks), 7), sharey=(what == "success"), squeeze=False)
        for c, task in enumerate(tasks):
            for r, (edges, x_of, xlabel) in enumerate((
                    (step_edges, lambda m: m["env_steps"], f"env steps (millions; {a.step_bin / 1e6:g} M bins)"),
                    (time_edges, lambda m: m["wall_time"] / 3600, f"wall-clock hours ({a.time_bin_h:g} h bins)"))):
                ax = axes[r][c]
                for label, color, m in runs:
                    y = binned(m, task, a.block, x_of, edges)[idx]
                    x = (edges[:-1] + edges[1:]) / 2 / (1e6 if r == 0 else 1)
                    ax.plot(x, y, color=color, label=label, lw=2)
                    if r == 0 and what == "success":  # mean of the last three bins that have episodes
                        last = y[np.isfinite(y)][-3:]
                        summary.setdefault(label, {})[task] = float(last.mean()) if last.size else None
                ax.set_xlabel(xlabel)
                if r == 0:
                    ax.set_title(task)
                if c == 0:
                    ax.set_ylabel(f"{what} (training policy, {a.block} blocks)")
                ax.grid(alpha=0.3)
        axes[0][0].legend()
        fig.tight_layout()
        fig.savefig(a.out / f"p2p_{what}_{a.block}.png", dpi=110)
        plt.close(fig)
    (a.out / f"p2p_summary_{a.block}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
