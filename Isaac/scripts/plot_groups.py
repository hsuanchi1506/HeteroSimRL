"""Readable comparison figures for groups of runs (e.g. Hetero, Hetero with per-task advantage normalisation, Homo).

The raw curves of analyze_runs.py are hard to compare across Hetero and Homo: a Hetero point is one episode of
one task's ~43 envs (128 / 3), drawn every episode; a Homo point is one episode of all 128 envs, drawn only when
the task's block comes round (every 12 PPO epochs). So Hetero lines look jagged from sampling noise alone. Here
every run is averaged into the same bins (env steps and wall-clock), weighted by envs, and each group is drawn as
the mean over its seeds with the min-max range of the seeds as a band.

    python scripts/plot_groups.py --out runs/analysis/groups \\
        --group "Hetero (official)" tab:blue logs/rl_games/Factory/hetero_n128_s0 logs/... \\
        --group "Hetero (per-task adv. norm.)" tab:green logs/... \\
        --group "Homo" tab:orange logs/...

Writes success_binned.png, return_binned.png (RULES section 4: reward vs env steps and wall-clock) and
gradients_clean.png (matplotlib, no Isaac Sim needed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ("peg_insert", "gear_mesh", "nut_thread")
SHORT = {"peg_insert": "peg", "gear_mesh": "gear", "nut_thread": "nut"}


def load(d: Path):
    info = json.loads((d / "run_info.json").read_text())
    eps = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l.strip()]
    per_epoch = info["num_envs"] * info["horizon_length"]
    for e in eps:
        e["steps_end"] = e["epoch"] * per_epoch
        wt = e.get("wall_time_excl_diag")
        e["wall_h"] = (wt if wt is not None else e["wall_time"]) / 3600
    gd = d / "grad_diag.jsonl"
    grad = [json.loads(l) for l in gd.read_text().splitlines() if l.strip()] if gd.exists() else []
    return dict(info=info, eps=eps, grad=grad)


def binned(run, task, key, edges, field="success"):
    """Env-weighted mean ``field`` (success, or ``ret`` = episode return) of the run's full-length episodes of
    ``task`` in each bin of ``key``."""
    out = np.full(len(edges) - 1, np.nan)
    eps = [e for e in run["eps"] if e["task"] == task and not e["truncated"] and e.get(field) is not None]
    x = np.array([e[key] for e in eps])
    s = np.array([e[field] for e in eps])
    w = np.array([e["n_envs"] for e in eps], dtype=float)
    for i in range(len(edges) - 1):
        m = (x > edges[i]) & (x <= edges[i + 1])
        if m.any():
            out[i] = (s[m] * w[m]).sum() / w[m].sum()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", nargs="+", action="append", required=True, metavar=("LABEL COLOR", "RUN_DIR"),
                   help="label, matplotlib color, then the run directories of the group")
    p.add_argument("--step_bin", type=float, default=0.2e6)
    p.add_argument("--time_bin_h", type=float, default=0.1)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [(g[0], g[1], [load(Path(d)) for d in g[2:]]) for g in args.group]
    args.out.mkdir(parents=True, exist_ok=True)
    max_steps = max(r["eps"][-1]["steps_end"] for _, _, runs in groups for r in runs)
    max_h = max(r["eps"][-1]["wall_h"] for _, _, runs in groups for r in runs)
    step_edges = np.arange(0, max_steps + args.step_bin, args.step_bin)
    time_edges = np.arange(0, max_h + args.time_bin_h, args.time_bin_h)

    # ---- success and episode return, binned, mean over seeds with min-max band
    for field, ylabel, ylim, fname in (("success", "success (training policy)", (-0.02, 1.02), "success_binned.png"),
                                       ("ret", "episode return (training policy)", None, "return_binned.png")):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharey=True)
        for row, (key, edges, scale, xlabel) in enumerate((
                ("steps_end", step_edges, 1e6, f"env steps (millions; {args.step_bin / 1e6:g} M bins)"),
                ("wall_h", time_edges, 1.0, f"wall-clock hours ({args.time_bin_h:g} h bins)"))):
            centers = (edges[:-1] + edges[1:]) / 2 / scale
            for j, task in enumerate(TASKS):
                ax = axes[row, j]
                for label, color, runs in groups:
                    y = np.array([binned(r, task, key, edges, field) for r in runs])
                    with np.errstate(all="ignore"):
                        mean, lo, hi = np.nanmean(y, 0), np.nanmin(y, 0), np.nanmax(y, 0)
                    ok = ~np.isnan(mean)
                    ax.fill_between(centers[ok], lo[ok], hi[ok], color=color, alpha=0.15, lw=0)
                    ax.plot(centers[ok], mean[ok], color=color, lw=2, label=label)
                ax.set_xlabel(xlabel)
                ax.grid(alpha=0.3)
                if ylim is not None:
                    ax.set_ylim(*ylim)
                if row == 0:
                    ax.set_title(SHORT[task], fontsize=13)
            axes[row, 0].set_ylabel(ylabel)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(groups), frameon=False, fontsize=11)
        fig.text(0.5, 0.005, "line: mean of the seeds; band: lowest to highest seed", ha="center", fontsize=9,
                 color="0.35")
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        fig.savefig(args.out / fname, dpi=110)
        plt.close(fig)

    # ---- gradients: update-to-update cosine by category, and task-pair cosine over training
    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1))
    ax0 = fig.add_subplot(gs[0, :])
    cats = []  # (label, color, values)
    for label, color, runs in groups:
        same, across = [], []
        for r in runs:
            probes = {g["epoch"]: g for g in r["grad"] if g["kind"] == "probe"}
            for e, g in probes.items():
                if g.get("adjacent_to_prev") and g.get("cos_total_prev") is not None and g.get("prev_epoch") in probes:
                    switched = set(probes[g["prev_epoch"]]["tasks"]) != set(g["tasks"])
                    (across if switched else same).append(g["cos_total_prev"])
        if across:
            cats.append((f"{label}\nacross a task switch", color, across))
            cats.append((f"{label}\nsame task", color, same))
        else:
            cats.append((f"{label}\nconsecutive updates", color, same))
    rng = np.random.default_rng(0)
    for i, (lab, color, vals) in enumerate(cats):
        x = i + rng.uniform(-0.18, 0.18, len(vals))
        ax0.scatter(x, vals, s=14, color=color, alpha=0.55, lw=0)
        m = float(np.mean(vals))
        ax0.plot([i - 0.3, i + 0.3], [m, m], color="k", lw=2.5)
        ax0.text(i + 0.33, m, f"{m:+.2f}", va="center", fontsize=10)
    ax0.axhline(0, color="0.4", lw=0.8)
    ax0.set_xticks(range(len(cats)), [c[0] for c in cats], fontsize=9.5)
    ax0.set_xlim(-0.6, len(cats) - 0.2)
    ax0.set_ylabel("cosine of two consecutive\nupdates (all seeds)")
    ax0.set_title("Does the update direction flip?  (black bar: mean)", fontsize=12)
    ax0.grid(axis="y", alpha=0.3)

    pairs = ("peg_insert|gear_mesh", "peg_insert|nut_thread", "gear_mesh|nut_thread")
    bin_ep = 20  # probes come in adjacent pairs (k*10, k*10+1): average them, and the seeds, per 20 epochs
    for j, pair in enumerate(pairs):
        ax = fig.add_subplot(gs[1, j])
        for label, color, runs in groups:
            pts = [(g["epoch"], g["pair_cos"][pair]) for r in runs for g in r["grad"]
                   if g["kind"] == "probe" and pair in g.get("pair_cos", {})]
            if not pts:
                continue
            ep = np.array([e for e, _ in pts])
            val = np.array([v for _, v in pts])
            edges = np.arange(0, ep.max() + bin_ep, bin_ep)
            xs, ys = [], []
            for i in range(len(edges) - 1):
                m = (ep > edges[i]) & (ep <= edges[i + 1])
                if m.any():
                    xs.append((edges[i] + edges[i + 1]) / 2)
                    ys.append(val[m].mean())
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=3, label=label)
        a, b = pair.split("|")
        ax.set_title(f"{SHORT[a]} vs {SHORT[b]}", fontsize=12)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_ylim(-0.6, 0.8)
        ax.set_xlabel(f"PPO epoch ({bin_ep}-epoch bins)")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("task-pair gradient cosine\n(mean of seeds and probes)")
            ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "gradients_clean.png", dpi=110)
    plt.close(fig)
    print(f"[INFO] wrote success_binned.png, return_binned.png and gradients_clean.png in {args.out}")


if __name__ == "__main__":
    main()
