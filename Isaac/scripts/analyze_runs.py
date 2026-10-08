"""Compare Homo and Hetero training runs: per-task success and return against env steps and wall-clock time.

Reads each run directory written by train_hetero.py / train_homo.py (metrics.jsonl, episodes.jsonl,
run_info.json) and writes a summary table (markdown) and one figure. No Isaac Sim needed (numpy, matplotlib).

    python scripts/analyze_runs.py logs/rl_games/Factory/hetero_n128_s0 logs/rl_games/Factory/homo_rollout4_n128_s0 \\
        --out results/n128

Notes on the logs:
* rl_games hands the observer the frame count from before the epoch's rollout, so ``env_steps`` in both files
  lags one epoch; the env steps at the end of epoch e are e * num_envs * horizon, which is what is used here.
* Episodes cut by a Homo task switch (``truncated``) are left out of the success curves (their success is
  measured after a fraction of an episode); their count is reported.
* Success is the fraction of a task's envs that meet the task's success condition at the episode's last step,
  with the training (stochastic) policy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TASKS = ("peg_insert", "gear_mesh", "nut_thread")


def load_run(d: Path):
    info = json.loads((d / "run_info.json").read_text())
    metrics = [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines() if line.strip()]
    ep_path = d / "episodes.jsonl"
    episodes = [json.loads(line) for line in ep_path.read_text().splitlines() if line.strip()] if ep_path.exists() else []
    per_epoch = info["num_envs"] * info["horizon_length"]
    for e in episodes:
        e["steps_end"] = e["epoch"] * per_epoch
        wt = e.get("wall_time_excl_diag")
        e["wall_h"] = (wt if wt is not None else e["wall_time"]) / 3600
    gd_path = d / "grad_diag.jsonl"
    grad = [json.loads(line) for line in gd_path.read_text().splitlines() if line.strip()] if gd_path.exists() else []
    label = info.get("mode", "?")
    if info.get("homo"):
        label += f"-{info['homo']['switch']}{info['homo']['block']}"
    if info.get("adv_norm") == "per_task":
        label += "-ptadv"
    label += f" {info.get('scene', '')} n{info['num_envs']} s{info['seed']}"
    mode = info.get("mode")
    if info.get("adv_norm") == "per_task":
        mode += "_ptadv"  # its own color: hetero with per-task advantage normalisation
    return dict(dir=d, info=info, metrics=metrics, episodes=episodes, grad=grad, label=label, mode=mode)


def grad_summary(run):
    """RULES section 5 from grad_diag.jsonl: task-pair cosines (Hetero), update-to-update cosine of the total
    gradient on adjacent probes, split for Homo into 'same task' and 'across a task switch'."""
    probes = {g["epoch"]: g for g in run["grad"] if g["kind"] == "probe"}
    pair = defaultdict(list)
    norms = defaultdict(list)
    same, across = [], []
    for e, g in sorted(probes.items()):
        for k, v in g.get("pair_cos", {}).items():
            pair[k].append(v)
        for t, v in g["tasks"].items():
            norms[t].append(v["grad_norm"])
        if g.get("adjacent_to_prev") and g.get("cos_total_prev") is not None:
            prev = probes.get(g.get("prev_epoch"))
            switched = prev is not None and set(prev["tasks"]) != set(g["tasks"])
            (across if switched else same).append(g["cos_total_prev"])
    out = dict(run=run["label"], probes=len(probes))
    for k, v in pair.items():
        out[f"pair_cos_mean {k}"] = float(np.mean(v))
        out[f"pair_cos_neg_frac {k}"] = float(np.mean(np.array(v) < 0))
    out["adjacent_cos_same_task_mean"] = float(np.mean(same)) if same else float("nan")
    out["adjacent_cos_same_task_n"] = len(same)
    out["adjacent_cos_across_switch_mean"] = float(np.mean(across)) if across else float("nan")
    out["adjacent_cos_across_switch_n"] = len(across)
    for t, v in norms.items():
        out[f"grad_norm_median {t}"] = float(np.median(v))
    return out, probes


def curves(run, task, key="success"):
    pts = [(e["steps_end"], e["wall_h"], e[key]) for e in run["episodes"]
           if e["task"] == task and not e["truncated"] and e.get(key) is not None]
    return np.array(pts).reshape(-1, 3)


def summary(run, last_frac=0.2):
    m = run["metrics"]
    total_steps = m[-1]["epoch"] * run["info"]["num_envs"] * run["info"]["horizon_length"] if m else 0
    wall = (m[-1].get("wall_time_excl_diag") or m[-1]["wall_time"]) / 3600 if m else float("nan")
    env_time = sum(x.get("env_step_time", 0.0) for x in m) / 3600
    row = dict(run=run["label"], epochs=m[-1]["epoch"] if m else 0, env_steps=total_steps, wall_h=wall,
               env_step_h=env_time, truncated=sum(e["truncated"] for e in run["episodes"]))
    for t in TASKS:
        c = curves(run, t)
        if len(c):
            cut = c[:, 0] >= (1 - last_frac) * total_steps
            row[f"succ_{t}"] = float(c[cut, 2].mean()) if cut.any() else float("nan")
            row[f"succ_max_{t}"] = float(c[:, 2].max())
            r = curves(run, t, "ret")
            row[f"ret_{t}"] = float(r[r[:, 0] >= (1 - last_frac) * total_steps, 2].mean()) if len(r) else float("nan")
        else:
            row[f"succ_{t}"] = row[f"succ_max_{t}"] = row[f"ret_{t}"] = float("nan")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("analysis"))
    p.add_argument("--last_frac", type=float, default=0.2, help="Final success = mean over this last fraction of steps.")
    args = p.parse_args()
    runs = [load_run(d) for d in args.runs]
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [summary(r, args.last_frac) for r in runs]
    cols = ["run", "epochs", "env_steps", "wall_h", "env_step_h", "truncated"] + \
           [f"succ_{t}" for t in TASKS] + [f"succ_max_{t}" for t in TASKS] + [f"ret_{t}" for t in TASKS]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    table = "\n".join(lines)
    grad_rows = [grad_summary(r) for r in runs]
    glines = []
    for g, _ in grad_rows:
        glines.append(f"- **{g['run']}** ({g['probes']} probes): " + ", ".join(
            f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in g.items() if k not in ("run", "probes")))
    (args.out / "summary.md").write_text(
        f"Final success = mean over the last {args.last_frac:.0%} of env steps (non-truncated episodes, training "
        f"policy). wall_h excludes gradient diagnostics.\n\n{table}\n\n## Gradients (RULES section 5)\n\n"
        + "\n".join(glines) + "\n", encoding="utf-8")
    print(table)
    print("\n".join(glines))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] no matplotlib: table only")
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    colors = {"hetero": "tab:blue", "homo": "tab:orange", "hetero_ptadv": "tab:green", "homo_ptadv": "tab:red"}
    seen = defaultdict(int)
    for run in runs:
        c = colors.get(run["mode"], "gray")
        seen[run["mode"]] += 1
        for j, t in enumerate(TASKS):
            cv = curves(run, t)
            if not len(cv):
                continue
            axes[0, j].plot(cv[:, 0] / 1e6, cv[:, 2], color=c, alpha=0.8, lw=1.2, label=run["label"])
            axes[1, j].plot(cv[:, 1], cv[:, 2], color=c, alpha=0.8, lw=1.2, label=run["label"])
    for j, t in enumerate(TASKS):
        axes[0, j].set_title(t)
        axes[0, j].set_xlabel("env steps (M)")
        axes[1, j].set_xlabel("wall-clock (h, excl. gradient diagnostics)")
        for i in range(2):
            axes[i, j].grid(alpha=0.3)
            axes[i, j].set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("success (end of episode)")
    axes[1, 0].set_ylabel("success (end of episode)")
    axes[0, 2].legend(fontsize=7, loc="lower right")
    fig.suptitle("Factory on PhysX: Hetero (blue), Hetero with per-task advantage normalisation (green), "
                 "Homo (orange); training policy")
    fig.tight_layout()
    fig.savefig(args.out / "success_curves.png", dpi=110)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for run, (_, probes) in zip(runs, grad_rows):
        c = colors.get(run["mode"], "gray")
        ep = sorted(probes)
        for k, ls in zip(("peg_insert|gear_mesh", "peg_insert|nut_thread", "gear_mesh|nut_thread"), ("-", "--", ":")):
            pts = [(e, probes[e]["pair_cos"][k]) for e in ep if k in probes[e].get("pair_cos", {})]
            if pts:
                axes[0].plot(*zip(*pts), ls, color=c, alpha=0.7, label=f"{run['label']} {k}")
        for t, m in zip(TASKS, ("o", "s", "^")):
            pts = [(e, probes[e]["tasks"][t]["grad_norm"]) for e in ep if t in probes[e]["tasks"]]
            if pts:
                axes[1].semilogy(*zip(*pts), m, color=c, alpha=0.6, ms=3, label=f"{run['label']} {t}")
        adj = [(e, probes[e]["cos_total_prev"], set(probes[probes[e]["prev_epoch"]]["tasks"]) != set(probes[e]["tasks"]))
               for e in ep if probes[e].get("adjacent_to_prev") and probes[e].get("prev_epoch") in probes]
        for flag, m in ((False, "o"), (True, "x")):
            pts = [(e, v) for e, v, sw in adj if sw == flag]
            if pts:
                axes[2].plot(*zip(*pts), m, color=c, alpha=0.8,
                             label=f"{run['label']} {'across switch' if flag else 'same task'}")
    axes[0].set_title("task-pair gradient cosine (Hetero)")
    axes[1].set_title("per-task actor gradient norm")
    axes[2].set_title("update-to-update cosine (adjacent probes)")
    for a in axes:
        a.set_xlabel("PPO epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=6)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[2].axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(args.out / "gradients.png", dpi=110)
    print(f"[INFO] wrote {args.out / 'summary.md'}, success_curves.png and gradients.png in {args.out}")


if __name__ == "__main__":
    main()
