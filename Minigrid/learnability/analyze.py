import functools
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#b4b2a9", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "lines.linewidth": 2, "axes.grid": True, "grid.color": "#e8e7e2",
                     "grid.linewidth": 0.6, "legend.frameon": False})

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figs")
TAU = 0.8
WIN = 5


def load(name):
    return torch.load(os.path.join(RES, f"{name}.pt"), weights_only=False)


def curves(log):
    sc, ep = log["success"].numpy(), log["episodes"].numpy()
    k = np.ones(WIN)
    win = lambda x: np.stack([np.convolve(r, k, mode="full")[:len(r)] for r in x])
    sm = win(sc) / np.maximum(win(ep), 1)
    sr = sc / np.maximum(ep, 1)
    return sr, sm


def metrics(log):
    sr, sm = curves(log)
    spi = log["steps_per_iter"]
    n = sr.shape[1]
    steps = (np.arange(n) + 1) * spi

    def first(mask):
        has = mask.any(1)
        return np.where(has, steps[mask.argmax(1)], np.inf)

    m = {"t80": first(sm >= TAU), "t50": first(sm >= 0.5), "disc": first(log["success"].numpy() > 0),
         "auc": sm.mean(1), "final": sm[:, -1], "budget": n * spi}
    return m


def table(name):
    d = load(name)
    m = metrics(d["log"])
    for i, run in enumerate(d["runs"]):
        print(f"{run.get('cond', '-'):14s} {run['task'][0]:16s} final {m['final'][i]:.2f} disc {m['disc'][i] / 1e6:.2f}M t80 {m['t80'][i] / 1e6:.2f}M")


def cell_stats(name, budget=None, group=("cond", "task")):
    d = load(name)
    log = d["log"]
    if budget:
        n = budget // log["steps_per_iter"]
        log = {k: (v[:, :n] if torch.is_tensor(v) and v.dim() > 1 else v) for k, v in log.items()}
    m = metrics(log)
    out = {}
    for i, run in enumerate(d["runs"]):
        key = tuple(run.get(g, "none") if g != "task" else run["task"][0] for g in group)
        out.setdefault(key, []).append({k: (v[i] if isinstance(v, np.ndarray) else v) for k, v in m.items()})
    return out


def summarize(cells):
    res = {}
    for k, v in cells.items():
        t80 = np.array([x["t80"] for x in v])
        res[k] = {"solve": np.isfinite(t80).mean(), "t80_med": np.median(t80), "auc": np.mean([x["auc"] for x in v]),
                  "disc": np.median([x["disc"] for x in v]), "n": len(v), "t80": t80}
    return res


def frontier(t80s, Ks):
    return np.array([np.mean([np.mean(t <= K) for t in t80s]) for K in Ks])


DOWN = ["goal", "goal_with_key", "pick_ball2", "ball_right", "goal_with_ball", "ball2_left", "key_near_goal",
        "open_with_ball", "close_from_right", "box_right", "ball2_near_goal"]
KS = np.arange(0, 10_000_001, 250_000)


def p0_table(probe_entry, key="stoch"):
    from gridworld import PREDICATES
    hit = probe_entry[key]["hit"].numpy()
    return {(r["cond"], r["pseed"]): dict(zip(PREDICATES, hit[i])) for i, r in enumerate(probe_entry["runs"])}


def sweep_table(name="finetune", budget=6_000_000):
    cells = cell_stats(name, budget=budget, group=("cond", "pseed", "task"))
    sc = cell_stats("scratch", budget=budget, group=("seed", "task"))
    return cells, sc


def fig_matrix():
    cells, sc = sweep_table()
    S = summarize({(c, t): sum([v for (c2, p, t2), v in cells.items() if c2 == c and t2 == t], []) for (c, p, t) in cells})
    Ssc = summarize({t: sum([v for (s, t2), v in sc.items() if t2 == t], []) for (s, t) in sc})
    conds = sorted({c for c, t in S}, key=lambda c: -np.mean([S[(c, t)]["auc"] for t in DOWN]))
    probe = load("probe")
    p0 = p0_table(probe["pre_final.pt"])
    p0g = p0_table(probe["pre_final.pt"], "greedy")
    p0_fresh = probe["fresh"]["stoch"]["hit"].numpy().mean(0)
    from gridworld import PREDICATES
    rows = ["scratch"] + conds
    M = np.array([[Ssc[t]["auc"] for t in DOWN]] + [[S[(c, t)]["auc"] for t in DOWN] for c in conds])
    Z = np.array([[p0_fresh[PREDICATES.index(t)] for t in DOWN]] +
                 [[np.mean([p0[(c, ps)][t] for ps in range(3)]) for t in DOWN] for c in conds])
    G = np.array([[0.0] * len(DOWN)] + [[np.mean([p0g[(c, ps)][t] for ps in range(3)]) for t in DOWN] for c in conds])
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"wspace": 0.35})
    im = axs[0].imshow(np.log10(np.maximum(Z, 1e-5)), cmap=SEQ, vmin=-5, vmax=0, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(DOWN)):
            axs[0].text(j, i, f"{Z[i, j]:.0e}" if Z[i, j] > 0 else "0", ha="center", va="center", fontsize=6,
                        color="white" if Z[i, j] > 1e-2 else INK)
            if G[i, j] > 0.05:
                axs[0].add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=SERIES[1], lw=1.5))
    axs[0].set_title("Zero-shot: stochastic hitting prob. p0 (log10)\norange box = greedy success > 5%")
    plt.colorbar(im, ax=axs[0], fraction=0.04)
    im = axs[1].imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(DOWN)):
            axs[1].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] > 0.5 else INK)
    axs[1].set_title("Learnability: mean success over 6M finetune steps (AUC)")
    plt.colorbar(im, ax=axs[1], fraction=0.04)
    for ax in axs:
        ax.set_xticks(range(len(DOWN)), DOWN, rotation=50, ha="right")
        ax.set_yticks(range(len(rows)), rows)
        ax.grid(False)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig_matrix.png"), dpi=150, bbox_inches="tight")
    return rows, M, Z, G


def t80_by(name, key_fields, budget=None):
    d = load(name)
    log = d["log"]
    if budget:
        n = budget // log["steps_per_iter"]
        log = {k: (v[:, :n] if torch.is_tensor(v) and v.dim() > 1 else v) for k, v in log.items()}
    m = metrics(log)
    out = {}
    for i, r in enumerate(d["runs"]):
        key = tuple(r.get(f, "none") for f in key_fields)
        out.setdefault(key, {}).setdefault(r["task"][0], []).append(m["t80"][i])
    return out


def frontier_curve(task_t80, tasks, Ks):
    return np.array([np.mean([np.mean(np.array(task_t80.get(t, [np.inf])) <= K) for t in tasks]) for K in Ks])


def fig_frontier(highlight=None, pre_steps=2_000_000):
    ft = t80_by("finetune", ("cond",))
    sc = t80_by("scratch", ())[()]
    Ks = np.arange(0, 6_000_001, 100_000)
    Kl = np.arange(0, 10_000_001, 100_000)
    fs = frontier_curve(sc, DOWN, Kl)
    curves = {c[0]: frontier_curve(v, DOWN, Ks) for c, v in ft.items()}
    order = sorted(curves, key=lambda c: -curves[c].mean())
    highlight = highlight or order[:3] + [c for c in ["mix", "impact", "explore"] if c not in order[:3]][:3]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, shift, title in [(axs[0], 0, "Frontier vs finetune budget K"),
                             (axs[1], pre_steps, f"Frontier vs total steps (pretrain {pre_steps / 1e6:.0f}M + K), non-amortized")]:
        for c in order:
            if c not in highlight:
                ax.plot((Ks + shift) / 1e6, curves[c], color=MUTED, lw=1, alpha=0.7)
        for j, c in enumerate(highlight):
            ax.plot((Ks + shift) / 1e6, curves[c], color=SERIES[j], label=c)
        ax.plot(Kl / 1e6, fs, color=INK, lw=2.5, ls="--", label="scratch")
        ax.set_title(title)
        ax.set_xlabel("env steps (M)")
    axs[0].set_ylabel(f"Learning frontier F(K)\n(fraction of {len(DOWN)} downstream tasks at >={TAU:.0%} success)")
    axs[1].legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(FIG, "fig_frontier.png"), dpi=150, bbox_inches="tight")
    return curves, fs, Ks, Kl


MECH_TGT = ["goal", "key_near_goal", "close_from_right", "open_with_ball"]
VARS = [("full", "full (behavior+rep+critic)"), ("soft", "soft: full, logits x0.1"), ("nocritic", "reset critic head"),
        ("prior", "behavior only (frozen prior + fresh net)"), ("trunk", "representation only (trunk, uniform init)"),
        ("perm", "permuted trunk (norm control)")]


def mech_table(budget=6_000_000):
    auc = {}
    for name, field in [("mech", "variant"), ("finetune", None)]:
        d = load(name)
        m = metrics(d["log"])
        for i, r in enumerate(d["runs"]):
            v = r.get("variant", "full") if field else "full"
            auc.setdefault((r["cond"], v, r["task"][0]), []).append(m["auc"][i])
    sc = load("scratch")
    log = sc["log"]
    n = budget // log["steps_per_iter"]
    m = metrics({k: (v[:, :n] if torch.is_tensor(v) and v.dim() > 1 else v) for k, v in log.items()})
    for i, r in enumerate(sc["runs"]):
        auc.setdefault(("scratch", "scratch", r["task"][0]), []).append(m["auc"][i])
    return auc


def fig_mech(srcs=("enter_right", "open_door", "key_right", "pick_key", "face_door", "ball_near_box", "impact", "mix")):
    auc = mech_table()
    tg = [t for t in MECH_TGT if t != "open_with_ball"]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    w = 0.13
    sc = np.mean([np.mean(auc[("scratch", "scratch", t)]) for t in tg])
    for j, (v, lab) in enumerate(VARS):
        vals = [np.mean([np.mean(auc.get((c, v, t), [np.nan])) for t in tg]) for c in srcs]
        errs = [np.std([np.mean(auc.get((c, v, t), [np.nan])) for t in tg]) / np.sqrt(len(tg)) for c in srcs]
        ax.bar(np.arange(len(srcs)) + (j - 2.5) * w, vals, w * 0.92, color=SERIES[j], label=lab)
    ax.axhline(sc, color=INK, ls="--", lw=1.5, label=f"scratch ({sc:.2f})")
    ax.set_xticks(range(len(srcs)), srcs)
    ax.set_ylabel(f"mean success over 6M finetune steps\n(avg over {', '.join(tg)})")
    ax.set_title("Mechanism decomposition: which part of the practiced agent carries (or blocks) learnability?")
    ax.legend(fontsize=7.5, ncol=4, loc="upper right")
    ax.set_ylim(0, 1.15)
    fig.savefig(os.path.join(FIG, "fig_mech.png"), dpi=150, bbox_inches="tight")
    return auc


PRE_STEPS = [50_000, 100_000, 250_000, 500_000, 1_000_000]
CK_SRC = ["pick_key", "open_door", "enter_right", "face_door", "mix", "impact"]


def ckpt_table():
    out = {}
    d = load("ckpt")
    m = metrics(d["log"])
    for i, r in enumerate(d["runs"]):
        out.setdefault(r["cond"], {}).setdefault(r["pre_step"], {}).setdefault(r["task"][0], []).append(m["auc"][i])
    d = load("finetune")
    m = metrics(d["log"])
    for i, r in enumerate(d["runs"]):
        if r["cond"] in out and r["task"][0] in MECH_TGT:
            out[r["cond"]].setdefault(2_000_000, {}).setdefault(r["task"][0], []).append(m["auc"][i])
    sc = mech_table()
    for c in out:
        out[c][0] = {t: sc[("scratch", "scratch", t)] for t in MECH_TGT}
    return out


def competence(cond):
    d = load("pretrain")
    sr, sm = curves(d["log"])
    ids = [i for i, r in enumerate(d["runs"]) if r["cond"] == cond]
    spi = d["log"]["steps_per_iter"]
    return {st: sm[ids, max(st // spi - 1, 0)].mean() for st in [0] + PRE_STEPS + [2_000_000]}


def fig_ckpt(tg=("goal", "key_near_goal", "close_from_right")):
    out = ckpt_table()
    probe = load("probe")
    from gridworld import PREDICATES
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    steps = [0] + PRE_STEPS + [2_000_000]
    xs = np.array([25_000] + PRE_STEPS + [2_000_000])
    for j, c in enumerate(CK_SRC):
        y = [np.mean([np.mean(out[c][st][t]) for t in tg]) for st in steps]
        axs[0].plot(xs, y, color=SERIES[j], marker="o", ms=4, label=c)
        comp = competence(c)
        if c not in ("impact",):
            axs[1].plot(xs, [comp[st] for st in steps], color=SERIES[j], marker="o", ms=4, label=c)
        p0 = []
        for st in steps:
            if st == 0:
                e = probe["fresh"]
                p0.append(np.mean([e["stoch"]["hit"][:, PREDICATES.index(t)].mean().item() for t in tg]))
            else:
                e = probe["pre_final.pt" if st == 2_000_000 else f"pre_{st}.pt"]
                ids = [i for i, r in enumerate(e["runs"]) if r["cond"] == c]
                p0.append(np.mean([e["stoch"]["hit"][ids, PREDICATES.index(t)].mean().item() for t in tg]))
        axs[2].plot(xs, np.maximum(p0, 1e-5), color=SERIES[j], marker="o", ms=4, label=c)
    axs[0].set_title("Learnability on targets (finetune AUC)")
    axs[1].set_title("Competence on the practice task itself")
    axs[2].set_title("Zero-shot stochastic hit prob. p0 on targets")
    axs[2].set_yscale("log")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("pretraining steps (log; leftmost = no pretraining)")
    axs[0].legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "fig_ckpt.png"), dpi=150, bbox_inches="tight")
    return out


def fig_decomp(conds=("scratch", "enter_right", "open_door", "key_right", "close_door", "pick_key", "mix", "impact", "face_door"),
               tg=("goal", "key_near_goal", "close_from_right", "goal_with_key"), name="reps"):
    rows = {}
    for nm in ([name] if name else []) + ["finetune"]:
        d = load(nm)
        m = metrics(d["log"])
        for i, r in enumerate(d["runs"]):
            c = "scratch" if r.get("variant") == "scratch" else r["cond"]
            if r["task"][0] in tg and r.get("variant", "full") in ("full", "scratch"):
                rows.setdefault(c, []).append((m["disc"][i], m["t80"][i]))
    sc = load("scratch")
    n = 6_000_000 // sc["log"]["steps_per_iter"]
    m = metrics({k: (v[:, :n] if torch.is_tensor(v) and v.dim() > 1 else v) for k, v in sc["log"].items()})
    for i, r in enumerate(sc["runs"]):
        if r["task"][0] in tg:
            rows.setdefault("scratch", []).append((m["disc"][i], m["t80"][i]))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    cap = 7e6
    rng = np.random.default_rng(0)
    for j, c in enumerate(conds):
        v = np.array(rows[c])
        d, t = v[:, 0], v[:, 1]
        col = INK if c == "scratch" else SERIES[(j - 1) % 8]
        jit = rng.uniform(-0.25, 0.25, len(d))
        axs[0].scatter(j + jit, np.minimum(d, cap) / 1e6, s=10, color=col, alpha=0.6)
        axs[0].plot([j - .3, j + .3], [np.median(np.minimum(d, cap)) / 1e6] * 2, color=col, lw=3)
        ok = np.isfinite(t)
        axs[1].scatter(j + jit[ok], (t - d)[ok] / 1e6, s=10, color=col, alpha=0.6)
        if ok.any():
            axs[1].plot([j - .3, j + .3], [np.median((t - d)[ok]) / 1e6] * 2, color=col, lw=3)
        axs[2].barh(j, ok.mean(), color=col)
        axs[2].text(ok.mean() + 0.01, j, f"{ok.sum()}/{len(ok)}", va="center", fontsize=7, color=INK2)
    for ax in axs[:2]:
        ax.set_xticks(range(len(conds)), conds, rotation=40, ha="right")
    axs[0].axhline(6, color=MUTED, lw=1, ls=":")
    axs[0].set_title("Discovery: steps to first success\n(dots at top = never within 6M)")
    axs[0].set_ylabel("M env steps")
    axs[1].set_title("Consolidation: first success -> 80% success\n(learned runs only)")
    axs[2].set_yticks(range(len(conds)), conds)
    axs[2].invert_yaxis()
    axs[2].set_xlim(0, 1.15)
    axs[2].set_title("P(learned within 6M)")
    fig.suptitle(f"Targets: {', '.join(tg)}", fontsize=9, color=INK2, y=1.06)
    fig.savefig(os.path.join(FIG, "fig_decomp.png"), dpi=150, bbox_inches="tight")
    return rows


def cost_table(K=6_000_000, pre=2_000_000, tg=("goal", "goal_with_key", "key_near_goal", "close_from_right"), name="finetune"):
    d = load(name)
    m = metrics(d["log"])
    cost = {}
    for i, r in enumerate(d["runs"]):
        if r["task"][0] in tg:
            cost.setdefault(r["cond"], []).append(min(m["t80"][i], K))
    sc = load("scratch")
    n = K // sc["log"]["steps_per_iter"]
    ms = metrics({k: (v[:, :n] if torch.is_tensor(v) and v.dim() > 1 else v) for k, v in sc["log"].items()})
    base = np.mean([min(ms["t80"][i], K) for i, r in enumerate(sc["runs"]) if r["task"][0] in tg])
    out = {}
    for c, v in cost.items():
        saving = base - np.mean(v)
        out[c] = {"cost_per_task": np.mean(v), "saving": saving, "breakeven": pre / saving if saving > 0 else np.inf}
    return base, out


def scratch_metrics(budget=6_000_000):
    sc = load("scratch")
    n = budget // sc["log"]["steps_per_iter"]
    return sc, metrics({k: (v[:, :n] if torch.is_tensor(v) and v.dim() > 1 else v) for k, v in sc["log"].items()})


def fig_residual(tg=("goal", "key_near_goal", "close_from_right", "open_with_ball")):
    sc, ms = scratch_metrics()
    rd = load("residual")
    mr = metrics(rd["log"])
    ft = load("finetune")
    mf = metrics(ft["log"])
    groups = [("scratch (base world)", sc, ms, lambda r, t: r["task"][0] == t),
              ("env gives KEY (residual)", rd, mr, lambda r, t: r["task"][0] == t and r["world"] == "key"),
              ("practiced pick_key (habit)", ft, mf, lambda r, t: r["task"][0] == t and r["cond"] == "pick_key"),
              ("env opens DOOR (residual)", rd, mr, lambda r, t: r["task"][0] == t and r["world"] == "open"),
              ("practiced open_door (habit)", ft, mf, lambda r, t: r["task"][0] == t and r["cond"] == "open_door")]
    cols = [INK, SERIES[0], "#9ec5f4", SERIES[1], "#f3b49a"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.16
    for j, (lab, d, m, f) in enumerate(groups):
        vals = [np.mean([m["auc"][i] for i, r in enumerate(d["runs"]) if f(r, t)]) for t in tg]
        ns = [sum(1 for r in d["runs"] if f(r, t)) for t in tg]
        ax.bar(np.arange(len(tg)) + (j - 2) * w, vals, w * 0.92, color=cols[j], label=lab)
    ax.set_xticks(range(len(tg)), tg)
    ax.set_ylabel("mean success over 6M steps (AUC)")
    ax.set_title("Residual-chain test: does a practiced habit act like the environment solving the prefix?")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_ylim(0, 1.05)
    fig.savefig(os.path.join(FIG, "fig_residual.png"), dpi=150, bbox_inches="tight")


def fig_collapse(tasks=("open_with_ball", "pick_ball2", "goal")):
    from gridworld import PREDICATES
    sc = load("scratch")
    log = sc["log"]
    spi = log["steps_per_iter"]
    win = 25
    fig, axs = plt.subplots(1, len(tasks), figsize=(14, 3.6), sharey=False)
    for ax, t in zip(axs, tasks):
        ids = [i for i, r in enumerate(sc["runs"]) if r["task"][0] == t]
        ep = log["episodes"][ids].sum(0).numpy()
        k = np.ones(win)
        sm = lambda x: np.convolve(x, k, mode="valid") / np.maximum(np.convolve(ep, k, mode="valid"), 1)
        x = (np.arange(len(ep) - win + 1) + win) * spi / 1e6
        ax.plot(x, sm(log["success"][ids].sum(0).numpy()), color=INK, label="success (the task)")
        for j, e in enumerate(["pick_key", "open_door", "pick_ball"]):
            ax.plot(x, sm(log["hits"][ids, :, PREDICATES.index(e)].sum(0).numpy()), color=SERIES[j], lw=1.5, label=f"event: {e}")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.2)
        ax.set_title(f"scratch PPO on {t}")
        ax.set_xlabel("env steps (M)")
    axs[0].set_ylabel("fraction of episodes (log)")
    axs[0].legend(fontsize=7.5, loc="lower left")
    fig.savefig(os.path.join(FIG, "fig_collapse.png"), dpi=150, bbox_inches="tight")


MG_TYPE = np.array([1, 2, 4, 5, 6, 7, 8, 0])
VW, VH, FPS = 1280, 940, 12
rgb = lambda c: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
GREEN, RED, BLUE, TRACK = rgb("#008300"), rgb("#e34948"), rgb("#2a78d6"), rgb("#e2e1dc")
GOAL_EV = (["pick_key", "open_door", "enter_right", "goal"], ["Key", "Door open", "Right room", "Goal"])


@functools.lru_cache(None)
def font(size, bold=False):
    from PIL import ImageFont
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", f"DejaVuSans{'-Bold' if bold else ''}.ttf"), size)


def wrap(dr, text, f, width):
    out, cur = [], ""
    for w in text.split(" "):
        if cur and dr.textlength(cur + " " + w, font=f) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    return out + [cur]


def render_env(world, b, S, tile=32):
    from minigrid.core.grid import Grid
    from gridworld import PAD
    g = world.grid[b, PAD:PAD + S, PAD:PAD + S].cpu().numpy().astype(int)
    arr = np.stack([MG_TYPE[g & 7], (g >> 3) & 7, g >> 6], -1).transpose(1, 0, 2)
    grid, _ = Grid.decode(arr)
    pos = (world.pos[b] - PAD).cpu().numpy()
    view = (world.pos[b][None] + world.offsets[world.dir[b]] - PAD).cpu().numpy()
    hl = np.zeros((S, S), bool)
    ok = (view >= 0).all(1) & (view < S).all(1)
    hl[view[ok, 0], view[ok, 1]] = True
    img = grid.render(tile, tuple(int(v) for v in pos), int(world.dir[b]), highlight_mask=hl)
    strip = np.full((tile, img.shape[1], 3), 40, np.uint8)
    c = int(world.carry[b])
    if c:
        from minigrid.core.world_object import WorldObj
        strip[:, :tile] = Grid.render_tile(WorldObj.decode(int(MG_TYPE[c & 7]), (c >> 3) & 7, 0), tile_size=tile)
    return np.concatenate([img, strip])


def mg_render(env, tile=40):
    from minigrid.core.grid import Grid
    u = env.unwrapped
    img = u.get_frame(True, tile, False)
    strip = np.full((tile, img.shape[1], 3), 40, np.uint8)
    if u.carrying is not None:
        strip[:, :tile] = Grid.render_tile(u.carrying, tile_size=tile)
    return np.concatenate([img, strip])


@torch.no_grad()
def gw_rollout(params, task, events, layout, max_steps=200, S=10, variant=0, eps=0.0):
    import learn
    from gridworld import PREDICATES, World, featurize, feature_lut
    R = len(params)
    lut = feature_lut("cuda")
    world = World(torch.full((R,), learn.TASK[task]), torch.full((R,), S), torch.full((R,), max_steps), "cuda", seed=0, desync=False,
                  variant=torch.full((R,), variant))
    p = {k: torch.stack([q[k] for q in params]) for k in learn.KEYS}
    gen = torch.Generator(device="cuda").manual_seed(layout)
    idx = world.bank_start + torch.randint(0, world.bank_n, (1,), generator=gen, device="cuda")
    for k in ["grid", "pos", "dir", "wx", "door"]:
        getattr(world, k)[:] = world.bank[k][idx]
    world.carry.zero_(); world.t.zero_(); world.hit.zero_()
    ev = [PREDICATES.index(e) for e in events]
    status, ts, out = ["run"] * R, [0] * R, []
    for _ in range(max_steps + 1):
        logits, v, _ = learn.forward(p, featurize(world.obs(), lut).view(R, 1, -1))
        hit = world.hit[:, ev].cpu().numpy()
        out.append([dict(img=render_env(world, b, S), ev=hit[b], v=float(v[b, 0]), t=ts[b], status=status[b]) for b in range(R)])
        if "run" not in status:
            break
        # one shared Gumbel draw per step: agents with identical actors take identical actions
        a = (logits[:, 0] - torch.log(-torch.log(torch.rand(logits.shape[-1], generator=gen, device="cuda")))).argmax(-1)
        if eps > 0:
            u, ua = torch.rand(1, generator=gen, device="cuda"), torch.randint(0, 6, (1,), generator=gen, device="cuda")
            a = torch.where(u < eps, ua.expand(R), a)
        saved = {k: getattr(world, k).clone() for k in ["grid", "pos", "dir", "carry", "t", "hit"]}
        _, _, d, info = world.step(a, reset=False)
        for b in range(R):
            if status[b] != "run":
                for k in saved:
                    getattr(world, k)[b] = saved[k][b]
            else:
                ts[b] += 1
                if d[b]:
                    status[b] = "win" if info["success"][b] else "end" if task == "null" else "fail"
    return out


@torch.no_grad()
def mg_rollout(params, env_seed, max_steps=200):
    import gymnasium as gym
    import minigrid  # noqa: F401
    import learn
    envs = [gym.make("MiniGrid-DoorKey-8x8-v0") for _ in params]
    obs = [e.reset(seed=env_seed)[0] for e in envs]
    gen = torch.Generator().manual_seed(env_seed)
    R = len(params)
    status, ts, ev, out = ["run"] * R, [0] * R, np.zeros((R, 3), bool), []
    for _ in range(max_steps + 1):
        step, logits = [], []
        for i, e in enumerate(envs):
            u = e.unwrapped
            ev[i] |= [u.carrying is not None and u.carrying.type == "key",
                      any(o is not None and o.type == "door" and o.is_open for o in u.grid.grid), status[i] == "win"]
            lg, v, _ = learn.forward(params[i], learn.mg_obs(obs[i]["image"][None]))
            step.append(dict(img=mg_render(e), ev=ev[i].copy(), v=float(v[0, 0]), t=ts[i], status=status[i]))
            logits.append(lg[0, 0])
        out.append(step)
        if "run" not in status:
            break
        g = -torch.log(-torch.log(torch.rand(len(logits[0]), generator=gen)))
        for i, e in enumerate(envs):
            if status[i] == "run":
                obs[i], r, te, tr, _ = e.step(int((logits[i] + g).argmax()))
                ts[i] += 1
                if te or tr:
                    status[i] = "win" if r > 0 else "fail"
    for s in out[-1]:
        s["status"] = "cut" if s["status"] == "run" else s["status"]
    return out


@torch.no_grad()
def mg_eval(params, n=32):
    import gymnasium as gym
    import learn
    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    torch.manual_seed(0)
    win = door = 0
    for ep in range(n):
        obs, _ = env.reset(seed=5000 + ep)
        opened = False
        while True:
            logits = learn.forward(params, learn.mg_obs(obs["image"][None]))[0][0, 0]
            obs, r, te, tr, _ = env.step(torch.distributions.Categorical(logits=logits).sample().item())
            opened |= any(o is not None and o.type == "door" and o.is_open for o in env.unwrapped.grid.grid)
            if te or tr:
                win += r > 0
                door += opened
                break
    return win / n, door / n


def chart(panels, cursors, xmax, xlabel):
    with plt.rc_context({"font.size": 11, "axes.titlesize": 11.5, "axes.titleweight": "bold"}):
        fig, axs = plt.subplots(1, len(panels), figsize=(12.2, 2.9), dpi=100, squeeze=False)
        for ax, (ttl, ylab, ylim, log, series) in zip(axs[0], panels):
            for lab, col, x, Y in series:
                for y in Y if len(Y) > 1 else []:
                    ax.plot(x, y, color=col, lw=0.7, alpha=0.3)
                ax.plot(x, Y.mean(0), color=col, lw=2.6, label=lab)
            if log:
                ax.set_yscale("log")
            ax.set_xlim(0, xmax)
            ax.set_ylim(*ylim)
            for c in cursors:
                ax.axvline(c, color=INK, lw=1.3, ls="--")
            ax.text(cursors[-1], ylim[1], " video position", va="top", ha="left", fontsize=10, color=INK)
            ax.set_title(ttl, loc="left")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylab)
        hl = {lb: h for ax in axs[0] for h, lb in zip(*ax.get_legend_handles_labels())}
        fig.legend(hl.values(), hl.keys(), loc="center left", bbox_to_anchor=(0.71, 0.5), fontsize=9.3)
        fig.subplots_adjust(left=0.06, right=0.70, bottom=0.2, top=0.87, wspace=0.22)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
    return img


def scene(steps, labels, colors, subs, title, hint, stage, events, show_v, chart_img, fast_after=30, k=5):
    from PIL import Image, ImageDraw

    def draw(st, ff):
        cv = Image.new("RGB", (VW, VH), rgb(SURF))
        dr = ImageDraw.Draw(cv)
        w = dr.textlength(stage, font=font(20, True))
        dr.rounded_rectangle((VW - 60 - w, 18, VW - 40, 52), 8, fill=rgb(INK))
        dr.text((VW - 50 - w, 23), stage, fill=(255, 255, 255), font=font(20, True))
        sz = 28
        while dr.textlength(title, font=font(sz, True)) > VW - 120 - w:
            sz -= 1
        dr.text((40, 18), title, fill=rgb(INK), font=font(sz, True))
        for j, part in enumerate(wrap(dr, hint, font(18, True), VW - 240)[:2]):
            dr.text((40, 62 + 24 * j), part, fill=BLUE, font=font(18, True))
        if ff:
            dr.text((VW - 140, 62), f"Fast ×{k}", fill=rgb(INK2), font=font(17, True))
        pw = st[0]["img"].shape[1]
        gap = 90 if len(st) < 3 else 60
        x = (VW - len(st) * pw - (len(st) - 1) * gap) // 2
        for s, lab, col, sub in zip(st, labels, colors, subs):
            dr.rectangle((x, 124, x + 14, 138), fill=rgb(col))
            dr.text((x + 22, 120), lab, fill=rgb(INK), font=font(17, True))
            dr.text((x, 146), sub, fill=rgb(INK2), font=font(14))
            cv.paste(Image.fromarray(s["img"]), (x, 170))
            y, cx = 180 + s["img"].shape[0], x
            for name, on in zip(events, s["ev"]):
                tw = dr.textlength(name, font=font(14))
                dr.rounded_rectangle((cx, y, cx + tw + 14, y + 25), 6, fill=GREEN if on else None, outline=GREEN if on else rgb(MUTED))
                dr.text((cx + 7, y + 4), name, fill=(255, 255, 255) if on else rgb(MUTED), font=font(14))
                cx += tw + 20
            y += 34
            if show_v:
                dr.text((x, y), f"Critic value {s['v']:.2f}", fill=rgb(INK2), font=font(14))
                bx = x + 160
                dr.rounded_rectangle((bx, y + 5, x + pw, y + 17), 4, fill=TRACK)
                dr.rounded_rectangle((bx, y + 5, bx + max(6, int((x + pw - bx) * min(max(s["v"], 0), 1))), y + 17), 4, fill=BLUE)
                y += 26
            msg, c = {"run": (f"Step {s['t']}", rgb(INK)), "win": (f"Step {s['t']}: SUCCESS", GREEN),
                      "fail": (f"Step {s['t']}: time out, FAIL", RED), "cut": (f"Not done in {s['t']} steps", RED),
                      "end": (f"Step {s['t']}: end of episode", rgb(INK))}[s["status"]]
            dr.text((x, y), msg, fill=c, font=font(17, True))
            x += pw + gap
        cv.paste(Image.fromarray(chart_img), ((VW - chart_img.shape[1]) // 2, VH - chart_img.shape[0] - 6))
        return np.array(cv)

    keep = [i for i in range(len(steps)) if i < fast_after or i % k == 0 or i == len(steps) - 1]
    frames = [draw(steps[i], i >= fast_after) for i in keep]
    return frames + [frames[-1]] * int(1.5 * FPS)


def write_video(frames, path, fps=FPS):
    import imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def videos(layout=1, mg_seed=0):
    import learn
    out = os.path.join(ROOT, "videos")
    d = load("demo2")
    lg = d["log"]
    m = metrics(lg)
    _, sm = curves(lg)
    spi = lg["steps_per_iter"]
    x = (np.arange(sm.shape[1]) + 1) * spi / 1e6
    groups = {}
    for i, r in enumerate(d["runs"]):
        groups.setdefault((r["cond"], r["variant"]), []).append(i)
    med = {g: ids[int(np.argsort(m["auc"][ids])[len(ids) // 2])] for g, ids in groups.items()}
    zero = {g: {k: v[0].detach() for k, v in learn.build_params([d["runs"][i]]).items()} for g, i in med.items()}
    ck = {K: torch.load(os.path.join(learn.CKPT, f"demo2_{K}.pt"), map_location="cuda")["params"]
          for K in (245_760, 491_520, 1_998_848, 2_998_272)}
    agent = lambda g, K: {k: ck[K][k][med[g]] for k in learn.KEYS} if K else zero[g]
    lab = {("none", "scratch"): "From scratch", ("open_door", "full"): "After open_door practice",
           ("face_door", "full"): "After face_door practice", ("face_door", "nocritic"): "face_door practice, critic reset"}
    col = {("none", "scratch"): INK2, ("open_door", "full"): SERIES[0], ("face_door", "full"): SERIES[1], ("face_door", "nocritic"): SERIES[2]}
    learned = {g: f"{int(np.isfinite(m['t80'][ids]).sum())}/{len(ids)}" for g, ids in groups.items()}
    print("learned within 4M:", learned, "median runs:", med)
    stage = lambda K: "Zero-shot (0 steps)" if K == 0 else f"After {round(K / 5e4) / 20:g}M steps on goal"
    sub = lambda g, K: "Not trained on goal" if K == 0 else f"Success in training now: {sm[med[g], K // spi - 1]:.0%}"
    xl = "Training steps on goal (M)"
    succ = lambda gs, lb=lab: ("Success rate (thin = one seed, thick = mean)", "Success rate", (0, 1.05), False,
                               [(f"{lb[g]}: {learned[g]} learn", col[g], x, sm[groups[g]]) for g in gs])

    g1 = [("none", "scratch"), ("open_door", "full"), ("face_door", "full")]
    t1 = "Practice can make an agent learn faster or slower."
    ch1 = []
    hints1 = {0: "No agent goes to the goal. The open_door agent already opens the door. The face_door agent goes to the door and stops.",
              491_520: "The open_door agent can do the task now. The other two agents still have 0% success.",
              2_998_272: "The agent from scratch can do the task now. The face_door agent still cannot."}
    for K in hints1:
        st = gw_rollout([agent(g, K) for g in g1], "goal", GOAL_EV[0], layout)
        print("ch1", K, [(s["status"], s["t"], s["ev"].tolist()) for s in st[-1]])
        ch1 += scene(st, [lab[g] for g in g1], [col[g] for g in g1], [sub(g, K) for g in g1], t1, hints1[K], stage(K),
                     GOAL_EV[1], False, chart([succ(g1)], [K / 1e6], 4.0, xl))

    g2 = [("face_door", "full"), ("face_door", "nocritic")]
    l2 = ["face_door practice, full transfer", "face_door practice, critic reset"]
    lab2 = dict(zip(g2, l2)) | {("none", "scratch"): lab[("none", "scratch")]}
    t2 = "The old critic stops learning."
    ch2 = []
    ent = ("Policy entropy (max 1.79)", "Entropy", (0, 1.85), False,
           [(f"{lab2[g]}: {learned[g]} learn", col[g], x, lg["entropy"][groups[g]].numpy()) for g in g2])
    hints2 = {0: "The actions are the same. Only the critic is different: left expects a high return, right expects 0.",
              245_760: "Left repeats the old habit: it stands at the door (low entropy). Right explores the room.",
              1_998_848: "Right can do the task now. Left still cannot."}
    for K in hints2:
        st = gw_rollout([agent(g, K) for g in g2], "goal", GOAL_EV[0], layout)
        print("ch2", K, [(s["status"], s["t"], s["ev"].tolist()) for s in st[-1]], "V0", [round(s["v"], 2) for s in st[0]])
        ch2 += scene(st, l2, [col[g] for g in g2], [sub(g, K) for g in g2], t2, hints2[K], stage(K), GOAL_EV[1], True,
                     chart([succ(g2 + [("none", "scratch")], lab2), ent], [K / 1e6], 4.0, xl))

    mg_keys = [("scratch", "From scratch", INK2), ("unlock-full", "Unlock practice, full transfer", SERIES[1]),
               ("unlock-reset", "Unlock practice, critic reset", SERIES[2])]
    pre = torch.load(os.path.join(learn.CKPT, f"mg_unlock_{mg_seed}.pt"))
    torch.manual_seed(mg_seed + 100)
    fresh = {k: v.cpu() for k, v in learn.fresh_params_dim(1, 980).items()}
    mg_zero = {"scratch": fresh, "unlock-full": {k: pre[k].clone() for k in learn.KEYS},
               "unlock-reset": {k: (pre[k].clone() if k in learn.KEYS[:6] else fresh[k]) for k in learn.KEYS}}
    mg_ag = lambda key, K: mg_zero[key] if K == 0 else torch.load(os.path.join(learn.CKPT, f"mg_demo_{key}_{mg_seed}_{K}.pt"))
    mgt = mg_table()
    spu = 16 * 128
    mg_curve = lambda l: np.convolve(l[:, 1].numpy(), np.ones(10), "same") / np.maximum(np.convolve(l[:, 0].numpy(), np.ones(10), "same"), 1)
    mg_series = [(f"{lb} (10 seeds)", c, (np.arange(len(mgt[key][0])) + 1) * spu / 1e6, np.array([mg_curve(l) for l in mgt[key]]))
                 for key, lb, c in mg_keys]
    t3 = "The result is the same in the real MiniGrid."
    ch3 = []
    hints3 = {0: lambda e: f"The two Unlock actors are the same. They open the door in {e[1][1]:.0%} of the episodes, but they do not go to the goal.",
              102400: lambda e: f"Success in the test: critic reset {e[2][0]:.0%}, full transfer {e[1][0]:.0%}, from scratch {e[0][0]:.0%}.",
              307200: lambda e: f"Full transfer can do the task now ({e[1][0]:.0%}). From scratch: {e[0][0]:.0%}."}
    for K in hints3:
        ags = [mg_ag(key, K) for key, _, _ in mg_keys]
        st = mg_rollout(ags, env_seed=2000)
        evs = [mg_eval(a) for a in ags]
        print("ch3", K, [(s["status"], s["t"], s["ev"].tolist()) for s in st[-1]], "eval", evs, "V0", [round(s["v"], 2) for s in st[0]])
        ch3 += scene(st, [lb for _, lb, _ in mg_keys], [c for _, _, c in mg_keys], [f"Test (32 episodes): success {e[0]:.0%}, door {e[1]:.0%}" for e in evs],
                     t3, hints3[K](evs), "Zero-shot (0 steps)" if K == 0 else f"After {K / 1e6:.1f}M steps on DoorKey",
                     ["Key", "Door open", "Goal"], True,
                     chart([("Success rate (thin = one seed, thick = mean of 10)", "Success rate", (0, 1.05), False, mg_series)],
                           [K / 1e6], 1.0, "Training steps on DoorKey-8x8 (M)"))

    sc = load("scratch")
    ids = [i for i, r in enumerate(sc["runs"]) if r["task"][0] == "open_with_ball"]
    from gridworld import PREDICATES
    win = 25
    ep = sc["log"]["episodes"][ids].sum(0).numpy()
    frac = lambda v: np.convolve(v, np.ones(win), "valid")[None] / np.maximum(np.convolve(ep, np.ones(win), "valid"), 1)[None]
    xs = (np.arange(len(ep) - win + 1) + win) * sc["log"]["steps_per_iter"] / 1e6
    ap_ev = (["pick_key", "open_door", "pick_ball", "open_with_ball"], ["Key", "Door open", "Ball", "Success"])
    ap_series = [(nm, c, xs, frac(sc["log"]["hits"][ids, :, PREDICATES.index(e)].sum(0).numpy()))
                 for e, nm, c in zip(ap_ev[0][:3], ap_ev[1][:3], SERIES[:3])]
    ap_series.append(("Success (the task)", INK, xs, frac(sc["log"]["success"][ids].sum(0).numpy())))
    t4 = "Without practice, more training makes the agent touch fewer objects."
    ap = []
    Ks = [100_000, 1_000_000, 2_000_000]
    ags = [ck_slice(f"demo_{K}.pt", "none", 0, "open_with_ball") for K in Ks]
    torch.manual_seed(0)
    pr = learn.probe({k: torch.stack([a[k] for a in ags]) for k in learn.KEYS}, len(Ks), E=1024, n_eps=2)
    hp = lambda i, e: float(pr["hit"][i, PREDICATES.index(e)])
    ap_sub = [f"Key {hp(i, 'pick_key'):.0%}, ball {hp(i, 'pick_ball'):.0%}, door open {hp(i, 'open_door'):.1%}" for i in range(len(Ks))]
    print("appendix probe", ap_sub, pr["pickups"].tolist())
    ap_chart = chart([("Fraction of episodes with the event (4 seeds from scratch, log scale)", "Fraction", (1e-4, 1.5), True, ap_series)],
                     [K / 1e6 for K in Ks], 10.0, "Training steps on open_with_ball (M)")
    for lay in [layout, layout + 1]:
        st = gw_rollout(ags, "open_with_ball", ap_ev[0], lay)
        print("appendix", lay, [(s["status"], s["ev"].tolist()) for s in st[-1]])
        ap += scene(st, [f"After {K / 1e6:g}M steps" for K in Ks], [SERIES[3], SERIES[4], SERIES[6]], ap_sub, t4,
                    f"In a test of 2048 episodes, the agent picks up {pr['pickups'][0]:.1f} objects per episode at 0.1M and {pr['pickups'][2]:.1f} at 2M.",
                    "From scratch, no practice", ap_ev[1], False, ap_chart)

    for f in os.listdir(out) if os.path.isdir(out) else []:
        if f[0] in "01234":
            os.remove(os.path.join(out, f))
    write_video(ch1 + ch2 + ch3, os.path.join(out, "0_skills_help_wrong_expectations_stop_learning.mp4"))
    write_video(ch1, os.path.join(out, "1_practice_can_make_an_agent_learn_faster_or_slower.mp4"))
    write_video(ch2, os.path.join(out, "2_the_old_critic_stops_learning.mp4"))
    write_video(ch3, os.path.join(out, "3_the_result_is_the_same_in_the_real_minigrid.mp4"))
    write_video(ap, os.path.join(out, "4_without_practice_more_training_makes_the_agent_touch_fewer_objects.mp4"))


def ck_slice(file, cond, pseed=0, task=None):
    import learn
    ck = torch.load(os.path.join(learn.CKPT, file), map_location="cuda")
    i = next(i for i, r in enumerate(ck["runs"]) if r.get("cond") == cond and r.get("pseed", 0) == pseed and
             (task is None or r["task"][0] == task))
    return {k: ck["params"][k][i] for k in learn.KEYS}


def fig_optimism(tg=("goal", "key_near_goal", "close_from_right")):
    op = load("optimism")
    mo = metrics(op["log"])
    auc = mech_table()
    cr = load("critic")
    mc = metrics(cr["log"])
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1.7]})
    vals, labs = [np.mean([np.mean(auc[("scratch", "scratch", t)]) for t in tg])], ["V init 0\n(standard)"]
    for vi in [0.3, 0.8]:
        vals.append(np.mean([mo["auc"][[i for i, r in enumerate(op["runs"]) if r.get("variant2") == f"vinit{vi}" and r["task"][0] == t]].mean() for t in tg]))
        labs.append(f"V init {vi}")
    axs[0].bar(range(3), vals, 0.6, color=[INK, SERIES[3], SERIES[1]])
    for j, v in enumerate(vals):
        axs[0].text(j, v + 0.02, f"{v:.2f}", ha="center", color=INK2, fontsize=8)
    axs[0].set_xticks(range(3), labs)
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel("scratch AUC (6M)")
    axs[0].set_title("Scratch agent with an optimistic constant value")
    srcs = ["pick_key", "face_door", "mix", "open_door", "enter_right"]
    var = [("full", "stale critic (full)", SERIES[0]), ("center", "stale critic, mean-centered", SERIES[4]),
           ("warmup", "stale critic + 200k critic warm-up", SERIES[6]), ("detach", "stale critic, no grad to trunk", SERIES[7]),
           ("nocritic", "reset critic head", SERIES[2])]
    w = 0.16
    for j, (v, lab, col) in enumerate(var):
        ys = []
        for c in srcs:
            if v == "center":
                ys.append(np.mean([mo["auc"][[i for i, r in enumerate(op["runs"]) if r["cond"] == c and r["task"][0] == t and r["variant"] == "center"]].mean() for t in tg]))
            elif v in ("warmup", "detach"):
                ids = [[i for i, r in enumerate(cr["runs"]) if r["cond"] == c and r["task"][0] == t and r.get("variant2") == v] for t in tg]
                ys.append(np.mean([mc["auc"][x].mean() for x in ids]) if min(map(len, ids)) > 0 else np.nan)
            else:
                ys.append(np.mean([np.mean(auc.get((c, v, t), [np.nan])) for t in tg]))
        axs[1].bar(np.arange(len(srcs)) + (j - 2) * w, ys, w * 0.92, color=col, label=lab)
    axs[1].axhline(vals[0], color=INK, ls="--", lw=1.2, label="scratch")
    axs[1].set_xticks(range(len(srcs)), srcs)
    axs[1].set_ylim(0, 1.1)
    axs[1].set_title("Which critic fix works? (practiced agents)")
    axs[1].legend(fontsize=7, ncol=3, loc="upper left")
    fig.savefig(os.path.join(FIG, "fig_optimism.png"), dpi=150, bbox_inches="tight")


NC_TGT = ["goal", "goal_with_key", "key_near_goal", "close_from_right", "open_with_ball", "pick_ball2"]


def nc_table():
    d = load("ncsweep")
    m = metrics(d["log"])
    out = {}
    for i, r in enumerate(d["runs"]):
        out.setdefault((r["cond"], r["variant"]), {}).setdefault(r["task"][0], []).append((m["auc"][i], m["t80"][i]))
    return out


def fig_ncsweep():
    out = nc_table()
    conds = sorted({c for c, v in out if c != "none"}, key=lambda c: -np.mean([x[0] for t in NC_TGT for x in out[(c, "nocritic")][t]]))
    sc = np.mean([x[0] for t in NC_TGT for x in out[("none", "scratch")][t]])
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(conds))
    full = [np.mean([x[0] for t in NC_TGT for x in out[(c, "full")][t]]) if (c, "full") in out else np.nan for c in conds]
    nc = [np.mean([x[0] for t in NC_TGT for x in out[(c, "nocritic")][t]]) for c in conds]
    axs[0].bar(x - 0.2, full, 0.38, color=SERIES[0], label="practice + transfer everything")
    axs[0].bar(x + 0.2, nc, 0.38, color=SERIES[2], label="practice + reset critic head")
    axs[0].axhline(sc, color=INK, ls="--", lw=1.5, label=f"scratch ({sc:.2f})")
    axs[0].set_xticks(x, conds, rotation=40, ha="right")
    axs[0].set_ylabel(f"mean AUC over {len(NC_TGT)} targets")
    axs[0].set_title("Same practice, different transfer: resetting expectations flips the sign")
    axs[0].legend(fontsize=8)
    Ks = np.arange(0, 6_000_001, 100_000)

    def front(key):
        return np.array([np.mean([np.mean([x[1] <= K for x in out[key][t]]) for t in NC_TGT]) for K in Ks])

    axs[1].plot(Ks / 1e6, front(("none", "scratch")), color=INK, ls="--", label="scratch")
    for j, c in enumerate(conds[:3]):
        axs[1].plot(Ks / 1e6, front((c, "nocritic")), color=SERIES[j], label=f"{c} + reset critic")
        if (c, "full") in out:
            axs[1].plot(Ks / 1e6, front((c, "full")), color=SERIES[j], lw=1, ls=":", label=f"{c} full")
    allnc = np.mean([front((c, "nocritic")) for c in conds], 0)
    allfull = np.mean([front((c, "full")) for c in conds if (c, "full") in out], 0)
    axs[1].plot(Ks / 1e6, allnc, color=SERIES[2], lw=3, alpha=0.5, label="avg over all practice, reset")
    axs[1].plot(Ks / 1e6, allfull, color=SERIES[0], lw=3, alpha=0.5, label="avg over all practice, full")
    axs[1].set_xlabel("finetune steps K (M)")
    axs[1].set_ylabel(f"frontier F(K) over {len(NC_TGT)} targets")
    axs[1].set_title("Learning frontier")
    axs[1].legend(fontsize=6.5)
    fig.savefig(os.path.join(FIG, "fig_ncsweep.png"), dpi=150, bbox_inches="tight")
    return out


def fig_predictors():
    from scipy.stats import spearmanr
    from gridworld import PREDICATES
    out = nc_table()
    pr = load("probe")["pre_final.pt"]
    pr2 = load("probe2")
    fresh = load("probe")["fresh"]["stoch"]["hit"].numpy().mean(0)
    if len(fresh) < len(PREDICATES):
        import learn
        torch.manual_seed(0)
        fresh = learn.probe(learn.fresh_params(4), 4, E=1024, n_eps=16)["hit"].numpy().mean(0)
    px = load("proxy")
    mp = metrics(px["log"])
    proxy = {}
    for i, r in enumerate(px["runs"]):
        proxy.setdefault((r["cond"], r["pseed"]), []).append(mp["auc"][i] if r["task"][0] != r["cond"] else np.nan)
    rows = []
    for e in [pr, pr2]:
        for j, r in enumerate(e["runs"]):
            c, ps = r["cond"], r["pseed"]
            y_nc = np.mean([x[0] for t in NC_TGT for x in out[(c, "nocritic")][t]])
            y_full = np.mean([x[0] for t in NC_TGT for x in out[(c, "full")][t]]) if (c, "full") in out else np.nan
            st, rp = e["stoch"], e["rep"]
            rows.append({"cond": c, "y_nc": y_nc, "y_full": y_full,
                         "practice difficulty (-log p0 random)": -np.log10(fresh[PREDICATES.index(c)]) if c in PREDICATES else np.nan,
                         "policy entropy": st["entropy"][j].item(), "cell coverage": st["cells"][j].item(),
                         "world-change novelty": st["novel"][j].item(), "interactions": st["interact"][j].item(),
                         "feature srank": rp["srank"][j].item(), "dormant frac": rp["dormant"][j].item(),
                         "proxy learnability": np.nanmean(proxy[(c, ps)]) if (c, ps) in proxy else np.nan,
                         "door-open rate (uses downstream info)": st["hit"][j, PREDICATES.index("open_door")].item()})
    feats = [k for k in rows[0] if k not in ("cond", "y_nc", "y_full")]
    fig, ax = plt.subplots(figsize=(9, 4))
    res = {}
    for j, (target, col, lab) in enumerate([("y_full", SERIES[0], "transfer everything"), ("y_nc", SERIES[2], "reset critic head")]):
        rhos = []
        for f in feats:
            xs = np.array([r[f] for r in rows])
            ys = np.array([r[target] for r in rows])
            ok = np.isfinite(xs) & np.isfinite(ys)
            rho, pv = spearmanr(xs[ok], ys[ok])
            rhos.append(rho)
            res[(target, f)] = (rho, pv, ok.sum())
        ax.barh(np.arange(len(feats)) + (j - 0.5) * 0.38, rhos, 0.36, color=col, label=lab)
    ax.set_yticks(range(len(feats)), feats)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Spearman rho with downstream learnability (per pretrained agent)")
    ax.set_title("Can we predict which practice helps, without the downstream tasks?")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.savefig(os.path.join(FIG, "fig_predictors.png"), dpi=150, bbox_inches="tight")
    return res, rows


def mg_table():
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, "mg", "ft_*.pt"))):
        d = torch.load(f)
        out.setdefault(d["job"].split(":")[1], []).append(d["log"].float())
    return out


def fig_mg():
    out = mg_table()
    order = [("scratch", "scratch", INK), ("unlock-full", "practiced Unlock, transfer all", SERIES[0]),
             ("unlock-reset", "practiced Unlock, reset critic", SERIES[2]), ("empty-full", "practiced Empty-8x8, transfer all", SERIES[1]),
             ("empty-reset", "practiced Empty-8x8, reset critic", SERIES[3])]
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    spu = 16 * 128
    res = {}
    for j, (c, lab, col) in enumerate(order):
        if c not in out:
            continue
        cv = []
        for l in out[c]:
            k = np.ones(10)
            cv.append(np.convolve(l[:, 1].numpy(), k, "same") / np.maximum(np.convolve(l[:, 0].numpy(), k, "same"), 1))
        cv = np.array(cv)
        x = (np.arange(cv.shape[1]) + 1) * spu / 1e6
        axs[0].plot(x, cv.mean(0), color=col, label=f"{lab} (n={len(cv)})")
        se = cv.std(0) / np.sqrt(len(cv))
        axs[0].fill_between(x, cv.mean(0) - se, cv.mean(0) + se, color=col, alpha=0.12, lw=0)
        auc = cv.mean(1)
        res[c] = auc
        axs[1].scatter(np.full(len(auc), j) + np.linspace(-0.12, 0.12, len(auc)), auc, color=col, s=18)
        axs[1].plot([j - 0.25, j + 0.25], [auc.mean()] * 2, color=col, lw=3)
    axs[0].set_xlabel("finetune env steps on MiniGrid-DoorKey-8x8-v0 (M)")
    axs[0].set_ylabel("success rate (training episodes)")
    axs[0].set_title("Replication on the real MiniGrid package (mean ± s.e. over seeds)")
    axs[0].legend(fontsize=7.5, loc="lower right")
    axs[1].set_xticks(range(len(order)), [o[0] for o in order], rotation=30, ha="right")
    axs[1].set_ylabel("AUC over 3M steps (each dot = seed)")
    axs[1].set_title("Per-seed learnability")
    fig.savefig(os.path.join(FIG, "fig_minigrid.png"), dpi=150, bbox_inches="tight")
    return res


def compute_budget():
    import glob
    total, runs, wall = 0, 0, 0.0
    for f in glob.glob(os.path.join(RES, "*.pt")):
        d = torch.load(f, weights_only=False)
        if isinstance(d, dict) and isinstance(d.get("log"), dict) and "success" in d["log"]:
            n, it = d["log"]["success"].shape
            total += n * it * d["log"]["steps_per_iter"]
            runs += n
            wall += d["log"].get("wall", 0.0)
    mg = sum(len(torch.load(f)["log"]) * 16 * 128 for f in glob.glob(os.path.join(RES, "mg", "*.pt")))
    return {"gpu_runs": runs, "gpu_env_steps": total, "gpu_wall_hours_summed": wall / 3600, "minigrid_cpu_env_steps": mg}


def fig_ckptnc(tg=("goal", "key_near_goal", "close_from_right"), conds=("enter_right", "open_door", "key_right", "close_door", "pick_key", "mix")):
    d = load("ckptnc")
    m = metrics(d["log"])
    nc = nc_table()
    sc = np.mean([np.mean([x[0] for x in nc[("none", "scratch")][t]]) for t in tg])
    steps = [0] + PRE_STEPS + [2_000_000]
    xs = np.array([25_000] + PRE_STEPS + [2_000_000])
    full = ckpt_table()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    res = {}
    for j, c in enumerate(conds):
        y = [sc]
        for st in PRE_STEPS:
            y.append(np.mean([m["auc"][[i for i, r in enumerate(d["runs"]) if r["cond"] == c and r["pre_step"] == st and r["task"][0] == t]].mean() for t in tg]))
        y.append(np.mean([np.mean([x[0] for x in nc[(c, "nocritic")][t]]) for t in tg]))
        comp = competence(c)
        res[c] = (y, [comp[st] for st in steps])
        axs[0].plot(xs, y, color=SERIES[j], marker="o", ms=4, label=c)
        if c in full:
            axs[0].plot(xs[1:], [np.mean([np.mean(full[c][st][t]) for t in tg]) for st in steps[1:]], color=SERIES[j], lw=1, ls=":")
        axs[1].plot(xs, [comp[st] for st in steps], color=SERIES[j], marker="o", ms=4, label=c)
    axs[0].axhline(sc, color=INK, ls="--", lw=1, label="scratch")
    axs[0].set_title("Learnability on held-out targets\n(solid: practice + reset critic, dotted: transfer all)")
    axs[0].set_ylabel("finetune AUC (6M)")
    axs[1].set_title("Competence on the practice task itself")
    axs[1].set_ylabel("practice success rate")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("pretraining steps (leftmost = none)")
    axs[0].legend(fontsize=7.5)
    fig.savefig(os.path.join(FIG, "fig_ckptnc.png"), dpi=150, bbox_inches="tight")
    return res


def fig_reponly(tg=("goal", "key_near_goal", "close_from_right")):
    d = load("reponly")
    m = metrics(d["log"])
    nc = nc_table()
    conds = ["enter_right", "open_door", "key_right", "close_door"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    sc = [x[0] for t in tg for x in nc[("none", "scratch")][t]]
    res = {"scratch": np.array(sc)}
    ax.axhline(np.mean(sc), color=INK, ls="--", lw=1.5, label=f"scratch (n={len(sc)})")
    for j, (v, lab, col) in enumerate([("trunk", "pretrained trunk, uniform policy, zero critic", SERIES[0]),
                                       ("perm", "same trunk weights randomly permuted", MUTED)]):
        for k, c in enumerate(conds):
            a = np.array([m["auc"][i] for i, r in enumerate(d["runs"]) if r["cond"] == c and r["variant"] == v and r["task"][0] in tg])
            res[(c, v)] = a
            ax.bar(k + (j - 0.5) * 0.38, a.mean(), 0.36, color=col, label=lab if k == 0 else None,
                   yerr=a.std() / np.sqrt(len(a)), ecolor=INK2, capsize=2)
    ax.set_xticks(range(len(conds)), [f"practiced {c}" for c in conds])
    ax.set_ylabel("finetune AUC (mean ± s.e.)")
    ax.set_title("Identical initial behavior (uniform policy, V=0), different learnability")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.set_ylim(0, 1)
    fig.savefig(os.path.join(FIG, "fig_reponly.png"), dpi=150, bbox_inches="tight")
    return res


def fig_sep(tg=("goal", "key_near_goal", "close_from_right")):
    d = load("sep")
    m = metrics(d["log"])
    srcs = ["pick_key", "face_door", "mix", "open_door", "enter_right"]
    var = [("full", "transfer everything", SERIES[0]), ("vhead", "reset critic head only", SERIES[3]),
           ("nocritic", "reset whole critic network", SERIES[2])]
    sc = np.array([m["auc"][i] for i, r in enumerate(d["runs"]) if r["variant"] == "scratch" and r["task"][0] in tg])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    res = {"scratch": sc}
    for j, (v, lab, col) in enumerate(var):
        ys, es = [], []
        for c in srcs:
            a = np.array([m["auc"][i] for i, r in enumerate(d["runs"]) if r["cond"] == c and r["variant"] == v and r["task"][0] in tg])
            res[(c, v)] = a
            ys.append(a.mean())
            es.append(a.std() / np.sqrt(len(a)))
        ax.bar(np.arange(len(srcs)) + (j - 1) * 0.26, ys, 0.24, color=col, label=lab, yerr=es, ecolor=INK2, capsize=2)
    ax.axhline(sc.mean(), color=INK, ls="--", lw=1.5, label=f"scratch ({sc.mean():.2f})")
    ax.set_xticks(range(len(srcs)), [f"practiced {c}" for c in srcs])
    ax.set_ylabel("finetune AUC (mean ± s.e.)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Separate actor / critic networks (no shared trunk)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.savefig(os.path.join(FIG, "fig_sep.png"), dpi=150, bbox_inches="tight")
    return res


def make_all():
    os.makedirs(FIG, exist_ok=True)
    lines = []

    def say(x=""):
        lines.append(x)
        print(x)

    rows, M, Z, G = fig_matrix()
    say("== zero-shot greedy success > 5%% cells: %d of %d" % ((G > 0.05).sum(), G.size))
    curves_, fs, Ks, Kl = fig_frontier()
    say("== frontier F(K) (11 downstream tasks, tau=0.8, full transfer)")
    for c in sorted(curves_, key=lambda c: -curves_[c].mean()):
        say(f"   {c:15s} F(1M)={curves_[c][10]:.2f} F(3M)={curves_[c][30]:.2f} F(6M)={curves_[c][60]:.2f}")
    say(f"   {'scratch':15s} F(1M)={fs[10]:.2f} F(3M)={fs[30]:.2f} F(6M)={fs[60]:.2f} F(8M)={fs[80]:.2f}")
    base, out = cost_table()
    say(f"== break-even downstream tasks N* (scratch steps/task {base / 1e6:.2f}M)")
    for c, v in sorted(out.items(), key=lambda kv: kv[1]["cost_per_task"]):
        say(f"   {c:15s} steps/task {v['cost_per_task'] / 1e6:.2f}M  N*={v['breakeven']:.2f}")
    fig_decomp()
    fig_mech()
    fig_residual()
    fig_collapse()
    fig_ckpt()
    for name in ["fig_optimism", "fig_ncsweep", "fig_predictors", "fig_mg", "fig_ckptnc", "fig_reponly", "fig_sep"]:
        if name in globals():
            try:
                globals()[name]()
            except FileNotFoundError as e:
                say(f"skip {name}: {e}")
    say("== compute: " + str(compute_budget()))
    open(os.path.join(RES, "summary.txt"), "w").write("\n".join(lines))


def gt_table(name="gt"):
    import learn
    d = load(name)
    m = metrics(d["log"])
    _, sm = curves(d["log"])
    B = d["log"]["steps_per_iter"] * d["log"]["success"].shape[1]
    sc, ck = {}, {}
    for i, r in enumerate(d["runs"]):
        t = r["gt_task"]
        if r["variant"] == "scratch":
            sc.setdefault(t, []).append((m["auc"][i], m["t80"][i], m["disc"][i]))
        else:
            ck.setdefault((r["cond"], r["pseed"], r["pre_step"]), {})[t] = (m["auc"][i], m["t80"][i], sm[i, :12].mean(), sm[i, :30].mean(), m["disc"][i])
    sc = {t: {"auc": np.mean([q[0] for q in v]), "learned": np.mean([np.isfinite(q[1]) for q in v]),
              "t80": np.median([min(q[1], B) for q in v]), "disc": np.median([min(q[2], B) for q in v])} for t, v in sc.items()}
    return sc, ck, B


def gt_scores(sc, ck, B, tasks):
    out = {}
    for key, row in ck.items():
        ts = [t for t in tasks if t in row]
        if len(ts) < len(tasks):
            continue
        out[key] = {"gain": np.mean([row[t][0] - sc[t]["auc"] for t in ts]),
                    "disc": np.mean([np.log(sc[t]["disc"] + 1e4) - np.log(min(row[t][4], B) + 1e4) for t in ts]),
                    "cons": np.mean([(min(row[t][1], B) - min(row[t][4], B)) / 1e6 for t in ts if np.isfinite(row[t][1])] or [np.nan]),
                    "frontier": np.sum([np.isfinite(row[t][1]) for t in ts]) - np.sum([sc[t]["learned"] for t in ts]),
                    "saved": np.mean([sc[t]["t80"] - min(row[t][1], B) for t in ts]) / 1e6}
    return out


def metric_rows(mname="metrics", pname="pretrain", gname="metrics2", lname="lprobe"):
    import learn
    d = load(mname)
    ref = d["ref"]
    u_ev, u_ev_o = ref["uni"]["evrate"].mean(0), ref["uni_open"]["evrate"].mean(0)
    pre = load(pname)
    _, psm = curves(pre["log"])
    spi = pre["log"]["steps_per_iter"]
    ph, pe = pre["log"]["hits"].numpy(), pre["log"]["episodes"].numpy()
    from gridworld import PREDICATES
    surp = lambda ev, u: (ev * -torch.log(u + 1e-5)).sum(-1)
    lift = lambda ev, u: torch.log((ev + 1e-3) / (u + 1e-3)).mean(-1)
    ent = lambda q: -(q * q.clamp(min=1e-12).log()).sum(-1)
    zok = ref["uni"]["zrate"].mean(0) > 0
    logreach = lambda zr: torch.log(zr[:, zok] + 1e-4).mean(1)

    def rows_of(mm, R):
        ro, rep = mm["roll"], mm["rep"]
        a = ro["aff"]
        rows = [dict() for _ in range(R)]
        cols = {
            "door": ro["evrate"][:, 8], "right": ro["evrate"][:, 11], "key": ro["evrate"][:, 0],
            "ev_lift": lift(ro["evrate"], u_ev), "ev_surp": surp(ro["evrate"], u_ev), "z_rr": ro["rr"], "z_depth": ro["depth"],
            "z_n": ro["nz"], "occ_ent": ent(ro["occ"]),
            "aff_pick": a[:, 0, 1], "aff_drop": a[:, 1, 1], "aff_tog": a[:, 2, 1], "aff_unlock": a[:, 3, 1], "aff_fwd": a[:, 4, 1],
            "aff_sel": ((a[:, 0, 1] - a[:, 0, 0]) + (a[:, 1, 1] - a[:, 1, 0]) + (a[:, 2, 1] - a[:, 2, 0])) / 3,
            "futile": (a[:, 0, 0] + a[:, 1, 0] + a[:, 2, 0] + a[:, 4, 0]) / 4,
            "dv_mean": (ro["dv"] * (ro["dvn"] > 0)).sum(1) / (ro["dvn"] > 0).sum(1).clamp(min=1), "dv_unlock": ro["dv"][:, 8],
            "jump_surp": surp(mm["jump"]["evrate"], u_ev), "jump_rr": mm["jump"]["rr"], "jump_depth": mm["jump"]["depth"],
            "mix_surp": surp(mm["mix"]["evrate"], u_ev), "mix_rr": mm["mix"]["rr"], "mix_depth": mm["mix"]["depth"],
            "open_surp": surp(mm["roll_open"]["evrate"], u_ev_o), "open_rr": mm["roll_open"]["rr"],
            "z_logreach": logreach(ro["zrate"]), "jump_logreach": logreach(mm["jump"]["zrate"]), "mix_logreach": logreach(mm["mix"]["zrate"]),
            "ev_logreach": torch.log(ro["evrate"][:, u_ev > 0] + 1e-4).mean(1), "mix_ev_logreach": torch.log(mm["mix"]["evrate"][:, u_ev > 0] + 1e-4).mean(1),
            "v_mean": rep["v_mean"], "v_std": rep["v_std"], "ent": rep["ent_ds"],
            "plast10": mm["plast"][:, 0], "plast50": mm["plast"][:, 1], "plast200": mm["plast"][:, 2],
        }
        for nm in ["h", "h1"]:
            cols[f"aff_r2_{nm}"] = rep[f"aff_{nm}"].mean(1)
            cols[f"fut_r2_{nm}"] = rep[f"fut_{nm}"][:, [0, 1, 2, 4, 5, 6, 8, 11]].mean(1)
            cols[f"inv_r2_{nm}"] = rep[f"inv_{nm}"].mean(1)
            cols[f"z_r2_{nm}"] = rep[f"z_{nm}"].mean(1)
        for kind in ["front", "carry", "door", "ctrl"]:
            for q in ["kl", "h", "v"]:
                cols[f"cf_{q}_{kind}"] = rep[f"cf_{q}_{kind}"]
        cols["cf_kl_obj"] = (rep["cf_kl_front"] + rep["cf_kl_carry"] + rep["cf_kl_door"]) / 3 / rep["cf_kl_ctrl"].clamp(min=1e-6)
        cols["cf_h_obj"] = (rep["cf_h_front"] + rep["cf_h_carry"] + rep["cf_h_door"]) / 3 / rep["cf_h_ctrl"].clamp(min=1e-6)
        if "cka0_h" in rep:
            cols["cka0_h"], cols["cka0_h1"] = rep["cka0_h"], rep["cka0_h1"]
            cols["drift1"], cols["drift2"] = rep["drift"][:, 0], rep["drift"][:, 1]
        for k, v in cols.items():
            for r in range(R):
                rows[r][k] = float(v[r])
        return rows

    out = {}
    for st in [k for k in d if isinstance(k, int)]:
        e = d[st]
        rows = rows_of(e["m"], len(e["runs"]))
        it = max(st // spi - 1, 0)
        for r, run in enumerate(e["runs"]):
            pi = next(i for i, q in enumerate(pre["runs"]) if q["cond"] == run["cond"] and q["pseed"] == run["pseed"])
            rows[r]["practice_sr"] = float(psm[pi, it]) if not run["cond"].endswith(("explore", "impact")) else np.nan
            for nm in ["open_door", "enter_right", "pick_key"]:
                rows[r][f"prac_{nm}"] = float(ph[pi, :it + 1, PREDICATES.index(nm)].sum() / max(pe[pi, :it + 1].sum(), 1))
            rows[r]["prac_ent"] = float(pre["log"]["entropy"][pi, max(it - 5, 0):it + 1].mean())
            rows[r]["prac_vloss"] = float(pre["log"]["vloss"][pi, max(it - 5, 0):it + 1].mean())
            rows[r]["step"] = st
            out[(run["cond"], run["pseed"], st)] = rows[r]
    fresh = rows_of(load("metrics")["fresh"][0], 16)
    add_extra(out, fresh, gname, lname)
    return out, fresh, d


def grad_cols(g):
    ok = (g["rate"] > 0.005) & (g["rate"] < 0.995)
    return {"grad_cos": (g["cos"] * ok).sum(1) / ok.sum(1).clamp(min=1), "grad_cos_unlock": g["cos"][:, 8],
            "grad_norm": (g["gnorm"] * ok).sum(1) / ok.sum(1).clamp(min=1), "fisher": g["fisher"]}


def add_extra(out, fresh, gname, lname, sc_name="lprobe"):
    import learn
    if os.path.exists(os.path.join(RES, gname + ".pt")):
        g = load(gname)
        for st, e in g.items():
            if st == "fresh":
                cols = grad_cols(e)
                for r in range(len(fresh)):
                    fresh[r].update({k: float(v[r]) for k, v in cols.items()})
                continue
            cols = grad_cols(e["g"])
            for r, run in enumerate(e["runs"]):
                if (run["cond"], run["pseed"], st) in out:
                    out[(run["cond"], run["pseed"], st)].update({k: float(v[r]) for k, v in cols.items()})
    if os.path.exists(os.path.join(RES, lname + ".pt")):
        d = load(lname)
        _, sm = curves(d["log"])
        scd = load(sc_name)
        _, ssm = curves(scd["log"])
        base = {}
        for i, r in enumerate(scd["runs"]):
            if r["variant"] == "scratch":
                base.setdefault(r["gt_task"], []).append([ssm[i, :k].mean() for k in (6, 12, 30)])
        base = {t: np.mean(v, 0) for t, v in base.items()}
        acc = {}
        for i, r in enumerate(d["runs"]):
            if r["variant"] == "scratch" or r["gt_task"] == r["cond"]:
                continue
            acc.setdefault((r["cond"], r["pseed"], r["pre_step"]), []).append(np.array([sm[i, :k].mean() for k in (6, 12, 30)]) - base[r["gt_task"]])
        for key, v in acc.items():
            if key in out:
                v = np.mean(v, 0)
                out[key].update({"lp50": v[0], "lp100": v[1], "lp250": v[2]})
        for f in fresh:
            f.update({"lp50": 0.0, "lp100": 0.0, "lp250": 0.0})


def reliability(target="gain", tasks="test"):
    M, Y, meta, F0, sets, sc, ck, B = pot_data("ckpt")
    by = {}
    for m_, y in zip(meta, Y[tasks][target]):
        by.setdefault((m_["cond"], m_["step"]), {})[m_["pseed"]] = y
    pairs = [(v[0], v[1], v[2]) for v in by.values() if len(v) == 3]
    a = np.array(pairs)
    r = np.mean([spearman(a[:, i], a[:, j]) for i, j in [(0, 1), (0, 2), (1, 2)]])
    return r, 3 * r / (1 + 2 * r)


def spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[ok], y[ok])[0] if ok.sum() > 3 else np.nan


def pot_data(level="cond", heldout=False):
    import learn
    sc, ck, B = gt_table()
    rows, fresh, d = metric_rows()
    if heldout:
        ck = gt_table("gt_test")[1]
        rows = metric_rows("metrics_test", "pretrain_test", "metrics2_test", "lprobe_test")[0]
    fams = dict(learn.GT_TASKS)
    sets = {"all": [t for f in fams.values() for t in f], "dev": learn.GT_DEV}
    sets["test"] = [t for t in sets["all"] if t not in learn.GT_DEV]
    for f, ts in fams.items():
        sets[f] = ts
        sets[f + "_test"] = [t for t in ts if t not in learn.GT_DEV]
    gts = {s: gt_scores(sc, ck, B, ts) for s, ts in sets.items()}
    keys = [k for k in rows if k in gts["all"]]
    for k in keys:
        rows[k]["ds_trunc100_test"] = np.mean([ck[k][t][2] for t in sets["test"]])
        rows[k]["ds_trunc250_test"] = np.mean([ck[k][t][3] for t in sets["test"]])
        rows[k]["ds_trunc250_dev"] = np.mean([ck[k][t][3] for t in sets["dev"]])
        rows[k]["ds_full_dev"] = np.mean([ck[k][t][0] - sc[t]["auc"] for t in sets["dev"]])
    if level == "cond":
        groups = sorted({(c, st) for c, _, st in keys})
        agg = lambda tab, g, f: np.nanmean([tab[(g[0], ps, g[1])].get(f, np.nan) for ps in range(3) if (g[0], ps, g[1]) in tab] or [np.nan])
        names = list(dict.fromkeys(n for k in keys for n in rows[k]))
        M = {n: np.array([agg(rows, g, n) for g in groups]) for n in names}
        Y = {s: {q: np.array([agg(gts[s], g, q) for g in groups]) for q in ["gain", "frontier", "saved", "disc", "cons"]} for s in sets}
        meta = [{"cond": c, "step": st} for c, st in groups]
    else:
        names = list(dict.fromkeys(n for k in keys for n in rows[k]))
        M = {n: np.array([rows[k].get(n, np.nan) for k in keys]) for n in names}
        Y = {s: {q: np.array([gts[s][k][q] for k in keys]) for q in ["gain", "frontier", "saved", "disc", "cons"]} for s in sets}
        meta = [{"cond": c, "pseed": ps, "step": st} for c, ps, st in keys]
    F0 = {n: np.nanmean([f[n] for f in fresh]) if n in fresh[0] else np.nan for n in names}
    return M, Y, meta, F0, sets, sc, ck, B


def boot_rho(x, y, conds, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    uc = sorted(set(conds))
    idx = {c: np.where(np.array(conds) == c)[0] for c in uc}
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])
        out.append(spearman(x[pick], y[pick]))
    return np.nanpercentile(out, [5, 95])


def eval_pot(target="gain"):
    M, Y, meta, F0, sets, sc, ck, B = pot_data("cond")
    conds = [m["cond"] for m in meta]
    steps = np.array([m["step"] for m in meta])
    ls = np.log(steps)
    res = {}
    for n, x in M.items():
        if np.nanstd(x) == 0 or n == "step":
            continue
        r = {s: spearman(x, Y[s][target]) for s in sets}
        r["ci_test"] = boot_rho(x, Y["test"][target], conds, n=300)
        r["within_step"] = np.nanmean([spearman(x[steps == st], Y["test"][target][steps == st]) for st in sorted(set(steps))])
        r["within_cond"] = np.nanmean([spearman(x[np.array(conds) == c], Y["test"][target][np.array(conds) == c]) for c in sorted(set(conds))])
        from scipy.stats import rankdata
        rx, ry, rs = rankdata(x), rankdata(Y["test"][target]), rankdata(ls)
        rxs, rys = np.corrcoef(rx, rs)[0, 1], np.corrcoef(ry, rs)[0, 1]
        r["partial_step"] = (np.corrcoef(rx, ry)[0, 1] - rxs * rys) / np.sqrt((1 - rxs ** 2) * (1 - rys ** 2) + 1e-12)
        top = np.argsort(-np.nan_to_num(x, nan=-1e9))[:5]
        r["top5"] = Y["test"][target][top].mean()
        res[n] = r
    return res, M, Y, meta, F0


def auroc(x, y):
    from scipy.stats import mannwhitneyu
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if y.all() or (~y).all():
        return np.nan
    return mannwhitneyu(x[y], x[~y]).statistic / (y.sum() * (~y).sum())


METRIC_COST = {"rollout": ["door", "right", "key", "ev_lift", "ev_surp", "z_rr", "z_depth", "z_n", "occ_ent", "aff_pick", "aff_drop",
                           "aff_tog", "aff_unlock", "aff_fwd", "aff_sel", "futile", "dv_mean", "dv_unlock"],
               "jump": ["jump_surp", "jump_rr", "jump_depth"], "mix": ["mix_surp", "mix_rr", "mix_depth"],
               "rollout_open": ["open_surp", "open_rr"]}


def metric_cost(n, d):
    c = d[50_000]["cost"]
    for pre, grp in [("mix_", "mix"), ("jump_", "jump"), ("open_", "rollout_open")]:
        if n.startswith(pre):
            return c[grp]["env_steps"], c[grp]["sec"]
    if n.startswith(("z_rr", "z_depth", "z_n", "z_logreach", "ev_", "occ", "door", "right", "key", "aff_pick", "aff_drop", "aff_tog",
                     "aff_unlock", "aff_fwd", "aff_sel", "futile", "dv")):
        return c["rollout"]["env_steps"], c["rollout"]["sec"]
    if n.startswith("lp"):
        k = {"lp50": 50_000, "lp100": 100_000, "lp250": 250_000}[n]
        return 6 * k, np.nan
    if n.startswith("ds_trunc"):
        k = 100_000 if "100" in n else 250_000
        return k * (15 if "test" in n else 8), np.nan
    if n == "ds_full_dev":
        return 2_000_000 * 8, np.nan
    if n.startswith("grad") or n == "fisher":
        return 256 * 200, np.nan
    if n.startswith("prac") or n == "practice_sr":
        return 0, 0.0
    return 16384, c["rep"]["sec"] + (c["plast"]["sec"] if n.startswith("plast") else 0)


def pot_eval(target="gain", heldout=False):
    M, Y, meta, F0, sets, sc, ck, B = pot_data("cond", heldout)
    d = load("metrics")
    conds = np.array([m["cond"] for m in meta])
    steps = np.array([m["step"] for m in meta])
    from scipy.stats import rankdata
    rs = rankdata(np.log(steps))
    res = {}
    fam_test = {f: f + "_test" for f in ["D", "L", "O"]}
    for n, x in M.items():
        if n == "step" or not np.isfinite(x).any() or np.nanstd(x) == 0:
            continue
        r = {s: spearman(x, Y[s][target]) for s in sets}
        yt = Y["test"][target]
        r["ci"] = boot_rho(x, yt, list(conds), n=300)
        r["within_step"] = np.nanmean([spearman(x[steps == st], yt[steps == st]) for st in sorted(set(steps))])
        r["within_cond"] = np.nanmean([spearman(x[conds == c], yt[conds == c]) for c in sorted(set(conds))])
        ok = np.isfinite(x)
        rx, ry = rankdata(x[ok]), rankdata(yt[ok])
        a, b = np.corrcoef(rx, rs[ok])[0, 1], np.corrcoef(ry, rs[ok])[0, 1]
        r["partial"] = (np.corrcoef(rx, ry)[0, 1] - a * b) / np.sqrt(max((1 - a ** 2) * (1 - b ** 2), 1e-12))
        top = np.argsort(-np.where(ok, x, -np.inf))[:5]
        r["top5"] = yt[top].mean()
        r["front_auc"] = auroc(x, Y["all"]["frontier"] > 0)
        r["fresh"] = F0.get(n, np.nan)
        r["cost_steps"], r["cost_sec"] = metric_cost(n, d)
        res[n] = r
    yt = Y["test"][target]
    oracle = np.sort(yt)[::-1][:5].mean()
    return res, M, Y, meta, F0, oracle


FAMILY = [("ds_", "downstream-aware"), ("lp", "learning probe"), ("grad", "gradient"), ("fisher", "gradient"), ("plast", "gradient"),
          ("practice", "practice stats"), ("prac", "practice stats"), ("v_", "critic"), ("dv", "critic"), ("cf_v", "critic"),
          ("aff_r2", "representation"), ("fut_r2", "representation"), ("inv_r2", "representation"), ("z_r2", "representation"),
          ("cka", "representation"), ("drift", "representation"), ("cf_h", "representation"), ("cf_kl", "policy sensitivity"),
          ("aff", "affordance"), ("futile", "affordance"), ("jump", "reach + noise"), ("mix", "reach + noise"), ("ent", "practice stats")]
FAM_COL = {"downstream-aware": INK, "learning probe": SERIES[6], "gradient": SERIES[4], "practice stats": MUTED, "critic": SERIES[7],
           "representation": SERIES[0], "policy sensitivity": SERIES[3], "affordance": "#9a5b2e", "reach + noise": SERIES[1], "reach": SERIES[5]}


def metric_family(n):
    return next((f for p, f in FAMILY if n.startswith(p)), "reach")


def fig_pot(res, name="fig_pot", top=34):
    names = sorted(res, key=lambda n: -np.nan_to_num(res[n]["test"]))
    names = names[:top // 2] + [n for n in names[top // 2:] if n.startswith(("ds_", "practice", "lp", "door", "grad", "plast", "ent"))][:6] + names[-(top // 2 - 6):]
    names = list(dict.fromkeys(names))
    fig, axs = plt.subplots(1, 3, figsize=(16, 0.24 * len(names) + 1.6), gridspec_kw={"width_ratios": [1.2, 1, 1.1]})
    y = np.arange(len(names))[::-1]
    cols = [FAM_COL[metric_family(n)] for n in names]
    for ax, key, title in [(axs[0], "test", "Held-out tasks: Spearman rho, 90% CI"),
                           (axs[1], "within_step", "Same pretraining step")]:
        v = np.array([res[n][key] for n in names])
        ax.barh(y, v, color=cols, height=0.7)
        if key == "test":
            ci = np.array([res[n]["ci"] for n in names])
            ax.hlines(y, ci[:, 0], ci[:, 1], color=INK2, lw=1)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y, names if ax is axs[0] else [""] * len(names), fontsize=7.5)
        ax.set_xlim(-1, 1)
        ax.set_title(title, loc="left", fontsize=9)
    for n in names:
        c = res[n]["cost_steps"]
        axs[2].scatter(max(c, 1e3), res[n]["test"], color=FAM_COL[metric_family(n)], s=22, zorder=3)
        if abs(res[n]["test"]) > 0.35 or n.startswith(("ds_", "lp", "practice")):
            axs[2].annotate(n, (max(c, 1e3), res[n]["test"]), fontsize=6.5, color=INK2, xytext=(3, 2), textcoords="offset points")
    if os.path.exists(os.path.join(RES, "mixsweep.pt")):
        sw = mixsweep_eval()
        for g in [(0.5, 16, 1), (0.5, 64, 1)]:
            axs[2].scatter(sw[g]["cost"], sw[g]["rr"]["test"], color=FAM_COL["reach + noise"], s=22, marker="D", zorder=3)
            axs[2].annotate(f"mix_rr, {sw[g]['cost']} steps", (sw[g]["cost"], sw[g]["rr"]["test"]), fontsize=6.5, color=INK2,
                            xytext=(3, -8), textcoords="offset points")
    axs[2].set_xscale("log")
    axs[2].set_xlabel("env steps to measure one checkpoint (log; 1e3 = free)")
    axs[2].set_ylabel("held-out rho")
    axs[2].set_title("Accuracy vs measurement cost", loc="left", fontsize=9)
    for f, c in FAM_COL.items():
        axs[2].scatter([], [], color=c, label=f, s=22)
    axs[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def family_consistency(target="gain"):
    M, Y, meta, F0, sets, sc, ck, B = pot_data("cond")
    fams = ["D", "L", "O"]
    C = np.array([[spearman(Y[a][target], Y[b][target]) for b in fams] for a in fams])
    steps = np.array([m["step"] for m in meta])
    ws = {f"{a}-{b}": np.nanmean([spearman(Y[a][target][steps == st], Y[b][target][steps == st]) for st in sorted(set(steps))])
          for a, b in [("D", "L"), ("D", "O"), ("L", "O")]}
    return C, ws, Y, meta


def early_late(metrics_, at=50_000, later=(250_000, 2_000_000), target="gain", tasks="test"):
    M, Y, meta, F0, sets, sc, ck, B = pot_data("cond")
    conds = np.array([m["cond"] for m in meta])
    steps = np.array([m["step"] for m in meta])
    out = {}
    for n in metrics_:
        r = {}
        for L in later:
            cs = sorted(set(conds[steps == at]) & set(conds[steps == L]))
            x = np.array([M[n][(conds == c) & (steps == at)][0] for c in cs])
            y = np.array([Y[tasks][target][(conds == c) & (steps == L)][0] for c in cs])
            r[L] = spearman(x, y)
            pick = np.argsort(-np.nan_to_num(x, nan=-1e9))[:3]
            r[f"top3_{L}"] = (y[pick].mean(), np.sort(y)[::-1][:3].mean(), y.mean())
        out[n] = r
    return out


def q1q2(metrics_, target="gain"):
    M, Y, meta, F0, sets, sc, ck, B = pot_data("cond")
    conds = np.array([m["cond"] for m in meta])
    steps = np.array([m["step"] for m in meta])
    out = {}
    for n in metrics_:
        x = M[n]
        s50 = steps == 50_000
        od = x[s50 & (conds == "open_door")][0]
        pct = (x[s50] < od).mean()
        agree = []
        for st in sorted(set(steps)):
            a, b = (steps == st) & (conds == "open_door"), (steps == st) & (conds == "face_door")
            agree.append(np.sign(x[a][0] - x[b][0]) == np.sign(Y["test"][target][a][0] - Y["test"][target][b][0]))
        out[n] = {"open_door@50k pct": pct, "open_door@50k vs fresh": od - F0.get(n, np.nan), "od>fd sign agree": np.mean(agree)}
    gt = {"open_door@50k pct": (Y["test"][target][s50] < Y["test"][target][s50 & (conds == "open_door")][0]).mean()}
    return out, gt


def frozen_protocol(k=5, target="gain"):
    dev, M, Y, meta, F0, _ = pot_eval_on("dev", target)
    cand = [n for n in dev if not n.startswith("ds_")]
    best = sorted(cand, key=lambda n: -np.nan_to_num(dev[n]))[:k]
    test = pot_eval(target)[0]
    held = pot_eval(target, heldout=True)[0]
    return best, {n: (dev[n], test[n]["test"], held.get(n, {}).get("test", np.nan), held.get(n, {}).get("within_step", np.nan)) for n in best}


def pot_eval_on(task_set, target="gain", heldout=False):
    M, Y, meta, F0, sets, sc, ck, B = pot_data("cond", heldout)
    return {n: spearman(x, Y[task_set][target]) for n, x in M.items() if n != "step"}, M, Y, meta, F0, sets


def mixsweep_eval(target="gain", heldout=False):
    M, Y, meta, F0, sets, sc, ck, B = pot_data("cond", heldout)
    d = load("mixsweep")
    ref = load("metrics")["ref"]
    zok = ref["uni"]["zrate"].mean(0) > 0
    name = "test" if heldout else "main"
    out = {}
    for g in d["grid"]:
        vals = {}
        for st in [k[1] for k in d if isinstance(k, tuple) and k[0] == name]:
            e = d[(name, st)]
            r = e["r"][g]
            lr = torch.log(r["zrate"][:, zok] + 1e-4).mean(1)
            for i, run in enumerate(e["runs"]):
                vals.setdefault((run["cond"], st), []).append((float(r["rr"][i]), float(lr[i])))
        x = np.array([np.mean(vals[(m["cond"], m["step"])], 0) for m in meta])
        out[g] = {q: {s: spearman(x[:, j], Y[s][target]) for s in ["test", "D_test", "L_test", "O_test"]} for j, q in enumerate(["rr", "logreach"])}
        out[g]["cost"] = g[1] * 200 * g[2]
    return out


REL = {"D": ("mix", [8, 11]), "L": ("mix", [0, 1, 2, 4, 5, 6]), "O": ("mixopen", [3, 5, 6, 7, 11])}


def rel_reach(meta, heldout=False):
    d = load("metrics_test" if heldout else "metrics")
    mo = load("mixopen")
    vals = {}
    for st in [k for k in d if isinstance(k, int)]:
        e = d[st]
        o = mo[("test" if heldout else "main", st)]["r"]
        for f, (src, idx) in REL.items():
            ev = o["evrate"] if f == "O" else e["m"][src]["evrate"]
            lr = torch.log(ev[:, idx] + 1e-4).mean(1)
            for i, run in enumerate(e["runs"]):
                vals.setdefault(f, {}).setdefault((run["cond"], st), []).append(float(lr[i]))
        for i, run in enumerate(e["runs"]):
            vals.setdefault("mixopen_rr", {}).setdefault((run["cond"], st), []).append(float(o["rr"][i]))
    return {f: np.array([np.mean(vals[f][(m["cond"], m["step"])]) for m in meta]) for f in vals}


def fig_pot_family(name="fig_pot_family"):
    M, Y, meta, F0, sets, sc, ck, B = pot_data("cond")
    conds = np.array([m["cond"] for m in meta])
    cs = sorted(set(conds), key=lambda c: -Y["D"]["gain"][conds == c].mean())
    G = np.array([[Y[f]["gain"][conds == c].mean() for f in ["D", "L", "O"]] for c in cs])
    rr = rel_reach(meta)
    R = np.array([[spearman(rr[a], Y[b + "_test"]["gain"]) for b in ["D", "L", "O"]] for a in ["D", "L", "O"]])
    print("mixopen_rr", {b: round(spearman(rr["mixopen_rr"], Y[b + "_test"]["gain"]), 2) for b in ["D", "L", "O"]})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    v = np.abs(G).max()
    im = axs[0].imshow(G, cmap=DIV, vmin=-v, vmax=v, aspect="auto")
    axs[0].set_xticks(range(3), ["D: door chain", "L: left room", "O: door open"])
    axs[0].set_yticks(range(len(cs)), cs)
    for i in range(len(cs)):
        for j in range(3):
            axs[0].text(j, i, f"{G[i, j]:+.2f}", ha="center", va="center", fontsize=7.5, color=INK)
    axs[0].set_title("AUC gain vs scratch (critic reset, mean over checkpoints)", loc="left")
    axs[0].grid(False)
    fig.colorbar(im, ax=axs[0], shrink=0.7)
    im = axs[1].imshow(R, cmap=DIV, vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            axs[1].text(j, i, f"{R[i, j]:+.2f}", ha="center", va="center", fontsize=10, color=INK)
    axs[1].set_xticks(range(3), ["D test tasks", "L test tasks", "O test tasks"])
    axs[1].set_yticks(range(3), ["reach of D events\n(unlock, enter right)", "reach of L events\n(pick, drop)", "reach of O events\n(open world, noisy policy)"])
    axs[1].set_title("Spearman rho: event reachability (noisy policy) vs gain", loc="left")
    axs[1].grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return G, cs, R


def fig_pot_sweep(name="fig_pot_sweep"):
    out = mixsweep_eval()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    eps = [g for g in out if g[1] == 512]
    for j, (q, lab) in enumerate([("rr", "rarity-weighted reach"), ("logreach", "mean log-reach")]):
        axs[0].plot([g[0] for g in eps], [out[g][q]["test"] for g in eps], marker="o", color=SERIES[j], label=lab)
    axs[0].set_xlabel("epsilon (fraction of uniform random actions)")
    axs[0].set_ylabel("held-out rho")
    axs[0].set_title("Noise level of the probe policy", loc="left")
    axs[0].legend(fontsize=8)
    cost = sorted([g for g in out if g[0] == 0.5], key=lambda g: out[g]["cost"])
    for j, (q, lab) in enumerate([("rr", "rarity-weighted reach"), ("logreach", "mean log-reach")]):
        axs[1].plot([out[g]["cost"] for g in cost], [out[g][q]["test"] for g in cost], marker="o", color=SERIES[j], label=lab)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("env steps per checkpoint (epsilon = 0.5)")
    axs[1].set_title("Measurement budget", loc="left")
    for ax in axs:
        ax.set_ylim(0, 0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def combined(target="gain"):
    parts = []
    for ho in [False, True]:
        M, Y, meta, F0, sets, sc, ck, B = pot_data("cond", ho)
        rr = rel_reach(meta, ho)
        for f in rr:
            M[f"rel_{f}"] = rr[f]
        parts.append((M, Y, meta))
    names = [n for n in parts[0][0] if n in parts[1][0]]
    M = {n: np.concatenate([p[0][n] for p in parts]) for n in names}
    Y = {s: {q: np.concatenate([p[1][s][q] for p in parts]) for q in parts[0][1][s]} for s in parts[0][1]}
    meta = parts[0][2] + [dict(m, heldout=True) for m in parts[1][2]]
    return M, Y, meta


def selection_sim(metrics_, decide=50_000, later=(250_000, 1_000_000), target="gain", k=3):
    M, Y, meta = combined(target)
    conds = np.array([m["cond"] for m in meta])
    steps = np.array([m["step"] for m in meta])
    out = {}
    for fam in ["test", "D_test", "L_test", "O_test"]:
        for n in metrics_:
            for L in later:
                cs = sorted(set(conds[steps == decide]) & set(conds[steps == L]))
                x = np.array([M[n][(conds == c) & (steps == decide)][0] for c in cs])
                y = np.array([Y[fam][target][(conds == c) & (steps == L)][0] for c in cs])
                pick = np.argsort(-np.nan_to_num(x, nan=-1e9))[:k]
                out[(fam, n, L)] = {"rho": spearman(x, y), "picked": y[pick].mean(), "oracle": np.sort(y)[::-1][:k].mean(),
                                    "random": y.mean(), "names": [cs[i] for i in pick], "n": len(cs)}
    return out


def gt_curves(keys, tasks, names=("gt",)):
    out = {}
    for name in names:
        d = load(name)
        _, sm = curves(d["log"])
        x = (np.arange(sm.shape[1]) + 1) * d["log"]["steps_per_iter"] / 1e6
        for i, r in enumerate(d["runs"]):
            if r["gt_task"] not in tasks or r["variant"] == "full":
                continue
            k = ("scratch",) if r["variant"] == "scratch" else (r["cond"], r["pre_step"])
            if k in keys:
                out.setdefault(k, []).append(sm[i])
    return x, {k: np.array(v) for k, v in out.items()}


def cov_video(layouts=(3, 5)):
    out = os.path.join(ROOT, "videos")
    rows, _, _ = metric_rows()
    reach = load("pretrain_cov")["reach"][1_000_000]
    unlock = {c: float(reach["pol"]["evrate"][[i for i, r in enumerate(reach["runs"]) if r["cond"] == c], 8].mean()) for c in ["cover", "smm"]}
    unlock["explore"] = np.mean([rows[("explore", ps, 1_000_000)]["door"] for ps in range(3)])
    sc, ck, B = gt_table()
    co = cov_eval()
    dtasks = ["goal", "key_near_goal", "close_from_right", "goal_with_key"]
    g = ["cover", "smm", "explore"]
    keys = [(c, 1_000_000) for c in g] + [("scratch",)]
    x, cv = gt_curves(keys, dtasks, ("gt", "gt_cov"))
    names = {"cover": "coverage", "smm": "SMM", "explore": "explore"}
    series = [(f"{names[c]} practice: gain {cv[(c, 1_000_000)].mean() - cv[('scratch',)].mean():+.2f}", [SERIES[1], SERIES[3], MUTED][j], x,
               cv[(c, 1_000_000)].mean(0, keepdims=True)) for j, c in enumerate(g)] + [("From scratch", INK2, x, cv[("scratch",)].mean(0, keepdims=True))]
    ch = chart([("Door tasks after finetuning (critic reset, 4 tasks, 3 seeds)", "Success rate", (0, 1.05), False, series)], [0.0], 2.0,
               "Finetuning steps on the new task (M)")
    ev = (["pick_key", "open_door", "enter_right", "pick_ball2"], ["Key", "Door open", "Right room", "Purple ball"])
    title = "Reward-free coverage practice makes the agent open the door."
    hints = ["No agent receives a task reward. Coverage practice rewards rare situations. The coverage agent opens the door and enters the right room.",
             f"In many episodes, the agent opens the door: coverage {unlock['cover']:.0%}, SMM {unlock['smm']:.0%}, explore {unlock['explore']:.0%}. "
             "Door tasks learn faster after coverage practice."]
    subs = ["Reward: rare situations (1/q)", "Reward: low state density", "Reward: new cells"]
    frames = []
    for lay, hint in zip(layouts, hints):
        ags = [ck_slice("prec_1000000.pt", "cover", 0), ck_slice("prec_1000000.pt", "smm", 0), ck_slice("pre_1000000.pt", "explore", 0)]
        st = gw_rollout(ags, "null", ev[0], lay, max_steps=100)
        print("video7", lay, [s["ev"].tolist() for s in st[-1]])
        frames += scene(st, [f"After {names[c]} practice" for c in g], [SERIES[1], SERIES[3], MUTED], subs, title, hint,
                        "After 1M practice steps, no task reward", ev[1], False, ch)
    write_video(frames, os.path.join(out, "7_reward_free_coverage_practice_makes_the_agent_open_the_door.mp4"))


def pot_videos(layouts=(1, 4)):
    import learn
    out = os.path.join(ROOT, "videos")
    M, Y, meta, F0, sets, sc, ck, B = pot_data("cond")
    idx = {(m["cond"], m["step"]): i for i, m in enumerate(meta)}
    dtasks = ["goal", "key_near_goal", "close_from_right", "goal_with_key"]
    ltasks = learn.GT_TASKS["L"]
    ev = (["pick_key", "open_door", "enter_right"], ["Key", "Door open", "Right room"])
    stage = "After 2M practice steps, 50% random actions"

    g5 = ["pick_key", "face_door", "enter_right"]
    keys = [(c, 2_000_000) for c in g5] + [("scratch",)]
    x, cv = gt_curves(keys, dtasks)
    gain = lambda c, f: np.mean([np.mean([ck[(c, ps, 2_000_000)][t][0] for ps in range(3)]) - sc[t]["auc"] for t in f])
    series = [(f"{c} practice: gain {gain(c, dtasks):+.2f}", SERIES[j], x, cv[(c, 2_000_000)].reshape(-1, len(x)).mean(0, keepdims=True))
              for j, c in enumerate(g5)] + [("From scratch", INK2, x, cv[("scratch",)].mean(0, keepdims=True))]
    ch = chart([("Door tasks after finetuning (critic reset, 4 tasks, 3 seeds)", "Success rate", (0, 1.05), False, series)], [0.0], 2.0,
               "Finetuning steps on the new task (M)")
    sr = {c: M["practice_sr"][idx[(c, 2_000_000)]] for c in g5}
    title = "Practice success does not show what the agent can learn next."
    hints = ["All three agents do their practice task in 97-100% of episodes. With random actions, only the enter_right agent opens the door.",
             "A second layout. The result is the same: the enter_right agent reaches the right room, the other two do not."]
    frames = []
    for lay, hint in zip(layouts, hints):
        ags = [ck_slice("pre_final.pt", c, 0) for c in g5]
        st = gw_rollout(ags, "null", ev[0], lay, max_steps=100, eps=0.5)
        print("video5", lay, [s["ev"].tolist() for s in st[-1]])
        frames += scene(st, [f"After {c} practice" for c in g5], SERIES[:3], [f"Practice success {sr[c]:.0%}" for c in g5], title, hint,
                        stage, ev[1], False, ch)
    write_video(frames, os.path.join(out, "5_practice_success_does_not_show_what_the_agent_can_learn_next.mp4"))

    g6 = ["impact", "enter_right"]
    keys = [(c, 2_000_000) for c in g6] + [("scratch",)]
    panels = []
    for f, ts in [("Door tasks", dtasks), ("Left-room tasks", ltasks)]:
        x, cv = gt_curves(keys, ts)
        panels.append((f"{f} after finetuning", "Success rate", (0, 1.05), False,
                       [(f"{c} practice", [SERIES[4], SERIES[2]][j], x, cv[(c, 2_000_000)].mean(0, keepdims=True)) for j, c in enumerate(g6)] +
                       [("From scratch", INK2, x, cv[("scratch",)].mean(0, keepdims=True))]))
    ch = chart(panels, [0.0], 2.0, "Finetuning steps on the new task (M)")
    ev6 = (["pick_ball", "pick_box", "open_door", "enter_right"], ["Ball", "Box", "Door open", "Right room"])
    title = "Different practice reaches different states."
    hints = ["The impact agent picks up the ball and the box. The enter_right agent opens the door and goes to the right room.",
             f"Thus, the best practice depends on the next task. Door tasks: enter_right learns faster (gain {gain('enter_right', dtasks):+.2f} "
             f"vs {gain('impact', dtasks):+.2f}). Left-room tasks: impact learns faster ({gain('impact', ltasks):+.2f} vs {gain('enter_right', ltasks):+.2f})."]
    frames = []
    for lay, hint in zip(layouts, hints):
        ags = [ck_slice("pre_final.pt", c, 0) for c in g6]
        st = gw_rollout(ags, "null", ev6[0], lay, max_steps=100, eps=0.5)
        print("video6", lay, [s["ev"].tolist() for s in st[-1]])
        frames += scene(st, [f"After {c} practice" for c in g6], [SERIES[4], SERIES[2]],
                        [f"Practice success {M['practice_sr'][idx[(c, 2_000_000)]]:.0%}" if c != "impact" else "Intrinsic reward, no task" for c in g6],
                        title, hint, stage, ev6[1], False, ch)
    write_video(frames, os.path.join(out, "6_different_practice_reaches_different_states.mp4"))


def stopping(metrics_):
    M, Y, meta, F0, sets, sc, ck, B = pot_data("cond")
    M["-plast10"], M["-practice_sr"] = -M["plast10"], -M["practice_sr"]
    conds = np.array([m["cond"] for m in meta])
    steps = np.array([m["step"] for m in meta])
    out = {}
    for tasks in ["test", "D_test", "L_test", "O_test"]:
        y = Y[tasks]["gain"]
        cs = sorted(set(conds))
        r = {"oracle": np.mean([y[conds == c].max() for c in cs]), "random step": np.mean([y[conds == c].mean() for c in cs]),
             "always 2M": np.mean([y[(conds == c) & (steps == 2_000_000)][0] for c in cs])}
        for n in metrics_:
            r[n] = np.mean([y[conds == c][np.nanargmax(M[n][conds == c])] for c in cs])
        out[tasks] = r
    return out


def fig_pot_select(name="fig_pot_select"):
    ms = ["mix_logreach", "mix_rr", "lp50", "occ_ent", "door", "practice_sr"]
    el = {t: early_late(ms, tasks=t) for t in ["D_test", "L_test", "O_test"]}
    stp = stopping(["mix_logreach", "mix_rr", "lp50", "occ_ent", "door"])
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    w = 0.25
    for j, t in enumerate(["D_test", "L_test", "O_test"]):
        axs[0].bar(np.arange(len(ms)) + (j - 1) * w, [el[t][n][2_000_000] for n in ms], w, color=SERIES[j], label=t.replace("_test", " test tasks"))
    axs[0].set_xticks(range(len(ms)), ms, rotation=20)
    axs[0].axhline(0, color=INK2, lw=0.8)
    axs[0].set_ylim(-1, 1)
    axs[0].set_ylabel("Spearman rho")
    axs[0].set_title("Forecast: metric at 50k vs value of the same practice at 2M", loc="left")
    axs[0].legend(fontsize=8)
    keys = list(stp["test"])
    for j, t in enumerate(["D_test", "L_test", "O_test"]):
        axs[1].bar(np.arange(len(keys)) + (j - 1) * w, [stp[t][k] for k in keys], w, color=SERIES[j])
    axs[1].set_xticks(range(len(keys)), keys, rotation=25)
    axs[1].axhline(0, color=INK2, lw=0.8)
    axs[1].set_ylabel("AUC gain vs scratch")
    axs[1].set_title("Stopping rule: practice until the metric peaks (14 practice tasks)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return el, stp


def early_table():
    d = load("early")
    m = metrics(d["log"])
    tab = {}
    for i, r in enumerate(d["runs"]):
        key = ("scratch", "scratch") if r["variant"] == "scratch" else (r["cond"], r["variant"])
        tab.setdefault(key, {}).setdefault(r["gt_task"], []).append(m["auc"][i])
    return {k: {t: np.array(v) for t, v in row.items()} for k, row in tab.items()}


def early_eval():
    from scipy.stats import mannwhitneyu
    tab = early_table()
    tg = ["goal", "key_near_goal", "close_from_right"]
    sc = np.concatenate([tab[("scratch", "scratch")][t] - tab[("scratch", "scratch")][t].mean() for t in tg])
    scm = {t: tab[("scratch", "scratch")][t].mean() for t in tg}
    out = {}
    for (c, v), row in tab.items():
        if c == "scratch":
            continue
        g = np.concatenate([row[t] - scm[t] for t in tg])
        out[(c, v)] = {"gain": g.mean(), "se": g.std() / np.sqrt(len(g)), "p": mannwhitneyu(g, sc).pvalue, "n": len(g)}
    return out, scm


def fig_pot_early(name="fig_pot_early"):
    out, scm = early_eval()
    M, Y, meta, F0, sets, sc, ck, B = pot_data("cond")
    idx = {m["cond"]: i for i, m in enumerate(meta) if m["step"] == 50_000}
    cs = sorted({c for c, _ in out}, key=lambda c: -out[(c, "nocritic")]["gain"])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    y = np.arange(len(cs))[::-1]
    for j, (v, lab) in enumerate([("nocritic", "critic reset"), ("full", "full transfer")]):
        axs[0].barh(y + (0.2 if j == 0 else -0.2), [out[(c, v)]["gain"] for c in cs], 0.38, xerr=[out[(c, v)]["se"] for c in cs],
                    color=SERIES[j], label=lab, error_kw={"lw": 0.8, "ecolor": INK2})
    axs[0].set_yticks(y, [f"{c} ({M['practice_sr'][idx[c]]:.0%})" if np.isfinite(M['practice_sr'][idx[c]]) else c for c in cs])
    axs[0].axvline(0, color=INK2, lw=0.8)
    axs[0].set_xlabel("AUC gain vs scratch (3 door tasks, 18 runs per bar, mean +- s.e.)")
    axs[0].set_title("After only 50k practice steps (practice success in brackets)", loc="left")
    axs[0].legend(fontsize=8)
    gy = np.array([out[(c, "full")]["gain"] for c in cs])
    x = np.array([M["practice_sr"][idx[c]] for c in cs])
    axs[1].scatter(x, gy, color=SERIES[1], s=24)
    for c, a, b in zip(cs, x, gy):
        if np.isfinite(a):
            axs[1].annotate(c, (a, b), fontsize=6.5, color=INK2, xytext=(3, 2), textcoords="offset points")
    axs[1].set_xlabel("practice success at 50k")
    axs[1].set_ylabel("gain (full transfer)")
    axs[1].set_title(f"Practice success: rho = {spearman(x, gy):+.2f}", loc="left")
    rc = rehearse_cols()
    xb = np.array([M["door"][idx[c]] for c in cs])
    xa = np.array([rc[("full", "door")][(c, 50_000)] for c in cs])
    axs[2].scatter(xb, gy, facecolors="none", edgecolors=MUTED, s=24, label=f"before rehearsal: rho {spearman(xb, gy):+.2f}")
    axs[2].scatter(xa, gy, color=SERIES[1], s=24, label=f"after 100k zero-reward steps: rho {spearman(xa, gy):+.2f}")
    for c, a, b in zip(cs, xa, gy):
        axs[2].annotate(c, (a, b), fontsize=6.5, color=INK2, xytext=(3, 2), textcoords="offset points")
    axs[2].set_xlabel("door-open rate of the policy (reward-free world)")
    axs[2].set_ylabel("gain (full transfer)")
    axs[2].set_title("Reward-free rehearsal of the finetuning protocol", loc="left")
    axs[2].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out, scm


def fig_pot_heldout(n="mix_logreach", name="fig_pot_heldout"):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, f in zip(axs, ["test", "D_test", "O_test"]):
        for ho, col, lab in [(False, SERIES[0], "dev practice (14 tasks)"), (True, SERIES[1], "held-out practice (8 tasks)")]:
            M, Y, meta, F0, sets, sc, ck, B = pot_data("cond", ho)
            ax.scatter(M[n], Y[f]["gain"], color=col, s=18, label=f"{lab}: rho {spearman(M[n], Y[f]['gain']):+.2f}")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel(f"{n} (reward-free, noisy policy)")
        ax.set_ylabel("AUC gain vs scratch")
        ax.set_title({"test": "All 15 held-out tasks", "D_test": "Door-chain test tasks", "O_test": "Door-open test tasks"}[f], loc="left")
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def pot_report():
    lines = []
    say = lambda s: (print(s), lines.append(s))
    res, M, Y, meta, F0, oracle = pot_eval()
    fig_pot(res)
    say(f"== main practice: {len(meta)} (task, step) points, oracle top5 gain {oracle:.3f}, mean {Y['test']['gain'].mean():.3f}")
    say(f"   reliability (pseed split, 1 seed / Spearman-Brown 3 seeds): {reliability()}")
    cols = ["dev", "test", "D_test", "L_test", "O_test", "within_step", "partial", "front_auc", "top5"]
    say(f"{'metric':18s}" + "".join(f"{c[:9]:>10s}" for c in cols) + "   ci90        cost_steps")
    for n in sorted(res, key=lambda n: -np.nan_to_num(res[n]["test"])):
        r = res[n]
        say(f"{n:18s}" + "".join(f"{r[c]:10.2f}" for c in cols) + f"   [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] {r['cost_steps']}")
    held = pot_eval(heldout=True)[0]
    say("== held-out practice (8 new tasks x 3 steps)")
    for n in sorted(held, key=lambda n: -np.nan_to_num(held[n]["test"]))[:25]:
        r = held[n]
        say(f"{n:18s} test={r['test']:+.2f} D={r['D_test']:+.2f} L={r['L_test']:+.2f} O={r['O_test']:+.2f} within_step={r['within_step']:+.2f}")
    best, fr = frozen_protocol()
    say("== frozen protocol (top-5 by dev-task rho on dev practice): dev, test, held-out test, held-out within-step")
    for n, v in fr.items():
        say(f"   {n:18s} " + " ".join(f"{x:+.2f}" for x in v))
    C, ws, _, _ = family_consistency()
    say(f"== family gain correlation D/L/O:\n{np.round(C, 2)}\n   within-step {ws}")
    G, cs, R = fig_pot_family()
    say(f"== family-relevant reach (rows D/L/O events) vs family test gain:\n{np.round(R, 2)}")
    sw = fig_pot_sweep()
    for g, r in sw.items():
        say(f"   sweep eps={g[0]} E={g[1]} eps_per_env={g[2]} cost={r['cost']}: rr {r['rr']['test']:+.2f} logreach {r['logreach']['test']:+.2f}")
    el, stp = fig_pot_select()
    for t, r in stp.items():
        say(f"== stopping rule {t}: " + " ".join(f"{k}={v:+.3f}" for k, v in r.items()))
    for t, r in el.items():
        say(f"== forecast 50k->2M {t}: " + " ".join(f"{n}={v[2_000_000]:+.2f}" for n, v in r.items()))
    sel = selection_sim(["mix_logreach", "mix_rr", "rel_D", "rel_O", "practice_sr", "lp50", "door"])
    for k, v in sel.items():
        say(f"== select {k}: rho={v['rho']:+.2f} picked={v['picked']:+.3f} oracle={v['oracle']:+.3f} random={v['random']:+.3f} n={v['n']} {v['names']}")
    if os.path.exists(os.path.join(RES, "early.pt")):
        out, scm = fig_pot_early()
        say("== early (50k, 3 door tasks, 18 runs): scratch AUC " + str({t: round(v, 3) for t, v in scm.items()}))
        for (c, v), r in sorted(out.items(), key=lambda kv: -kv[1]["gain"]):
            say(f"   {c:14s} {v:9s} gain={r['gain']:+.3f} se={r['se']:.3f} p={r['p']:.3g}")
    if os.path.exists(os.path.join(RES, "rehearse.pt")):
        rc = rehearse_cols()
        out, _ = early_eval()
        rows, _, _ = metric_rows()
        cs = sorted({c for c, _ in out})
        for v in ["full", "nocritic"]:
            y = np.array([out[(c, v)]["gain"] for c in cs])
            for k in ["door", "mix_logreach", "mix_ev_logreach", "rel_D"]:
                xa = np.array([rc[(v, k)][(c, 50_000)] for c in cs])
                xb = np.array([np.mean([rows[(c, ps, 50_000)].get(k, np.nan) for ps in range(3)]) for c in cs])
                say(f"== rehearsal 50k GT={v:8s} {k:16s} after={spearman(xa, y):+.2f} before={spearman(xb, y):+.2f}")
    if os.path.exists(os.path.join(RES, "gt_cov.pt")):
        co = fig_pot_cov()
        say("== pretraining objectives (critic reset gains; full_door = full transfer on 3 door tasks)")
        for (c, st), r in sorted(co.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            say(f"   {c:14s} {st // 1000:5d}k D={r['D']:+.3f} L={r['L']:+.3f} O={r['O']:+.3f} test={r['test']:+.3f} "
                f"frontier={r['frontier']:+.2f} full_door={r.get('full_door', np.nan):+.3f}")
    fig_pot_heldout()
    open(os.path.join(RES, "potential.txt"), "w").write("\n".join(lines))


def rehearse_cols(heldout=False):
    d = load("rehearse")
    ref = load("metrics")["ref"]
    zok = ref["uni"]["zrate"].mean(0) > 0
    u_ev = ref["uni"]["evrate"].mean(0)
    out = {}
    for (name, st), e in [(k, v) for k, v in d.items() if isinstance(k, tuple)]:
        if (name == "test") != heldout:
            continue
        cols = {"mix_logreach": torch.log(e["mix"]["zrate"][:, zok] + 1e-4).mean(1), "mix_rr": e["mix"]["rr"],
                "mix_ev_logreach": torch.log(e["mix"]["evrate"][:, u_ev > 0] + 1e-4).mean(1),
                "door": e["pol"]["evrate"][:, 8], "rel_D": torch.log(e["mix"]["evrate"][:, [8, 11]] + 1e-4).mean(1),
                "rel_O": torch.log(e["mix_open"]["evrate"][:, [3, 5, 6, 7, 11]] + 1e-4).mean(1), "ent": e["entropy"]}
        for i, r in enumerate(e["runs"]):
            for k, v in cols.items():
                out.setdefault((r["variant"], k), {}).setdefault((r["cond"], st), []).append(float(v[i]))
    return {k: {g: np.mean(v) for g, v in t.items()} for k, t in out.items()}


def cov_eval():
    import learn
    sc, ck, B = gt_table()
    full = {}
    for name in ["gt_cov", "gt_cov2"]:
        if not os.path.exists(os.path.join(RES, name + ".pt")):
            continue
        d = load(name)
        m = metrics(d["log"])
        for i, r in enumerate(d["runs"]):
            if r["variant"] == "nocritic":
                ck.setdefault((r["cond"], r["pseed"], r["pre_step"]), {})[r["gt_task"]] = (m["auc"][i], m["t80"][i])
            else:
                full.setdefault((r["cond"], r["pre_step"]), {}).setdefault(r["gt_task"], []).append(m["auc"][i])
    esc = early_table()[("scratch", "scratch")] if os.path.exists(os.path.join(RES, "early.pt")) else None
    fams = dict(learn.GT_TASKS)
    fams["test"] = [t for f in learn.GT_TASKS.values() for t in f if t not in learn.GT_DEV]
    conds = ["explore", "impact", "smm", "cover", "cover_pos", "enter_right", "open_door", "mix"] + [c for c in learn.COND if c not in
                                                                                                    ("explore", "impact", "enter_right", "open_door", "mix")]
    out = {}
    for c in conds:
        for st in learn.COV_STEPS:
            rows = [ck[(c, ps, st)] for ps in range(3) if (c, ps, st) in ck]
            if len(rows) < 3:
                continue
            r = {f: np.mean([np.mean([row[t][0] - sc[t]["auc"] for t in ts]) for row in rows]) for f, ts in fams.items()}
            r["frontier"] = np.mean([sum(np.isfinite(row[t][1]) for t in sum(learn.GT_TASKS.values(), [])) for row in rows]) - \
                sum(sc[t]["learned"] for t in sum(learn.GT_TASKS.values(), []))
            if (c, st) in full and esc is not None:
                r["full_door"] = np.mean([np.mean(full[(c, st)][t]) - esc[t].mean() for t in full[(c, st)]])
            out[(c, st)] = r
    return out


def fig_pot_cov(name="fig_pot_cov"):
    out = cov_eval()
    conds = ["explore", "impact", "smm", "cover", "cover_pos", "enter_right", "open_door"]
    cols = {"explore": MUTED, "impact": SERIES[4], "smm": SERIES[3], "cover": SERIES[1], "cover_pos": SERIES[7], "enter_right": SERIES[2],
            "open_door": SERIES[0]}
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, (f, title) in zip(axs, [("D", "Door-chain tasks (11)"), ("L", "Left-room tasks (6)"), ("O", "Door-open world tasks (6)"),
                                     ("full_door", "Full transfer, 3 door tasks")]):
        w = 0.12
        for j, c in enumerate(conds):
            v = [out[(c, st)].get(f, np.nan) if (c, st) in out else np.nan for st in [250_000, 1_000_000, 2_000_000]]
            ax.bar(np.arange(3) + (j - 3) * w, v, w, color=cols[c], label=c)
        ax.set_xticks(range(3), ["250k", "1M", "2M"])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("pretraining steps")
    axs[0].set_ylabel("AUC gain vs scratch")
    axs[0].legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def story():
    import learn
    lines = []
    say = lambda s: (print(s), lines.append(s))
    parts = [pot_data("cond", ho) for ho in (False, True)]
    lab = ["14 practice tasks x 6 checkpoints", "8 held-out practice tasks x 3 checkpoints"]

    fig, ax = plt.subplots(figsize=(5.6, 4))
    for j, (M, Y, meta, *_) in enumerate(parts):
        x, y = M["practice_sr"], Y["test"]["gain"]
        ax.scatter(x, y, color=SERIES[j], s=18, label=f"{lab[j]}: rho {spearman(x, y):+.2f}")
        say(f"step2 practice success vs held-out gain ({lab[j]}): rho {spearman(x, y):+.2f}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("success rate on the practice task")
    ax.set_ylabel("AUC gain on 15 held-out tasks")
    ax.set_title("Practice success does not predict learnability", loc="left")
    ax.legend(fontsize=7.5)
    fig.savefig(os.path.join(FIG, "fig_story_2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    res = pot_eval()[0]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.1]})
    for j, (M, Y, meta, *_) in enumerate(parts):
        x, y = M["mix_logreach"], Y["test"]["gain"]
        axs[0].scatter(x, y, color=SERIES[j], s=18, label=f"{lab[j]}: rho {spearman(x, y):+.2f}")
        say(f"step3 coverage vs held-out gain ({lab[j]}): rho {spearman(x, y):+.2f}")
    axs[0].axhline(0, color=INK2, lw=0.8)
    axs[0].set_xlabel("coverage: mean over 30 abstract states of log P(reach in one episode)")
    axs[0].set_ylabel("AUC gain on 15 held-out tasks")
    axs[0].set_title("Coverage of the noisy policy predicts learnability", loc="left")
    axs[0].legend(fontsize=7.5)
    rows = [("practice success", "practice_sr", "0 env steps"), ("policy entropy", "ent", "16k states"), ("representation probe", "z_r2_h", "16k states"),
            ("short learning probe", "lp50", "300k env steps + gradients"), ("finetune on 8 dev tasks", "ds_full_dev", "16M env steps"),
            ("coverage (50% random actions)", "mix_logreach", "409.6k env steps, no gradients; 3.2k gives rho 0.59")]
    v = [res[n]["test"] for _, n, _ in rows]
    ci = np.array([res[n]["ci"] for _, n, _ in rows])
    y = np.arange(len(rows))[::-1]
    axs[1].barh(y, v, color=[MUTED, MUTED, SERIES[0], SERIES[6], INK, SERIES[1]], height=0.6)
    axs[1].hlines(y, ci[:, 0], ci[:, 1], color=INK2, lw=1)
    axs[1].set_yticks(y, [f"{a}\n(cost: {c})" for a, _, c in rows], fontsize=8)
    axs[1].axvline(0, color=INK2, lw=0.8)
    axs[1].set_xlim(-0.5, 1)
    axs[1].set_xlabel("Spearman rho with held-out learnability (90% CI)")
    axs[1].set_title("Accuracy and cost per checkpoint", loc="left")
    for (a, n, c), val in zip(rows, v):
        say(f"step3 metric {a:32s} rho {val:+.2f} ci [{res[n]['ci'][0]:+.2f}, {res[n]['ci'][1]:+.2f}] cost {c}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_story_3.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    stp = stopping(["mix_logreach"])["test"]
    say(f"step3 stopping rule: oracle {stp['oracle']:+.3f} coverage peak {stp['mix_logreach']:+.3f} random step {stp['random step']:+.3f} always 2M {stp['always 2M']:+.3f}")

    co = cov_eval()
    mrows, _, _ = metric_rows()
    steps = learn.COV_STEPS
    unlock = {}
    for name, cs in [("pretrain_cov", ["cover", "smm"]), ("pretrain_cov2", ["cover_pos"])]:
        d = load(name)
        for c in cs:
            unlock[c] = [float(d["reach"][st]["pol"]["evrate"][[i for i, r in enumerate(d["reach"][st]["runs"]) if r["cond"] == c], 8].mean()) for st in steps]
    for c in ["explore", "impact"]:
        unlock[c] = [np.mean([mrows[(c, ps, st)]["door"] for ps in range(3)]) for st in steps]
    names = {"cover": "coverage", "cover_pos": "coverage, no door variable", "smm": "SMM", "explore": "explore (new cells)", "impact": "impact (new configs)"}
    cols = {"cover": SERIES[1], "cover_pos": SERIES[7], "smm": SERIES[3], "explore": MUTED, "impact": SERIES[4], "hand": SERIES[2]}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.2), gridspec_kw={"width_ratios": [1, 1.2, 1.3]})
    for c in ["cover", "cover_pos", "smm", "impact", "explore"]:
        axs[0].plot([s / 1e6 for s in steps], unlock[c], marker="o", color=cols[c], label=names[c])
        say(f"step4 door-open rate without task reward {names[c]:28s} " + " ".join(f"{s // 1000}k={u:.2f}" for s, u in zip(steps, unlock[c])))
    axs[0].set_xlabel("practice steps (M)")
    axs[0].set_ylabel("episodes that open the door")
    axs[0].set_ylim(0, 1)
    axs[0].set_title("No task reward: who learns to open the door?", loc="left")
    axs[0].legend(fontsize=7.5)
    hand = [c for c in learn.COND if c not in ("explore", "impact")]
    best = lambda f: max(co[(c, st)][f] for c in hand for st in steps if (c, st) in co)
    bars = [(names["cover"], co[("cover", 2_000_000)], cols["cover"]), (names["cover_pos"], co[("cover_pos", 2_000_000)], cols["cover_pos"]),
            ("best hand-designed practice\n(best per column, any step)", {"test": best("test"), "frontier": best("frontier")}, cols["hand"]),
            (names["impact"], co[("impact", 2_000_000)], cols["impact"]), (names["explore"], co[("explore", 2_000_000)], cols["explore"]),
            (names["smm"], co[("smm", 2_000_000)], cols["smm"])]
    y = np.arange(len(bars))[::-1]
    axs[1].barh(y, [b[1]["test"] for b in bars], color=[b[2] for b in bars], height=0.6)
    for yy, b in zip(y, bars):
        axs[1].text(max(b[1]["test"], 0) + 0.01, yy, f"+{b[1]['frontier']:.1f} tasks learned" if b[1]["frontier"] >= 0 else f"{b[1]['frontier']:.1f} tasks learned",
                    va="center", fontsize=7.5, color=INK2)
        say(f"step4 {b[0].splitlines()[0]:28s} held-out gain {b[1]['test']:+.3f} extra tasks learned {b[1]['frontier']:+.2f}")
    axs[1].set_yticks(y, [b[0] for b in bars], fontsize=8)
    axs[1].axvline(0, color=INK2, lw=0.8)
    axs[1].set_xlim(-0.05, 0.6)
    axs[1].set_xlabel("AUC gain on 15 held-out tasks (after 2M practice steps, critic reset)")
    axs[1].set_title("Downstream learnability", loc="left")
    sc, ck, B = gt_table()
    dcov = load("gt_cov")
    mm = metrics(dcov["log"])
    cov_auc = {}
    for i, r in enumerate(dcov["runs"]):
        if r["variant"] == "nocritic" and r["cond"] == "cover" and r["pre_step"] == 2_000_000:
            cov_auc.setdefault(r["gt_task"], []).append(mm["auc"][i])
    hard = ["open_with_ball", "pick_ball2", "ball2_left", "ball_right", "box_right", "ball2_near_goal", "goal_with_ball"]
    old_best = {t: max(np.mean([ck[(c, ps, st)][t][0] for ps in range(3)]) for c in learn.COND for st in learn.GT_STEPS
                       if all((c, ps, st) in ck for ps in range(3))) for t in hard}
    w = 0.27
    for j, (lb, vals, col) in enumerate([("from scratch", [sc[t]["auc"] for t in hard], INK2),
                                         ("best of 252 earlier checkpoints", [old_best[t] for t in hard], cols["hand"]),
                                         ("after coverage practice", [np.mean(cov_auc[t]) for t in hard], cols["cover"])]):
        axs[2].bar(np.arange(len(hard)) + (j - 1) * w, vals, w, color=col, label=lb)
    for t in hard:
        say(f"step4 task {t:16s} scratch {sc[t]['auc']:.2f} best-of-252 {old_best[t]:.2f} coverage {np.mean(cov_auc[t]):.2f}")
    axs[2].set_xticks(range(len(hard)), [t.replace("_", "_\n", 1) for t in hard], fontsize=7.5)
    axs[2].set_ylabel("finetune AUC (2M steps)")
    axs[2].set_ylim(0, 1.05)
    axs[2].set_title("Tasks that PPO could not learn before", loc="left")
    axs[2].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_story_4.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    open(os.path.join(RES, "story.txt"), "w").write("\n".join(lines))


if __name__ == "__main__":
    {"all": make_all}.get(sys.argv[1], globals().get(sys.argv[1]))(*sys.argv[2:])
