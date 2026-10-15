import argparse
import functools
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from gridworld import INTRINSIC, N_ACTIONS, OBS_DIM, P, PREDICATES, World, feature_lut, featurize

DEV = "cuda"
HID = 256
ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "ckpt")
RES = os.path.join(ROOT, "results")
TASK = {p: i for i, p in enumerate(PREDICATES)} | INTRINSIC
KEYS = ["W1", "b1", "W2", "b2", "Wa", "ba", "Wv", "bv"]
TRUNK = ["W1", "b1", "W2", "b2"]


def fresh_params(R, sep=False):
    p = {}
    for k, (i, o) in [("W1", (OBS_DIM, HID)), ("W2", (HID, HID))] + ([("cW1", (OBS_DIM, HID)), ("cW2", (HID, HID))] if sep else []):
        w = torch.empty(R, i, o)
        for r in range(R):
            torch.nn.init.orthogonal_(w[r], math.sqrt(2))
        p[k] = w
    p["b1"], p["b2"] = torch.zeros(R, 1, HID), torch.zeros(R, 1, HID)
    p["Wa"], p["ba"] = torch.zeros(R, HID, N_ACTIONS), torch.zeros(R, 1, N_ACTIONS)
    p["Wv"], p["bv"] = torch.zeros(R, HID, 1), torch.zeros(R, 1, 1)
    if sep:
        p["cb1"], p["cb2"] = torch.zeros(R, 1, HID), torch.zeros(R, 1, HID)
    return {k: v.to(DEV) for k, v in p.items()}


@functools.lru_cache(4)
def load_ckpt(file):
    return torch.load(os.path.join(CKPT, file), map_location=DEV)


def load_slice(spec):
    return {k: v[spec["idx"]] for k, v in load_ckpt(spec["file"])["params"].items()}


def build_params(runs, sep=False):
    p = fresh_params(len(runs), sep)
    for r, run in enumerate(runs):
        spec = run.get("init")
        if spec is None:
            continue
        src = load_slice(spec)
        mode = spec.get("mode", "full")
        keys = list(src) if mode in ("full", "soft", "center") else KEYS[:6] if mode == "nocritic" else \
            [k for k in src if k not in ("Wv", "bv")] if mode == "vhead" else TRUNK
        for k in keys:
            w = src[k]
            if mode == "perm":
                w = w.flatten()[torch.randperm(w.numel(), device=DEV)].view_as(w)
            if mode == "shrink":
                w = spec.get("lam", 0.5) * w + (1 - spec.get("lam", 0.5)) * p[k][r]
            if mode == "soft" and k in ("Wa", "ba"):
                w = w * spec.get("lam", 0.1)
            p[k][r] = w
        if mode == "center":
            p["bv"][r] -= mean_value({k: p[k][r] for k in KEYS})
    for r, run in enumerate(runs):
        if "v_init" in run:
            p["bv"][r] += run["v_init"]
    return {k: v.clone().requires_grad_(True) for k, v in p.items()}


@torch.no_grad()
def mean_value(single, E=1024, n=200):
    lut = feature_lut(DEV)
    world = make_world([{"task": ["null"]}], E, 99, desync=False)
    p = {k: v[None] for k, v in single.items()}
    obs, tot = world.obs(), 0.0
    for _ in range(n):
        logits, v, _ = forward(p, featurize(obs, lut).view(1, E, -1))
        tot += v.mean().item()
        obs = world.step(torch.distributions.Categorical(logits=logits[0]).sample())[0]
    return tot / n


def build_prior(runs):
    if not any(run.get("prior") for run in runs):
        return None, None
    p = fresh_params(len(runs))
    on = torch.zeros(len(runs), device=DEV)
    for r, run in enumerate(runs):
        if run.get("prior"):
            src = load_slice(run["prior"])
            for k in KEYS:
                p[k][r] = src[k]
            on[r] = run["prior"].get("scale", 1.0)
    return p, on


def forward(p, x, detach=None):
    h1 = F.relu(torch.baddbmm(p["b1"], x, p["W1"]))
    h = F.relu(torch.baddbmm(p["b2"], h1, p["W2"]))
    hv = h if detach is None else torch.where(detach[:, None, None], h.detach(), h)
    if "cW1" in p:
        hv = F.relu(torch.baddbmm(p["cb2"], F.relu(torch.baddbmm(p["cb1"], x, p["cW1"])), p["cW2"]))
    return torch.baddbmm(p["ba"], h, p["Wa"]), torch.baddbmm(p["bv"], hv, p["Wv"]).squeeze(-1), h


def make_world(runs, E, seed, desync=True):
    task, size, ms, var, once = [], [], [], [], []
    for run in runs:
        ts = [TASK[t] if isinstance(t, str) else t for t in run["task"]]
        task += [ts[e % len(ts)] for e in range(E)]
        size += [run.get("size", 10)] * E
        ms += [run.get("max_steps", 200)] * E
        var += [{"base": 0, "key": 1, "open": 2}[run.get("world", "base")]] * E
        once += [run.get("once", False)] * E
    return World(torch.tensor(task), torch.tensor(size), torch.tensor(ms), DEV, seed=seed, desync=desync,
                 variant=torch.tensor(var), once=torch.tensor(once))


def train(runs, steps, E=256, T=32, lr=5e-4, epochs=4, n_mb=4, ent_coef=0.01, gamma=0.99, lam=0.95,
          clip=0.2, max_norm=0.5, seed=0, save=None, save_at=(), norm_adv=True, adv_floor=0.0, use_critic=True, sep=False):
    torch.manual_seed(seed)
    R = len(runs)
    lut = feature_lut(DEV)
    params = build_params(runs, sep)
    prior, prior_on = build_prior(runs)
    opt = torch.optim.Adam(params.values(), lr=lr, eps=1e-5)
    world = make_world(runs, E, seed)
    obs = world.obs()
    n_iter = steps // (E * T)
    log = {k: torch.zeros(R, n_iter) for k in ["episodes", "success", "ret", "entropy", "vloss", "len"]}
    log["hits"] = torch.zeros(R, n_iter, P)
    buf_obs = torch.zeros(T, R * E, obs.shape[1], dtype=torch.uint8, device=DEV)
    buf_a = torch.zeros(T, R, E, dtype=torch.long, device=DEV)
    buf_lp, buf_v, buf_r, buf_d = (torch.zeros(T, R, E, device=DEV) for _ in range(4))
    buf_pl = torch.zeros(T, R, E, N_ACTIONS, device=DEV) if prior is not None else None
    save_at = sorted(set(save_at))
    detach = torch.tensor([run.get("detach", False) for run in runs], device=DEV) if any(run.get("detach") for run in runs) else None
    warmup = torch.tensor([run.get("warmup", 0) for run in runs], device=DEV)
    vreset = torch.tensor([run.get("vreset", 0) for run in runs], device=DEV)
    intr = torch.tensor([{"smm": 1, "cover": 2, "cover_pos": 3}.get(run.get("intrinsic"), 0) for run in runs], device=DEV)
    if intr.any():
        clut = carry_lut()
        kind, rid = intr.repeat_interleave(E), world.ar // E
        zfn = lambda: torch.where(kind == 3, zpos(world, clut), zstate(world, clut))
        zvis = torch.zeros(R * E, NZP, dtype=torch.bool, device=DEV)
        zvis[world.ar, zfn()] = True
        rho, qhat = torch.full((R, NZP), 1 / NZ, device=DEV), torch.full((R, NZP), 0.1, device=DEV)
    t0 = time.time()
    for it in range(n_iter):
        if intr.any():
            occ_cnt, hit_cnt, ep_i = torch.zeros(R, NZP, device=DEV), torch.zeros(R, NZP, device=DEV), torch.zeros(R, device=DEV)
        ep = torch.zeros(R, device=DEV)
        sc, rt, ln = torch.zeros_like(ep), torch.zeros_like(ep), torch.zeros_like(ep)
        hits = torch.zeros(R, P, device=DEV)
        with torch.no_grad():
            for t in range(T):
                buf_obs[t] = obs
                x = featurize(obs, lut).view(R, E, -1)
                logits, v, _ = forward(params, x)
                if prior is not None:
                    pl = forward(prior, x)[0] * prior_on[:, None, None]
                    buf_pl[t] = pl
                    logits = logits + pl
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                buf_a[t], buf_lp[t], buf_v[t] = a, dist.log_prob(a), v
                if intr.any():
                    _, r, d, info = world.step(a.flatten(), reset=False)
                    z1 = zfn()
                    first = ~zvis[world.ar, z1]
                    zvis[world.ar, z1] = True
                    # SMM: -log state density per step; cover: 1/q on the first visit of an episode (gradient of sum_z log q)
                    r = r + torch.where(kind == 1, -torch.log(rho[rid, z1]) / 10,
                                        torch.where(kind >= 2, first * (1 / qhat[rid, z1]).clamp(max=1000) / 1000, 0.0))
                    occ_cnt.index_put_((rid, z1), torch.ones_like(r), accumulate=True)
                    if d.any():
                        hit_cnt += (zvis.view(R, E, NZP) & d.view(R, E, 1)).sum(1)
                        ep_i += d.view(R, E).sum(1)
                        world.reset_mask(d)
                        zvis &= ~d[:, None]
                        z0 = zfn()
                        zvis[world.ar, z0] |= d
                    obs = world.obs()
                else:
                    obs, r, d, info = world.step(a.flatten())
                buf_r[t], buf_d[t] = r.view(R, E), d.view(R, E).float()
                dm = d.view(R, E)
                ep += dm.sum(1)
                sc += (info["success"].view(R, E) & dm).sum(1)
                rt += (r.view(R, E) * dm).sum(1)
                ln += (info["t"].view(R, E) * dm).sum(1)
                hits += (info["hit"].view(R, E, P) & dm[..., None]).sum(1)
            if intr.any():
                rho = (0.9 * rho + 0.1 * occ_cnt / occ_cnt.sum(1, keepdim=True).clamp(min=1)).clamp(min=1e-6)
                qhat = torch.where(ep_i[:, None] > 0, 0.9 * qhat + 0.1 * hit_cnt / ep_i.clamp(min=1)[:, None], qhat)
            x = featurize(obs, lut).view(R, E, -1)
            last_v = forward(params, x)[1]
            adv = torch.zeros_like(buf_r)
            gae = torch.zeros(R, E, device=DEV)
            if not use_critic:
                buf_v.zero_()
                last_v = torch.zeros_like(last_v)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else buf_v[t + 1]
                nonterm = 1 - buf_d[t]
                delta = buf_r[t] + gamma * nv * nonterm - buf_v[t]
                gae = delta + gamma * lam * nonterm * gae
                adv[t] = gae
            ret = adv + buf_v
        N = T * E
        b_obs = buf_obs.view(T, R, E, -1).permute(1, 0, 2, 3).reshape(R, N, -1)
        b_a, b_lp, b_adv, b_ret = (z.permute(1, 0, 2).reshape(R, N) for z in (buf_a, buf_lp, adv, ret))
        if norm_adv:
            b_adv = (b_adv - b_adv.mean(1, keepdim=True)) / (b_adv.std(1, keepdim=True).clamp(min=adv_floor) + 1e-8)
        b_pl = buf_pl.permute(1, 0, 2, 3).reshape(R, N, N_ACTIONS) if prior is not None else None
        ent_sum, vl_sum = torch.zeros(R, device=DEV), torch.zeros(R, device=DEV)
        mb = N // n_mb
        actor_on = (warmup <= it * E * T).float()
        for _ in range(epochs):
            perm = torch.randperm(N, device=DEV)
            for k in range(n_mb):
                idx = perm[k * mb:(k + 1) * mb]
                x = featurize(b_obs[:, idx], lut)
                logits, v, _ = forward(params, x, detach)
                if b_pl is not None:
                    logits = logits + b_pl[:, idx]
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(b_a[:, idx])
                ratio = (lp - b_lp[:, idx]).exp()
                A = b_adv[:, idx]
                pg = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A).mean(1)
                vl = (v - b_ret[:, idx]).pow(2).mean(1)
                en = dist.entropy().mean(1)
                loss = (actor_on * (pg - ent_coef * en) + 0.5 * vl).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():
                    sq = sum(g.grad.pow(2).flatten(1).sum(1) for g in params.values())
                    scale = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
                    for g in params.values():
                        g.grad.mul_(scale.view(-1, *([1] * (g.dim() - 1))))
                opt.step()
                ent_sum += en.detach()
                vl_sum += vl.detach()
        log["episodes"][:, it] = ep.cpu()
        log["success"][:, it] = sc.cpu()
        log["ret"][:, it] = rt.cpu()
        log["len"][:, it] = ln.cpu()
        log["entropy"][:, it] = (ent_sum / (epochs * n_mb)).cpu()
        log["vloss"][:, it] = (vl_sum / (epochs * n_mb)).cpu()
        log["hits"][:, it] = hits.cpu()
        done_steps = (it + 1) * E * T
        hit_reset = (vreset > 0) & (done_steps // vreset.clamp(min=1) != (done_steps - E * T) // vreset.clamp(min=1))
        if hit_reset.any():
            with torch.no_grad():
                for k in ("Wv", "bv"):
                    keep = (~hit_reset).float().view(-1, *([1] * (params[k].dim() - 1)))
                    params[k].mul_(keep)
                    for st in ("exp_avg", "exp_avg_sq"):
                        if st in opt.state[params[k]]:
                            opt.state[params[k]][st].mul_(keep)
        if save and save_at and done_steps >= save_at[0]:
            torch.save({"params": {k: v.detach().clone() for k, v in params.items()}, "runs": runs, "step": done_steps},
                       os.path.join(CKPT, f"{save}_{save_at[0]}.pt"))
            save_at.pop(0)
        if it % 50 == 0 or it == n_iter - 1:
            sr = (log["success"][:, it] / log["episodes"][:, it].clamp(min=1)).mean().item()
            print(f"  it {it}/{n_iter} steps/run {done_steps} mean_sr {sr:.3f} fps {R * done_steps / (time.time() - t0):.0f}", flush=True)
    if save:
        torch.save({"params": {k: v.detach().clone() for k, v in params.items()}, "runs": runs, "step": n_iter * E * T},
                   os.path.join(CKPT, f"{save}_final.pt"))
    log["steps_per_iter"] = E * T
    log["wall"] = time.time() - t0
    return params, log


@torch.no_grad()
def probe(params, R, size=10, max_steps=200, E=1024, n_eps=8, greedy=False, seed=123, prior=None):
    lut = feature_lut(DEV)
    world = make_world([{"task": ["null"], "size": size, "max_steps": max_steps}] * R, E, seed, desync=False)
    obs = world.obs()
    hits = torch.zeros(R, P, device=DEV)
    eps = torch.zeros(R, device=DEV)
    stats = {k: torch.zeros(R, device=DEV) for k in ["entropy", "interact", "novel", "cells", "pickups", "toggles", "drops"]}
    steps = n_eps * max_steps
    for _ in range(steps):
        x = featurize(obs, lut).view(R, E, -1)
        logits = forward(params, x)[0]
        if prior is not None:
            logits = logits + prior
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if greedy else dist.sample()
        stats["entropy"] += dist.entropy().mean(1)
        front = world.grid[world.ar, (world.pos + world.dirs[world.dir])[:, 1], (world.pos + world.dirs[world.dir])[:, 0]].long() & 7
        af = a.flatten()
        stats["pickups"] += ((af == 3) & (front >= 3) & (front <= 5) & (world.carry == 0)).view(R, E).float().mean(1)
        stats["toggles"] += ((af == 5) & (front == 2)).view(R, E).float().mean(1)
        stats["drops"] += ((af == 4) & (front == 0) & (world.carry != 0)).view(R, E).float().mean(1)
        n_before = world.n_seen.clone()
        obs, r, d, info = world.step(af)
        stats["novel"] += ((world.n_seen - n_before).clamp(min=0) * ~d).view(R, E).float().mean(1)
        dm = d.view(R, E)
        stats["cells"] += (info["cells"].view(R, E) * dm).sum(1)
        hits += (info["hit"].view(R, E, P) & dm[..., None]).sum(1)
        eps += dm.sum(1)
    out = {"hit": (hits / eps[:, None]).cpu()}
    for k, v in stats.items():
        out[k] = (v / (steps if k == "entropy" else n_eps)).cpu()
    out["cells"] = (stats["cells"] / eps).cpu()
    out["interact"] = out["pickups"] + out["toggles"] + out["drops"]
    return out


@torch.no_grad()
def rep_stats(params, R, size=10, max_steps=200, E=512, n=64, seed=7):
    lut = feature_lut(DEV)
    world = make_world([{"task": ["null"], "size": size, "max_steps": max_steps}] * R, E, seed)
    feats = []
    obs = world.obs()
    for i in range(n):
        x = featurize(obs, lut).view(R, E, -1)
        logits, v, h = forward(params, x)
        if i % 8 == 0:
            feats.append(h)
        a = torch.distributions.Categorical(logits=logits).sample()
        obs = world.step(a.flatten())[0]
    h = torch.cat(feats, 1)
    sv = torch.linalg.svdvals(h)
    c = (sv.cumsum(1) / sv.sum(1, keepdim=True))
    srank = (c < 0.99).sum(1) + 1
    dormant = ((h.mean(1) / (h.mean((1, 2), keepdim=True).squeeze(-1) + 1e-8)) <= 0.025).float().mean(1)
    wn = sum(params[k].flatten(1).pow(2).sum(1) for k in TRUNK).sqrt()
    return {"srank": srank.float().cpu(), "dormant": dormant.cpu(), "wnorm": wn.cpu(), "feat_norm": h.norm(dim=-1).mean(1).cpu()}


EVENTS = ["pick_key", "pick_ball", "pick_box", "pick_ball2", "drop_key", "drop_ball", "drop_box", "drop_ball2",
          "unlock", "close", "reopen", "enter_right", "enter_left"]
NZ = 30


def zstate(world, clut):
    door = world.grid[world.ar, world.door[:, 1], world.door[:, 0]].long() >> 6
    side = (world.pos[:, 0] > world.wx).long()
    return clut[world.carry] * 6 + door * 2 + side


NZP = 80


def zpos(world, clut):
    from gridworld import PAD
    cell = ((world.pos - PAD - 1).clamp(0, 7) // 2)
    return clut[world.carry] * 16 + cell[:, 1] * 4 + cell[:, 0]


def z_events(z0, z1):
    c0, d0, s0, c1, d1, s1 = z0 // 6, (z0 // 2) % 3, z0 % 2, z1 // 6, (z1 // 2) % 3, z1 % 2
    ev = [(c0 == 0) & (c1 == x) for x in range(1, 5)] + [(c0 == x) & (c1 == 0) for x in range(1, 5)]
    ev += [(d0 == 2) & (d1 == 0), (d0 == 0) & (d1 == 1), (d0 == 1) & (d1 == 0), (s0 == 0) & (s1 == 1), (s0 == 1) & (s1 == 0)]
    return torch.stack(ev, 1)


def effects(world):
    from gridworld import BALL, BOX, DOOR, EMPTY, GOAL, KEY, LOCKED, OPEN
    fwd = world.pos + world.dirs[world.dir]
    front = world.grid[world.ar, fwd[:, 1], fwd[:, 0]].long()
    ft, fc, fs = front & 7, (front >> 3) & 7, front >> 6
    key_ok = ((world.carry & 7) == KEY) & (((world.carry >> 3) & 7) == fc)
    pick = ((ft == KEY) | (ft == BALL) | (ft == BOX)) & (world.carry == 0)
    drop = (ft == EMPTY) & (world.carry != 0)
    tog = (ft == DOOR) & ((fs != LOCKED) | key_ok)
    unlock = (ft == DOOR) & (fs == LOCKED) & key_ok
    fwd_ok = (ft == EMPTY) | (ft == GOAL) | ((ft == DOOR) & (fs == OPEN))
    return pick, drop, tog, unlock, fwd_ok


def carry_lut():
    from gridworld import BALL2_C, BALL_C, BOX_C, KEY_C
    c = torch.zeros(256, dtype=torch.long, device=DEV)
    for i, code in enumerate([KEY_C, BALL_C, BOX_C, BALL2_C]):
        c[code] = i + 1
    return c


@torch.no_grad()
def rollout_z(p, R, E=512, n_eps=4, mode="policy", k=40, eps=0.5, variant="base", seed=11, sz=None):
    lut, clut = feature_lut(DEV), carry_lut()
    world = make_world([{"task": ["null"], "world": variant}] * R, E, seed, desync=False)
    B, obs = R * E, world.obs()
    zvis = torch.zeros(B, NZ, dtype=torch.bool, device=DEV)
    evs = torch.zeros(B, len(EVENTS), dtype=torch.bool, device=DEV)
    acc = {k_: torch.zeros(R, device=DEV) for k_ in ["eps", "depth", "nz", "rr"]}
    zrate, evrate, occ = torch.zeros(R, NZ, device=DEV), torch.zeros(R, len(EVENTS), device=DEV), torch.zeros(R, NZ, device=DEV)
    aff = torch.zeros(R, 5, 2, 2, device=DEV)
    dv, dvn = torch.zeros(R, len(EVENTS), device=DEV), torch.zeros(R, len(EVENTS), device=DEV)
    z = zstate(world, clut)
    zvis[torch.arange(B, device=DEV), z] = True
    for _ in range(n_eps * 200):
        logits, v, _ = forward(p, featurize(obs, lut).view(R, E, -1))
        probs = logits.softmax(-1).view(B, -1)
        pol = torch.distributions.Categorical(logits=logits.view(B, -1)).sample()
        uni = torch.randint(0, N_ACTIONS, (B,), device=DEV)
        a = {"policy": pol, "uniform": uni, "jump": torch.where(world.t < k, pol, uni),
             "mix": torch.where(torch.rand(B, device=DEV) < eps, uni, pol)}[mode]
        if mode == "policy":
            for j, (m, act) in enumerate(zip(effects(world), [3, 4, 5, 5, 2])):
                for s_, mm in enumerate([~m, m]):
                    aff[:, j, s_, 0] += (probs[:, act] * mm).view(R, E).sum(1)
                    aff[:, j, s_, 1] += mm.view(R, E).sum(1)
        world.step(a, reset=False)
        z1 = zstate(world, clut)
        e = z_events(z, z1)
        if mode == "policy" and e.any():
            v1 = forward(p, featurize(world.obs(), lut).view(R, E, -1))[1].view(B)
            dv += ((v1 - v.view(B))[:, None] * e).view(R, E, -1).sum(1)
            dvn += e.view(R, E, -1).sum(1).float()
        evs |= e
        zvis[torch.arange(B, device=DEV), z1] = True
        occ += torch.nn.functional.one_hot(z1, NZ).view(R, E, NZ).sum(1)
        d = world.t >= world.max_steps
        if d.any():
            dm = d.view(R, E)
            acc["eps"] += dm.sum(1)
            zrate += (zvis.view(R, E, NZ) & dm[..., None]).sum(1)
            evrate += (evs.view(R, E, -1) & dm[..., None]).sum(1)
            acc["nz"] += (zvis.sum(1).view(R, E) * dm).sum(1)
            if sz is not None:
                acc["depth"] += ((zvis * sz).max(1)[0].view(R, E) * dm).sum(1)
                acc["rr"] += ((zvis * sz).sum(1).view(R, E) * dm).sum(1)
            world.reset_mask(d)
            zvis &= ~d[:, None]
            evs &= ~d[:, None]
            z1 = zstate(world, clut)
            zvis[torch.arange(B, device=DEV), z1] = d | zvis[torch.arange(B, device=DEV), z1]
        z, obs = z1, world.obs()
    n = acc["eps"].clamp(min=1)
    out = {"zrate": zrate / n[:, None], "evrate": evrate / n[:, None], "occ": occ / occ.sum(1, keepdim=True),
           "nz": acc["nz"] / n, "depth": acc["depth"] / n, "rr": acc["rr"] / n}
    if mode == "policy":
        out["aff"] = aff[..., 0] / aff[..., 1].clamp(min=1)
        out["dv"] = dv / dvn.clamp(min=1)
        out["dvn"] = dvn
    return {k_: v_.cpu() for k_, v_ in out.items()}


@torch.no_grad()
def rand_dataset(N=16384, E=256, T=400, H=16, seed=21):
    clut = carry_lut()
    world = make_world([{"task": ["null"]}], E, seed, desync=True)
    O, O1, A, Z, AF, EV, D = [], [], [], [], [], [], []
    obs = world.obs()
    for _ in range(T):
        z = zstate(world, clut)
        AF.append(torch.stack(effects(world)[:4], 1))
        a = torch.randint(0, N_ACTIONS, (E,), device=DEV)
        O.append(obs)
        world.step(a, reset=False)
        O1.append(world.obs())
        z1 = zstate(world, clut)
        EV.append(z_events(z, z1))
        A.append(a)
        Z.append(z)
        d = world.t >= world.max_steps
        D.append(d)
        world.reset_mask(d)
        obs = world.obs()
    EV, D = torch.stack(EV), torch.stack(D)
    dist = torch.full((E, len(EVENTS)), 10 ** 6, device=DEV)
    FUT = torch.zeros(T, E, len(EVENTS), dtype=torch.bool, device=DEV)
    for t in reversed(range(T)):
        dist = torch.where(D[t][:, None], torch.full_like(dist, 10 ** 6), dist + 1)
        dist = torch.where(EV[t], torch.zeros_like(dist), dist)
        FUT[t] = dist < H
    sel = torch.randperm(T * E, generator=torch.Generator(device=DEV).manual_seed(seed), device=DEV)[:N]
    flat = lambda x: torch.stack(x).flatten(0, 1)[sel] if isinstance(x, list) else x.flatten(0, 1)[sel]
    return {"obs": flat(O), "obs1": flat(O1), "a": flat(A), "z": flat(Z), "aff": flat(AF), "fut": flat(FUT)}


def ridge_r2(H, Y, n_tr, lam=1e-3):
    H = torch.cat([H, torch.ones_like(H[..., :1])], -1)
    Htr, Hte, Ytr, Yte = H[:, :n_tr], H[:, n_tr:], Y[:n_tr], Y[n_tr:]
    A = Htr.transpose(1, 2) @ Htr
    A = A + lam * A.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(A.shape[-1], device=DEV)
    W = torch.linalg.solve(A, Htr.transpose(1, 2) @ Ytr.expand(len(H), -1, -1))
    res = ((Hte @ W - Yte) ** 2).sum(1)
    sst = ((Yte - Ytr.mean(0)) ** 2).sum(0)
    return 1 - res / sst.clamp(min=1e-6)


def feats(p, obs, lut, chunk=8):
    x = featurize(obs, lut)
    out = []
    for i in range(0, len(p["W1"]), chunk):
        q = {k: v[i:i + chunk] for k, v in p.items()}
        h1 = F.relu(torch.baddbmm(q["b1"], x.expand(len(q["W1"]), -1, -1), q["W1"]))
        h = F.relu(torch.baddbmm(q["b2"], h1, q["W2"]))
        out.append((h1, h, torch.baddbmm(q["ba"], h, q["Wa"]), torch.baddbmm(q["bv"], h, q["Wv"]).squeeze(-1)))
    return [torch.cat([o[j] for o in out]) for j in range(4)]


def cf_obs(obs, kind):
    from gridworld import BALL, BOX, DOOR, KEY, KEY_C, LOCKED, OPEN, WALL_C
    o = obs.clone()
    t = obs[:, :49].long() & 7
    if kind == "front":
        m = (t[:, 38] == KEY) | (t[:, 38] == BALL) | (t[:, 38] == BOX)
        o[:, 38] = 0
    elif kind == "carry":
        m = torch.ones(len(obs), dtype=torch.bool, device=DEV)
        o[:, 49] = torch.where(obs[:, 49] == 0, KEY_C, 0)
    elif kind == "door":
        dm = t == DOOR
        m = dm.any(1)
        st = obs[:, :49].long() >> 6
        flip = torch.where(st == LOCKED, OPEN, LOCKED)
        o[:, :49] = torch.where(dm, ((obs[:, :49].long() & 63) | (flip << 6)), obs[:, :49].long()).to(torch.uint8)
    else:
        m = t[:, 0] == 0
        o[:, 0] = WALL_C
    return o, m


def plasticity(p, obs, lut, steps=200, bs=1024, lr=1e-3, seed=5):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = featurize(obs, lut)
    Wt1 = torch.randn(x.shape[1], 64, generator=g, device=DEV) / math.sqrt(x.shape[1] / 8)
    Wt2 = torch.randn(64, 8, generator=g, device=DEV) / 8
    y = torch.tanh(torch.relu(x @ Wt1) @ Wt2)
    y = (y - y.mean(0)) / (y.std(0) + 1e-6)
    R = len(p["W1"])
    q = {k: p[k].detach().clone().requires_grad_(True) for k in TRUNK}
    q["Wo"] = torch.zeros(R, HID, 8, device=DEV, requires_grad=True)
    q["bo"] = torch.zeros(R, 1, 8, device=DEV, requires_grad=True)
    opt = torch.optim.Adam(q.values(), lr=lr)
    n_te = 4096
    xtr, ytr, xte, yte = x[n_te:], y[n_te:], x[:n_te], y[:n_te]

    def loss_fn(xb, yb):
        h = F.relu(torch.baddbmm(q["b2"], F.relu(torch.baddbmm(q["b1"], xb.expand(R, -1, -1), q["W1"])), q["W2"]))
        return ((torch.baddbmm(q["bo"], h, q["Wo"]) - yb) ** 2).mean((1, 2))

    curve = []
    for s in range(steps):
        idx = torch.randint(0, len(xtr), (bs,), generator=g, device=DEV)
        loss = loss_fn(xtr[idx], ytr[idx])
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        if s in (9, 49, steps - 1):
            with torch.no_grad():
                curve.append(loss_fn(xte, yte).cpu())
    return torch.stack(curve, 1)


def cka(a, b):
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    num = (a.transpose(1, 2) @ b).pow(2).sum((1, 2))
    return num / ((a.transpose(1, 2) @ a).pow(2).sum((1, 2)).sqrt() * (b.transpose(1, 2) @ b).pow(2).sum((1, 2)).sqrt())


def measure(p, ref, ds, p0=None):
    lut = feature_lut(DEV)
    R = len(p["W1"])
    out, cost = {}, {}

    def timed(name, steps, fn):
        torch.cuda.synchronize()
        t0 = time.time()
        r = fn()
        torch.cuda.synchronize()
        cost[name] = {"sec": (time.time() - t0) / R, "env_steps": steps}
        return r

    pol = timed("rollout", 512 * 800, lambda: rollout_z(p, R, sz=ref["sz"]))
    out["roll"] = pol
    out["jump"] = timed("jump", 512 * 800, lambda: rollout_z(p, R, mode="jump", sz=ref["sz"]))
    out["mix"] = timed("mix", 512 * 800, lambda: rollout_z(p, R, mode="mix", sz=ref["sz"]))
    out["roll_open"] = timed("rollout_open", 512 * 800, lambda: rollout_z(p, R, variant="open", sz=ref["sz_open"]))

    def rep():
        h1, h, lg, v = feats(p, ds["obs"], lut)
        _, hn, _, _ = feats(p, ds["obs1"], lut)
        n_tr = int(0.75 * len(ds["obs"]))
        r = {"v_mean": v.mean(1).cpu(), "v_std": v.std(1).cpu(), "ent_ds": torch.distributions.Categorical(logits=lg).entropy().mean(1).cpu()}
        for nm, HH in [("h", h), ("h1", h1)]:
            r[f"aff_{nm}"] = ridge_r2(HH, ds["aff"].float(), n_tr).cpu()
            r[f"fut_{nm}"] = ridge_r2(HH, ds["fut"].float(), n_tr).cpu()
            r[f"inv_{nm}"] = ridge_r2(torch.cat([HH, hn if nm == "h" else feats(p, ds["obs1"], lut)[0]], -1),
                                      F.one_hot(ds["a"], N_ACTIONS).float(), n_tr).cpu()
            zl = torch.cat([F.one_hot(ds["z"] // 6, 5), F.one_hot((ds["z"] // 2) % 3, 3), F.one_hot(ds["z"] % 2, 2)], 1).float()
            r[f"z_{nm}"] = ridge_r2(HH, zl, n_tr).cpu()
        pa = lg.softmax(-1)
        for kind in ["front", "carry", "door", "ctrl"]:
            o2, m = cf_obs(ds["obs"], kind)
            _, h2, lg2, v2 = feats(p, o2, lut)
            kl = (pa * (pa.clamp(min=1e-8).log() - lg2.log_softmax(-1))).sum(-1)
            r[f"cf_kl_{kind}"] = (kl * m).sum(1).cpu() / m.sum().clamp(min=1).cpu()
            r[f"cf_h_{kind}"] = (((h2 - h).norm(dim=-1) / h.norm(dim=-1).clamp(min=1e-6)) * m).sum(1).cpu() / m.sum().clamp(min=1).cpu()
            r[f"cf_v_{kind}"] = ((v2 - v).abs() * m).sum(1).cpu() / m.sum().clamp(min=1).cpu()
        if p0 is not None:
            h01, h0, _, _ = feats(p0, ds["obs"], lut)
            r["cka0_h"] = cka(h0[:, :4096], h[:, :4096]).cpu()
            r["cka0_h1"] = cka(h01[:, :4096], h1[:, :4096]).cpu()
            r["drift"] = torch.stack([((p[k] - p0[k]).flatten(1).norm(dim=1) / p0[k].flatten(1).norm(dim=1).clamp(min=1e-6))
                                      for k in ["W1", "W2"]], 1).cpu()
        return r

    out["rep"] = timed("rep", 0, rep)
    out["plast"] = timed("plast", 0, lambda: plasticity(p, ds["obs"], lut))
    return out, cost


def metric_ref():
    torch.manual_seed(0)
    fresh = fresh_params(16)
    uni = rollout_z(fresh, 16, E=1024, n_eps=4, mode="uniform")
    uni_o = rollout_z(fresh, 16, E=1024, n_eps=4, mode="uniform", variant="open")
    sz = lambda u: -torch.log(u["zrate"].mean(0) + 1e-5)
    return fresh, {"sz": sz(uni).to(DEV), "sz_open": sz(uni_o).to(DEV), "uni": uni, "uni_open": uni_o}, rand_dataset()


def metrics_files(steps, file_of, p0, res, ref, ds):
    for st in steps:
        ck = torch.load(os.path.join(CKPT, file_of(st)), map_location=DEV)
        m, cost = measure({k: ck["params"][k] for k in KEYS}, ref, ds, p0)
        res[st] = {"runs": ck["runs"], "m": m, "cost": cost}
        print(file_of(st), cost, flush=True)
    return res


def exp_metrics(args):
    fresh, ref, ds = metric_ref()
    torch.manual_seed(0)
    p0 = fresh_params(len(COND) * 3)
    save_result("metrics", metrics_files(GT_STEPS, pre_file, p0, {"ref": ref, "fresh": measure(fresh, ref, ds)}, ref, ds))


def exp_metrics_test(args):
    _, ref, ds = metric_ref()
    torch.manual_seed(args.seed + 40000)
    p0 = fresh_params(len(TEST_COND) * 3)
    save_result("metrics_test", metrics_files(TEST_STEPS, test_file, p0, {"ref": ref}, ref, ds))


@torch.no_grad()
def grad_coherence(p, R, E=256, seed=31, gamma=0.99):
    lut, clut = feature_lut(DEV), carry_lut()
    world = make_world([{"task": ["null"]}] * R, E, seed, desync=False)
    B, obs = R * E, world.obs()
    z = zstate(world, clut)
    first = torch.full((B, len(EVENTS)), 10 ** 6, device=DEV)
    G1 = torch.zeros(R, len(EVENTS), HID, N_ACTIONS, device=DEV)
    hs, gs = [], []
    fisher = torch.zeros(R, device=DEV)
    for t in range(200):
        logits, _, h = forward(p, featurize(obs, lut).view(R, E, -1))
        pr = logits.softmax(-1)
        a = torch.distributions.Categorical(probs=pr).sample()
        g = F.one_hot(a, N_ACTIONS).float() - pr
        fisher += (h.pow(2).sum(-1) * (1 - pr.pow(2).sum(-1))).mean(1)
        hs.append(h.half())
        gs.append(g.half())
        world.step(a.flatten(), reset=False)
        z1 = zstate(world, clut)
        e = z_events(z, z1)
        first = torch.where(e & (first > t), torch.full_like(first, t), first)
        z = z1
    ret = torch.where(first < 10 ** 6, gamma ** first.clamp(max=200).float(), torch.zeros_like(first)).view(R, E, -1)
    adv = ret - ret.mean(1, keepdim=True)
    half = torch.arange(E, device=DEV) < E // 2
    Ga, Gb = torch.zeros(R, len(EVENTS), HID * N_ACTIONS, device=DEV), torch.zeros(R, len(EVENTS), HID * N_ACTIONS, device=DEV)
    ft = first.view(R, E, -1)
    for t in range(200):
        h, g = hs[t].float(), gs[t].float()
        live = (ft >= t).float() * adv
        outer = torch.einsum("reh,rea->reha", h, g).flatten(2)
        Ga += torch.einsum("rek,rew->rkw", live * half[None, :, None], outer)
        Gb += torch.einsum("rek,rew->rkw", live * ~half[None, :, None], outer)
    cos = F.cosine_similarity(Ga, Gb, dim=-1)
    rate = (first < 10 ** 6).float().view(R, E, -1).mean(1)
    return {"cos": cos.cpu(), "rate": rate.cpu(), "gnorm": (Ga + Gb).norm(dim=-1).cpu() / E, "fisher": (fisher / 200).cpu()}


def grad_files(steps, file_of, res):
    for st in steps:
        ck = torch.load(os.path.join(CKPT, file_of(st)), map_location=DEV)
        torch.cuda.synchronize()
        t0 = time.time()
        res[st] = {"runs": ck["runs"], "g": grad_coherence({k: ck["params"][k] for k in KEYS}, len(ck["runs"]))}
        torch.cuda.synchronize()
        res[st]["sec"] = (time.time() - t0) / len(ck["runs"])
        print(file_of(st), res[st]["sec"], flush=True)
    return res


def exp_metrics2(args):
    torch.manual_seed(0)
    save_result("metrics2", grad_files(GT_STEPS, pre_file, {"fresh": grad_coherence(fresh_params(16), 16)}))


def exp_metrics2_test(args):
    save_result("metrics2_test", grad_files(TEST_STEPS, test_file, {}))


def exp_mixsweep(args):
    fresh, ref, _ = metric_ref()
    grid = [(eps, E, n) for eps in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9] for E, n in [(512, 4)]] + [(0.5, E, n) for E, n in [(16, 1), (64, 1), (128, 2)]]
    res = {"grid": grid, "fresh": {g: rollout_z(fresh, 16, E=g[1], n_eps=g[2], mode="mix", eps=g[0], sz=ref["sz"]) for g in grid}}
    for name, steps, fof in [("main", GT_STEPS, pre_file), ("test", TEST_STEPS, test_file)]:
        for st in steps:
            ck = torch.load(os.path.join(CKPT, fof(st)), map_location=DEV)
            p = {k: ck["params"][k] for k in KEYS}
            res[(name, st)] = {"runs": ck["runs"], "r": {g: rollout_z(p, len(ck["runs"]), E=g[1], n_eps=g[2], mode="mix", eps=g[0], sz=ref["sz"]) for g in grid}}
            print(name, st, flush=True)
    save_result("mixsweep", res)


def exp_mixopen(args):
    fresh, ref, _ = metric_ref()
    res = {"fresh": rollout_z(fresh, 16, E=128, n_eps=2, mode="mix", eps=0.5, variant="open", sz=ref["sz_open"])}
    for name, steps, fof in [("main", GT_STEPS, pre_file), ("test", TEST_STEPS, test_file)]:
        for st in steps:
            ck = torch.load(os.path.join(CKPT, fof(st)), map_location=DEV)
            p = {k: ck["params"][k] for k in KEYS}
            res[(name, st)] = {"runs": ck["runs"], "r": rollout_z(p, len(ck["runs"]), E=128, n_eps=2, mode="mix", eps=0.5, variant="open", sz=ref["sz_open"])}
    save_result("mixopen", res)


def exp_rehearse(args):
    fresh, ref, _ = metric_ref()
    res = {"steps": [25_000, 100_000]}
    for name, steps, fof in [("main", GT_STEPS, pre_file), ("test", TEST_STEPS, test_file)]:
        conds = list(COND) if name == "main" else list(TEST_COND)
        for st in steps:
            f = fof(st)
            idx = {(c, ps): find(f, c, ps) for c in conds for ps in range(3)}
            runs = [ft_run(f, idx[(c, ps)], "null", v, cond=c, pseed=ps, pre_step=st) for v in ["full", "nocritic"] for c in conds for ps in range(3)]
            p, log = train(runs, 100_000, seed=args.seed + 80000, save=None)
            q = {k: v.detach() for k, v in p.items() if k in KEYS}
            res[(name, st)] = {"runs": runs, "mix": rollout_z(q, len(runs), E=64, n_eps=1, mode="mix", eps=0.5, sz=ref["sz"]),
                               "pol": rollout_z(q, len(runs), E=64, n_eps=1, mode="policy", sz=ref["sz"]),
                               "mix_open": rollout_z(q, len(runs), E=64, n_eps=1, mode="mix", eps=0.5, variant="open", sz=ref["sz_open"]),
                               "entropy": log["entropy"][:, -3:].mean(1)}
            print(name, st, flush=True)
    save_result("rehearse", res)


def save_result(name, obj):
    os.makedirs(RES, exist_ok=True)
    torch.save(obj, os.path.join(RES, f"{name}.pt"))


def exp_bench(args):
    for R, E, T in [(16, 256, 32), (48, 256, 32), (96, 128, 32)]:
        runs = [{"task": [PREDICATES[r % P]]} for r in range(R)]
        _, log = train(runs, E * T * 30, E=E, T=T)
        print(R, E, T, "fps", R * E * T * 30 / log["wall"])


def exp_scratch(args):
    runs = [{"task": [p], "seed": s} for s in range(4) for p in PREDICATES]
    _, log = train(runs, 10_000_000, seed=args.seed)
    save_result("scratch", {"runs": runs, "log": log})


POOL = ["pick_key", "pick_ball", "pick_box", "face_door", "open_door", "enter_right", "close_door", "key_right",
        "ball_near_box", "key_near_ball", "ball_near_door"]
DOWN = ["goal", "goal_with_key", "pick_ball2", "ball_right", "goal_with_ball", "ball2_left", "key_near_goal",
        "open_with_ball", "close_from_right", "box_right", "ball2_near_goal"]
COND = {t: [t] for t in POOL} | {"explore": ["explore"], "impact": ["impact"], "mix": POOL}
PRE_SAVE = [50_000, 100_000, 250_000, 500_000, 1_000_000]


def run_chunks(runs, steps, name, chunk=96, seed=0, **kw):
    logs = []
    for i in range(0, len(runs), chunk):
        print(f"[{name}] chunk {i // chunk + 1}/{math.ceil(len(runs) / chunk)}", flush=True)
        logs.append(train(runs[i:i + chunk], steps, seed=seed + i, **kw)[1])
        log = {k: torch.cat([l[k] for l in logs]) for k in logs[0] if torch.is_tensor(logs[0][k])}
        log["steps_per_iter"] = logs[0]["steps_per_iter"]
        log["wall"] = sum(l["wall"] for l in logs)
        save_result(name, {"runs": runs[:i + chunk], "log": log})


def exp_pretrain(args):
    runs = [{"task": COND[c], "cond": c, "pseed": s} for s in range(3) for c in COND]
    _, log = train(runs, 2_000_000, seed=args.seed, save="pre", save_at=PRE_SAVE)
    save_result("pretrain", {"runs": runs, "log": log})


def exp_finetune(args):
    pre = torch.load(os.path.join(CKPT, "pre_final.pt"))["runs"]
    runs = [{"task": [t], "cond": rp["cond"], "pseed": rp["pseed"], "init": {"file": "pre_final.pt", "idx": i}}
            for i, rp in enumerate(pre) for t in DOWN]
    run_chunks(runs, 6_000_000, "finetune", seed=args.seed)


def exp_probe(args):
    files = [f"pre_{s}.pt" for s in PRE_SAVE] + ["pre_final.pt"]
    out = {}
    for f in files:
        ck = torch.load(os.path.join(CKPT, f))
        p, R = ck["params"], len(ck["runs"])
        out[f] = {"runs": ck["runs"], "step": ck["step"], "stoch": probe(p, R, E=1024, n_eps=16),
                  "greedy": probe(p, R, E=256, n_eps=4, greedy=True), "rep": rep_stats(p, R)}
        print(f, flush=True)
    torch.manual_seed(0)
    p = fresh_params(16)
    out["fresh"] = {"runs": [{"cond": "none", "pseed": s} for s in range(16)], "step": 0,
                    "stoch": probe(p, 16, E=1024, n_eps=16), "greedy": probe(p, 16, E=256, n_eps=4, greedy=True),
                    "rep": rep_stats(p, 16)}
    save_result("probe", out)


def exp_drift(args):
    tasks = ["null", "goal", "open_with_ball", "pick_ball2", "key_near_goal", "close_from_right", "ball_right", "goal_with_key"]
    for norm in [True, False]:
        runs = [{"task": [t], "seed": s, "norm": norm} for s in range(3) for t in tasks]
        _, log = train(runs, 6_000_000, seed=args.seed, norm_adv=norm)
        save_result(f"drift_norm{int(norm)}", {"runs": runs, "log": log})


def exp_nocritic(args):
    tasks = ["goal", "open_with_ball", "pick_ball2", "key_near_goal", "close_from_right", "ball_right", "goal_with_key", "box_right"]
    runs = [{"task": [t], "seed": s} for s in range(3) for t in tasks]
    _, log = train(runs, 6_000_000, seed=args.seed, use_critic=False)
    save_result("nocritic_scratch", {"runs": runs, "log": log})


def exp_floor(args):
    tasks = ["goal", "open_with_ball", "pick_ball2", "key_near_goal", "close_from_right", "ball_right", "goal_with_key", "box_right"]
    for fl in [0.01, 0.03]:
        runs = [{"task": [t], "seed": s, "floor": fl} for s in range(3) for t in tasks]
        _, log = train(runs, 6_000_000, seed=args.seed, adv_floor=fl)
        save_result(f"floor_{fl}", {"runs": runs, "log": log})


def ft_run(file, idx, task, variant, **meta):
    src = {"file": file, "idx": idx}
    init = {"full": src, "scratch": None, "prior": None}.get(variant, src | {"mode": variant.replace("prior_", "")})
    prior = src if variant.startswith("prior") else None
    return {"task": [task], "variant": variant, "init": init, "prior": prior, **meta}


def find(file, cond, pseed):
    runs = torch.load(os.path.join(CKPT, file))["runs"]
    return next(i for i, r in enumerate(runs) if r["cond"] == cond and r["pseed"] == pseed)


MECH_SRC = ["enter_right", "open_door", "key_right", "pick_key", "face_door", "ball_near_box", "impact", "mix"]
MECH_TGT = ["goal", "key_near_goal", "close_from_right", "open_with_ball"]
MECH_VAR = ["trunk", "perm", "prior", "nocritic", "soft"]
CK_SRC = ["pick_key", "open_door", "enter_right", "face_door", "mix", "impact"]


def exp_mech(args):
    runs = [ft_run("pre_final.pt", find("pre_final.pt", c, ps), t, v, cond=c, pseed=ps)
            for c in MECH_SRC for ps in range(3) for t in MECH_TGT for v in MECH_VAR]
    run_chunks(runs, 6_000_000, "mech", seed=args.seed)


def exp_ckpt(args):
    runs = [ft_run(f"pre_{st}.pt", find(f"pre_{st}.pt", c, ps), t, "full", cond=c, pseed=ps, pre_step=st)
            for st in PRE_SAVE for c in CK_SRC for ps in range(3) for t in MECH_TGT]
    run_chunks(runs, 6_000_000, "ckpt", seed=args.seed)


def exp_demo(args):
    f = "pre_final.pt"
    runs = [ft_run(f, 0, "goal", "scratch", cond="none", pseed=0)]
    runs += [ft_run(f, find(f, c, 0), "goal", "full", cond=c, pseed=0) for c in ["open_door", "face_door", "pick_key", "enter_right"]]
    runs += [ft_run(f, 0, "open_with_ball", "scratch", cond="none", pseed=0)]
    runs += [ft_run(f, find(f, "mix", ps), "open_with_ball", "full", cond="mix", pseed=ps) for ps in [1, 0]]
    _, log = train(runs, 4_000_000, seed=args.seed, save="demo", save_at=[100_000, 250_000, 500_000, 1_000_000, 2_000_000])
    save_result("demo", {"runs": runs, "log": log})


def exp_proxy(args):
    pre = torch.load(os.path.join(CKPT, "pre_final.pt"))["runs"]
    runs = [{"task": [t], "cond": rp["cond"], "pseed": rp["pseed"], "init": {"file": "pre_final.pt", "idx": i}}
            for i, rp in enumerate(pre) for t in POOL]
    runs += [{"task": [t], "cond": "none", "pseed": s} for s in range(3) for t in POOL]
    run_chunks(runs, 500_000, "proxy", chunk=128, seed=args.seed)


REP_SRC = ["enter_right", "open_door", "close_door", "key_right", "pick_key", "face_door", "ball_near_box", "mix", "impact", "explore"]


def exp_reps(args):
    f = "pre_final.pt"
    runs = [ft_run(f, find(f, c, ps), t, "full", cond=c, pseed=ps, fseed=fs)
            for c in REP_SRC for ps in range(3) for fs in range(4) for t in MECH_TGT]
    runs += [ft_run(f, 0, t, "scratch", cond="none", pseed=s, fseed=0) for s in range(12) for t in MECH_TGT]
    run_chunks(runs, 6_000_000, "reps", seed=args.seed)


HARD_FREE = ["line3", "box_corner", "pair_corner"]


def exp_doorfree(args):
    runs = [{"task": [c], "cond": c, "pseed": s} for s in range(3) for c in HARD_FREE]
    _, log = train(runs, 2_000_000, seed=args.seed + 7, save="pre2")
    save_result("pretrain2", {"runs": runs, "log": log})
    ck = torch.load(os.path.join(CKPT, "pre2_final.pt"))
    p, R = ck["params"], len(ck["runs"])
    save_result("probe2", {"runs": ck["runs"], "stoch": probe(p, R, E=1024, n_eps=16), "rep": rep_stats(p, R)})
    f = "pre2_final.pt"
    runs = [ft_run(f, find(f, c, ps), t, "full", cond=c, pseed=ps, fseed=fs)
            for c in HARD_FREE for ps in range(3) for fs in range(4) for t in MECH_TGT]
    run_chunks(runs, 6_000_000, "doorfree", chunk=144, seed=args.seed + 2000)


def exp_residual(args):
    runs = [{"task": [t], "world": w, "seed": s} for w in ["key", "open"] for s in range(6) for t in MECH_TGT]
    _, log = train(runs, 6_000_000, seed=args.seed + 3000)
    save_result("residual", {"runs": runs, "log": log})


ONCE_SRC = ["pick_key", "open_door", "enter_right", "face_door", "ball_near_box", "key_right"]


def exp_once(args):
    runs = [{"task": [c], "cond": c, "pseed": s, "once": True} for s in range(3) for c in ONCE_SRC]
    _, log = train(runs, 2_000_000, seed=args.seed + 11, save="pre3")
    save_result("pretrain3", {"runs": runs, "log": log})
    ck = torch.load(os.path.join(CKPT, "pre3_final.pt"))
    p, R = ck["params"], len(ck["runs"])
    save_result("probe3", {"runs": ck["runs"], "stoch": probe(p, R, E=1024, n_eps=16), "rep": rep_stats(p, R)})
    f = "pre3_final.pt"
    runs = [ft_run(f, find(f, c, ps), t, "full", cond=c, pseed=ps, fseed=fs)
            for c in ONCE_SRC for ps in range(3) for fs in range(4) for t in MECH_TGT]
    run_chunks(runs, 6_000_000, "once", chunk=144, seed=args.seed + 4000)


def exp_critic(args):
    f = "pre_final.pt"
    tg = ["goal", "key_near_goal", "close_from_right"]
    runs = []
    for c in ["pick_key", "face_door", "mix"]:
        for ps in range(3):
            for fs in range(2):
                for t in tg:
                    runs.append(ft_run(f, find(f, c, ps), t, "full", cond=c, pseed=ps, fseed=fs, variant2="warmup", warmup=200_000))
                    runs.append(ft_run(f, find(f, c, ps), t, "full", cond=c, pseed=ps, fseed=fs, variant2="detach", detach=True))
    runs += [ft_run(f, 0, t, "scratch", cond="none", pseed=s, variant2="detach", detach=True) for s in range(6) for t in tg]
    runs += [ft_run(f, 0, t, "scratch", cond="none", pseed=s, variant2="warmup", warmup=200_000) for s in range(6) for t in tg]
    run_chunks(runs, 6_000_000, "critic", chunk=144, seed=args.seed + 5000)


NC_TGT = ["goal", "goal_with_key", "key_near_goal", "close_from_right", "open_with_ball", "pick_ball2"]


def exp_ncsweep(args):
    runs = []
    for f, conds in [("pre_final.pt", list(COND)), ("pre2_final.pt", HARD_FREE)]:
        for c in conds:
            for ps in range(3):
                for fs in range(2):
                    for t in NC_TGT:
                        for v in ["nocritic", "full"] if f == "pre_final.pt" and fs == 1 else ["nocritic"]:
                            runs.append(ft_run(f, find(f, c, ps), t, v, cond=c, pseed=ps, fseed=fs))
    runs += [ft_run("pre_final.pt", 0, t, "scratch", cond="none", pseed=s, fseed=0) for s in range(6) for t in NC_TGT]
    run_chunks(runs, 6_000_000, "ncsweep", chunk=144, seed=args.seed + 6000)


def exp_optimism(args):
    f = "pre_final.pt"
    tg = ["goal", "key_near_goal", "close_from_right"]
    runs = [ft_run(f, find(f, c, ps), t, "center", cond=c, pseed=ps, fseed=fs)
            for c in ["pick_key", "face_door", "mix", "enter_right", "open_door"] for ps in range(3) for fs in range(2) for t in tg]
    runs += [ft_run(f, 0, t, "scratch", cond="none", pseed=s, variant2=f"vinit{vi}", v_init=vi)
             for vi in [0.8, 0.3] for s in range(6) for t in tg]
    run_chunks(runs, 6_000_000, "optimism", chunk=144, seed=args.seed + 7000)


def exp_damage(args):
    f = "pre_final.pt"
    runs = [ft_run(f, find(f, "pick_key", ps), "goal", v, cond="pick_key", pseed=ps, **kw)
            for ps in range(3) for v, kw in [("full", {}), ("nocritic", {}), ("full", {"warmup": 200_000, "variant2": "warmup"})]]
    _, log = train(runs, 1_000_000, seed=args.seed + 8000, save="damage", save_at=[24_576, 98_304, 196_608, 393_216])
    save_result("damage", {"runs": runs, "log": log})
    out = {}
    for st in [24_576, 98_304, 196_608, 393_216, "final"]:
        ck = torch.load(os.path.join(CKPT, f"damage_{st}.pt"))
        out[st] = {"rep": rep_stats(ck["params"], len(runs)), "stoch": probe(ck["params"], len(runs), E=512, n_eps=4)}
    ck = torch.load(os.path.join(CKPT, f))
    idx = [find(f, "pick_key", ps) for ps in range(3)]
    p0 = {k: v[idx] for k, v in ck["params"].items()}
    out[0] = {"rep": rep_stats(p0, 3), "stoch": probe(p0, 3, E=512, n_eps=4)}
    save_result("damage_stats", out)


CKNC_SRC = ["enter_right", "open_door", "key_right", "close_door", "pick_key", "mix"]


def exp_ckptnc(args):
    runs = [ft_run(f"pre_{st}.pt", find(f"pre_{st}.pt", c, ps), t, "nocritic", cond=c, pseed=ps, pre_step=st)
            for st in PRE_SAVE for c in CKNC_SRC for ps in range(3) for t in MECH_TGT]
    run_chunks(runs, 6_000_000, "ckptnc", chunk=120, seed=args.seed + 9000)


def exp_demo2(args):
    f = "pre_final.pt"
    spec = [("none", "scratch"), ("open_door", "full"), ("face_door", "full"), ("face_door", "nocritic")]
    runs = [ft_run(f, find(f, c, ps) if c != "none" else 0, "goal", v, cond=c, pseed=ps, fseed=fs)
            for c, v in spec for ps in range(3) for fs in range(4)]
    _, log = train(runs, 4_000_000, seed=args.seed + 10000, save="demo2", save_at=[245_760, 491_520, 999_424, 1_998_848, 2_998_272])
    save_result("demo2", {"runs": runs, "log": log})


MG_ENV = {"unlock": "MiniGrid-Unlock-v0", "empty": "MiniGrid-Empty-8x8-v0", "doorkey": "MiniGrid-DoorKey-8x8-v0"}


def mg_obs(img):
    t = torch.as_tensor(img, dtype=torch.long)
    f = torch.cat([F.one_hot(t[..., 0], 11), F.one_hot(t[..., 1], 6), F.one_hot(t[..., 2], 3)], -1).float()
    return f.view(1, len(img), -1)


def mg_train(env_id, steps, params, seed, n_envs=16, T=128, lr=5e-4, epochs=4, n_mb=4, ent_coef=0.01, gamma=0.99,
             lam=0.95, clip=0.2, max_norm=0.5):
    import gymnasium as gym
    import minigrid  # noqa: F401
    import numpy as np
    torch.manual_seed(seed)
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n_envs)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs, _ = envs.reset(seed=seed)
    params = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    opt = torch.optim.Adam(params.values(), lr=lr, eps=1e-5)
    log = []
    for it in range(steps // (n_envs * T)):
        bo, ba, blp, bv, br, bd = [], [], [], [], [], []
        ep = sc = 0
        with torch.no_grad():
            for t in range(T):
                x = mg_obs(obs["image"])
                logits, v, _ = forward(params, x)
                dist = torch.distributions.Categorical(logits=logits[0])
                a = dist.sample()
                obs2, r, term, trunc, _ = envs.step(a.numpy())
                d = term | trunc
                bo.append(x[0]); ba.append(a); blp.append(dist.log_prob(a)); bv.append(v[0])
                br.append(torch.as_tensor(r, dtype=torch.float32)); bd.append(torch.as_tensor(d, dtype=torch.float32))
                ep += int(d.sum()); sc += int((d & (r > 0)).sum())
                obs = obs2
            last_v = forward(params, mg_obs(obs["image"]))[1][0]
            adv = torch.zeros(T, n_envs)
            gae = torch.zeros(n_envs)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else bv[t + 1]
                delta = br[t] + gamma * nv * (1 - bd[t]) - bv[t]
                gae = delta + gamma * lam * (1 - bd[t]) * gae
                adv[t] = gae
            ret = adv + torch.stack(bv)
        X, A, LP = torch.stack(bo).flatten(0, 1), torch.stack(ba).flatten(), torch.stack(blp).flatten()
        ADV, RET = adv.flatten(), ret.flatten()
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        N = len(A)
        for _ in range(epochs):
            perm = torch.randperm(N)
            for k in range(n_mb):
                idx = perm[k * N // n_mb:(k + 1) * N // n_mb]
                logits, v, _ = forward(params, X[idx][None])
                dist = torch.distributions.Categorical(logits=logits[0])
                ratio = (dist.log_prob(A[idx]) - LP[idx]).exp()
                pg = -torch.min(ratio * ADV[idx], ratio.clamp(1 - clip, 1 + clip) * ADV[idx]).mean()
                loss = pg + 0.5 * (v[0] - RET[idx]).pow(2).mean() - ent_coef * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params.values(), max_norm)
                opt.step()
        log.append((ep, sc))
    return {k: v.detach() for k, v in params.items()}, torch.tensor(log)


def exp_mg(args):
    torch.set_num_threads(1)
    kind, rest = args.job.split(":")
    os.makedirs(os.path.join(RES, "mg"), exist_ok=True)
    if kind == "pre":
        torch.manual_seed(args.seed)
        p0 = {k: v.cpu() for k, v in fresh_params_dim(1, 980).items()}
        params, log = mg_train(MG_ENV[rest], 500_000, p0, args.seed)
        torch.save(params, os.path.join(CKPT, f"mg_{rest}_{args.seed}.pt"))
    else:
        src, mode = rest.split("-") if "-" in rest else (rest, "scratch")
        torch.manual_seed(args.seed + 100)
        params = {k: v.cpu() for k, v in fresh_params_dim(1, 980).items()}
        if src != "scratch":
            pre = torch.load(os.path.join(CKPT, f"mg_{src}_{args.seed}.pt"))
            for k in (KEYS if mode == "full" else KEYS[:6]):
                params[k] = pre[k].clone()
        params, log = mg_train(MG_ENV["doorkey"], args.steps or 3_000_000, params, args.seed + 100)
        if args.steps:
            torch.save(params, os.path.join(CKPT, f"mg_demo_{rest}_{args.seed}_{args.steps}.pt"))
            return
    torch.save({"job": args.job, "seed": args.seed, "log": log}, os.path.join(RES, "mg", f"{args.job.replace(':', '_')}_{args.seed}.pt"))


def fresh_params_dim(R, dim):
    p = fresh_params(R)
    w = torch.empty(R, dim, HID)
    for r in range(R):
        torch.nn.init.orthogonal_(w[r], math.sqrt(2))
    p["W1"] = w.to(DEV)
    return p


def exp_rep_only(args):
    f = "pre_final.pt"
    tg = ["goal", "key_near_goal", "close_from_right"]
    runs = [ft_run(f, find(f, c, ps), t, v, cond=c, pseed=ps, fseed=fs)
            for c in ["enter_right", "open_door", "key_right", "close_door"] for ps in range(3) for fs in range(4)
            for t in tg for v in ["trunk", "perm"]]
    run_chunks(runs, 6_000_000, "reponly", chunk=144, seed=args.seed + 12000)


SEP_SRC = ["pick_key", "face_door", "mix", "open_door", "enter_right"]


def exp_sep(args):
    runs = [{"task": COND[c], "cond": c, "pseed": s} for s in range(3) for c in SEP_SRC]
    _, log = train(runs, 2_000_000, seed=args.seed + 13, save="pre4", sep=True)
    save_result("pretrain4", {"runs": runs, "log": log})
    f = "pre4_final.pt"
    tg = ["goal", "key_near_goal", "close_from_right"]
    runs = [ft_run(f, find(f, c, ps), t, v, cond=c, pseed=ps, fseed=fs)
            for c in SEP_SRC for ps in range(3) for fs in range(2) for t in tg for v in ["full", "vhead", "nocritic"]]
    runs += [ft_run(f, 0, t, "scratch", cond="none", pseed=s, fseed=0) for s in range(6) for t in tg]
    run_chunks(runs, 6_000_000, "sep", chunk=144, seed=args.seed + 14000, sep=True)


GT_TASKS = {"D": ["goal", "key_near_goal", "close_from_right", "open_with_ball", "goal_with_key", "pick_ball2", "ball_right",
                  "goal_with_ball", "ball2_left", "box_right", "ball2_near_goal"],
            "L": ["line3", "box_near_key", "pair_corner", "box_corner", "key_corner", "box_near_door"],
            "O": ["o:goal", "o:box_right", "o:pick_ball2", "o:ball2_left", "o:goal_with_ball", "o:ball2_near_goal"]}
GT_DEV = ["goal", "key_near_goal", "close_from_right", "open_with_ball", "line3", "box_near_key", "o:goal", "o:box_right"]
GT_STEPS = PRE_SAVE + [2_000_000]


def gt_run(file, idx, task, variant, **meta):
    world = "open" if task.startswith("o:") else "base"
    return ft_run(file, idx, task.removeprefix("o:"), variant, world=world, gt_task=task, **meta)


def pre_file(st):
    return "pre_final.pt" if st == 2_000_000 else f"pre_{st}.pt"


def exp_gt(args):
    tasks = [t for fam in GT_TASKS.values() for t in fam]
    runs = [gt_run("pre_final.pt", 0, t, "scratch", cond="none", pseed=s, pre_step=0) for s in range(8) for t in tasks]
    for ps in range(3):
        for st in GT_STEPS:
            f = pre_file(st)
            idx = {c: find(f, c, ps) for c in COND}
            runs += [gt_run(f, idx[c], t, "nocritic", cond=c, pseed=ps, pre_step=st) for c in COND for t in tasks]
    run_chunks(runs, 2_000_000, "gt", chunk=144, seed=args.seed + 20000)


LPROBE = ["pick_key", "pick_ball", "pick_box", "open_door", "enter_right", "close_door"]


def exp_lprobe(args):
    runs = [gt_run("pre_final.pt", 0, t, "scratch", cond="none", pseed=s, pre_step=0) for s in range(8) for t in LPROBE]
    for ps in range(3):
        for st in GT_STEPS:
            f = pre_file(st)
            idx = {c: find(f, c, ps) for c in COND}
            runs += [gt_run(f, idx[c], t, "nocritic", cond=c, pseed=ps, pre_step=st) for c in COND for t in LPROBE]
    run_chunks(runs, 250_000, "lprobe", chunk=144, seed=args.seed + 30000)


TEST_COND = {"k:open_door": ("open_door", "key"), "o:enter_right": ("enter_right", "open"), "o:close_door": ("close_door", "open"),
             "drop_key": ("drop_key", "base"), "face_box": ("face_box", "base"), "key_near_door": ("key_near_door", "base"),
             "box_near_ball": ("box_near_ball", "base"), "o:explore": ("explore", "open")}
TEST_STEPS = [50_000, 250_000, 1_000_000]


def test_file(st):
    return "pret_final.pt" if st == 1_000_000 else f"pret_{st}.pt"


def exp_pretrain_test(args):
    runs = [{"task": [t], "cond": c, "pseed": s, "world": w} for s in range(3) for c, (t, w) in TEST_COND.items()]
    _, log = train(runs, 1_000_000, seed=args.seed + 40000, save="pret", save_at=TEST_STEPS[:-1])
    save_result("pretrain_test", {"runs": runs, "log": log})


def exp_gt_test(args):
    tasks = [t for fam in GT_TASKS.values() for t in fam]
    runs = []
    for ps in range(3):
        for st in TEST_STEPS:
            f = test_file(st)
            idx = {c: find(f, c, ps) for c in TEST_COND}
            runs += [gt_run(f, idx[c], t, "nocritic", cond=c, pseed=ps, pre_step=st) for c in TEST_COND for t in tasks]
    run_chunks(runs, 2_000_000, "gt_test", chunk=144, seed=args.seed + 50000)


def exp_lprobe_test(args):
    runs = []
    for ps in range(3):
        for st in TEST_STEPS:
            f = test_file(st)
            idx = {c: find(f, c, ps) for c in TEST_COND}
            runs += [gt_run(f, idx[c], t, "nocritic", cond=c, pseed=ps, pre_step=st) for c in TEST_COND for t in LPROBE]
    run_chunks(runs, 250_000, "lprobe_test", chunk=144, seed=args.seed + 60000)


def exp_early(args):
    tg = ["goal", "key_near_goal", "close_from_right"]
    runs = [gt_run("pre_final.pt", 0, t, "scratch", cond="none", pseed=s, fseed=s, pre_step=0) for s in range(24) for t in tg]
    f = "pre_50000.pt"
    for ps in range(3):
        idx = {c: find(f, c, ps) for c in COND}
        runs += [gt_run(f, idx[c], t, v, cond=c, pseed=ps, fseed=fs, pre_step=50_000)
                 for c in COND for fs in range(6) for t in tg for v in ["nocritic", "full"]]
    run_chunks(runs, 2_000_000, "early", chunk=144, seed=args.seed + 70000)


COV_COND = ["smm", "cover"]
COV_STEPS = [250_000, 1_000_000, 2_000_000]


def cov_file(st):
    return "prec_final.pt" if st == 2_000_000 else f"prec_{st}.pt"


def exp_pretrain_cov(args):
    runs = [{"task": ["null"], "cond": c, "pseed": s, "intrinsic": c} for s in range(3) for c in COV_COND]
    _, log = train(runs, 2_000_000, seed=args.seed + 90000, save="prec", save_at=COV_STEPS[:-1])
    fresh, ref, _ = metric_ref()
    reach = {}
    for st in COV_STEPS:
        ck = torch.load(os.path.join(CKPT, cov_file(st)), map_location=DEV)
        p = {k: ck["params"][k] for k in KEYS}
        reach[st] = {"runs": ck["runs"], "mix": rollout_z(p, len(runs), E=128, n_eps=2, mode="mix", eps=0.5, sz=ref["sz"]),
                     "pol": rollout_z(p, len(runs), E=128, n_eps=2, mode="policy", sz=ref["sz"]),
                     "mix_open": rollout_z(p, len(runs), E=128, n_eps=2, mode="mix", eps=0.5, variant="open", sz=ref["sz_open"])}
    save_result("pretrain_cov", {"runs": runs, "log": log, "reach": reach})


def exp_gt_cov(args):
    tasks = [t for fam in GT_TASKS.values() for t in fam]
    door = ["goal", "key_near_goal", "close_from_right"]
    runs = []
    for ps in range(3):
        for st in COV_STEPS:
            f = cov_file(st)
            idx = {c: find(f, c, ps) for c in COV_COND}
            runs += [gt_run(f, idx[c], t, "nocritic", cond=c, pseed=ps, pre_step=st) for c in COV_COND for t in tasks]
            runs += [gt_run(f, idx[c], t, "full", cond=c, pseed=ps, fseed=fs, pre_step=st) for c in COV_COND for t in door for fs in range(2)]
            f = pre_file(st)
            idx = {c: find(f, c, ps) for c in ["explore", "impact", "enter_right", "open_door"]}
            runs += [gt_run(f, i, t, "full", cond=c, pseed=ps, fseed=fs, pre_step=st) for c, i in idx.items() for t in door for fs in range(2)]
    run_chunks(runs, 2_000_000, "gt_cov", chunk=144, seed=args.seed + 95000)


def exp_cover_pos(args):
    runs = [{"task": ["null"], "cond": "cover_pos", "pseed": s, "intrinsic": "cover_pos"} for s in range(3)]
    train(runs, 2_000_000, seed=args.seed + 96000, save="prec2", save_at=COV_STEPS[:-1])
    fof = lambda st: "prec2_final.pt" if st == 2_000_000 else f"prec2_{st}.pt"
    _, ref, _ = metric_ref()
    reach = {}
    for st in COV_STEPS:
        ck = torch.load(os.path.join(CKPT, fof(st)), map_location=DEV)
        p = {k: ck["params"][k] for k in KEYS}
        reach[st] = {"runs": ck["runs"], "mix": rollout_z(p, 3, E=128, n_eps=2, mode="mix", eps=0.5, sz=ref["sz"]),
                     "pol": rollout_z(p, 3, E=128, n_eps=2, mode="policy", sz=ref["sz"])}
    save_result("pretrain_cov2", {"runs": runs, "reach": reach})
    tasks = [t for fam in GT_TASKS.values() for t in fam]
    door = ["goal", "key_near_goal", "close_from_right"]
    gruns = []
    for ps in range(3):
        for st in COV_STEPS:
            i = find(fof(st), "cover_pos", ps)
            gruns += [gt_run(fof(st), i, t, "nocritic", cond="cover_pos", pseed=ps, pre_step=st) for t in tasks]
            gruns += [gt_run(fof(st), i, t, "full", cond="cover_pos", pseed=ps, fseed=fs, pre_step=st) for t in door for fs in range(2)]
    run_chunks(gruns, 2_000_000, "gt_cov2", chunk=144, seed=args.seed + 97000)


def exp_vreset(args):
    tg = ["open_with_ball", "pick_ball2", "goal", "key_near_goal"]
    runs = [{"task": [t], "seed": s, "vreset": n} for n in [100_000, 500_000] for s in range(6) for t in tg]
    _, log = train(runs, 6_000_000, seed=args.seed + 11000)
    save_result("vreset", {"runs": runs, "log": log})


if __name__ == "__main__":
    os.makedirs(CKPT, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job", default="")
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()
    globals()["exp_" + args.exp](args)
