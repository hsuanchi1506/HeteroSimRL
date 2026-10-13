"""Widen the released Play2Perfect play checkpoint for inputs that end with a task one-hot.

The official stage 2 resumes from the released stage-1 ("play") checkpoint (``train.py`` default
``--checkpoint_load_mode resume``). Its networks take the official 140-value policy input and 162-value critic
input, each followed by SAPG's 32-value learned block embedding (``a2c_network.extra_params``); the observation
normalisers cover the 140 / 162 env values. Our env appends a task one-hot of ``n_new`` values to both inputs,
before SAPG's block column. This writes a copy of the checkpoint in which:

* the input weights get ``n_new`` zero columns at the one-hot's position (policy LSTM ``weight_ih_l0``, critic
  first layer), so the widened networks compute exactly what the released ones compute;
* the normalisers get mean 0 / variance 1 for the new inputs (their sample count is billions of steps, so these
  values barely move during finetuning; a one-hot then enters as 0 or 1);
* the Adam moments of the widened weights get zero columns, so ``resume`` restores the optimiser as officially;
* the saved rollout state that ``resume`` restores (first observation) gets zero columns too.

Everything else (epoch counter, learning rate, SAPG block embeddings and sigmas, trackers) is left as released.
"""

from __future__ import annotations

import torch

SAPG_EMBED = 32  # expl_reward_coef_embd_size of the released config


def _insert_cols(w: torch.Tensor, at: int, n: int) -> torch.Tensor:
    pad = torch.zeros(*w.shape[:-1], n, dtype=w.dtype, device=w.device)
    return torch.cat([w[..., :at], pad, w[..., at:]], dim=-1)


def _widen_rms(d: dict, prefix: str, n: int):
    mean, var = d[prefix + "running_mean"], d[prefix + "running_var"]
    d[prefix + "running_mean"] = torch.cat([mean, torch.zeros(n, dtype=mean.dtype)])
    d[prefix + "running_var"] = torch.cat([var, torch.ones(n, dtype=var.dtype)])


def widen_checkpoint(src: str, dst: str, n_new: int, obs_dim: int = 140, state_dim: int = 162) -> dict:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    st = ck[0] if isinstance(ck, dict) and 0 in ck else ck
    report = {}

    m = st["model"]
    if tuple(m["running_mean_std.running_mean"].shape) != (obs_dim,):
        raise ValueError(f"policy normaliser has shape {tuple(m['running_mean_std.running_mean'].shape)}, "
                         f"expected ({obs_dim},)")
    _widen_rms(m, "running_mean_std.", n_new)
    key = "a2c_network.rnn.rnn.weight_ih_l0"
    w = m[key]
    if w.shape[1] != obs_dim + SAPG_EMBED:
        raise ValueError(f"{key} has {w.shape[1]} inputs, expected {obs_dim + SAPG_EMBED}")
    m[key] = _insert_cols(w, obs_dim, n_new)
    report[key] = [list(w.shape), list(m[key].shape)]

    vf = st["assymetric_vf_nets"]
    _widen_rms(vf, "model.running_mean_std.", n_new)
    vkey = "model.a2c_network.actor_mlp.0.weight"
    v = vf[vkey]
    if v.shape[1] != state_dim + SAPG_EMBED:
        raise ValueError(f"{vkey} has {v.shape[1]} inputs, expected {state_dim + SAPG_EMBED}")
    vf[vkey] = _insert_cols(v, state_dim, n_new)
    report[vkey] = [list(v.shape), list(vf[vkey].shape)]

    widened = 0
    for s in st.get("optimizer", {}).get("state", {}).values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in s and tuple(s[k].shape) == tuple(w.shape):
                s[k] = _insert_cols(s[k], obs_dim, n_new)
                widened += 1
    if widened != 2:
        raise ValueError(f"expected to widen 2 Adam moments of {key}, widened {widened}")
    report["adam_moments_widened"] = widened

    obs = st.get("obs")
    if isinstance(obs, dict):
        if "obs" in obs:
            obs["obs"] = _insert_cols(obs["obs"], obs_dim, n_new)
        if "states" in obs:
            obs["states"] = _insert_cols(obs["states"], state_dim, n_new)
        report["saved_obs"] = {k: list(v.shape) for k, v in obs.items()}

    report.update(epoch=int(st.get("epoch", -1)), frame=int(st.get("frame", -1)),
                  lr=float(st["optimizer"]["param_groups"][0]["lr"]))
    torch.save(ck, dst)
    return report


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--n_new", type=int, default=4)
    a = p.parse_args()
    print(json.dumps(widen_checkpoint(a.src, a.dst, a.n_new), indent=1))
