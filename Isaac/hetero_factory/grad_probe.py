"""Per-task gradient diagnostics for rl_games PPO (RULES.md section 5).

Measured at the start of a PPO update (rl_games calls ``algo_observer.after_steps()`` right after the rollout
has been turned into a dataset and before any optimisation step), with the parameters the update starts from:

* per-task actor gradient norm: gradient of the main network's loss, averaged over that task's samples;
* task-pair cosine similarity between those gradients (only tasks present in the rollout);
* each task's cosine with the gradient of the whole rollout;
* update-to-update cosine: the whole-rollout gradient (and each task's) compared with the previous probe. The
  probe runs on epochs 1 and 2, then ``k * every`` and ``k * every + 1``, so probes come in pairs of
  consecutive updates (``adjacent_to_prev``);
* every epoch, cheaply: the parameter change made by the last update, and its cosine with the change before.

"Main network's loss" is exactly what rl_games' ``A2CAgent.calc_gradients`` minimises for the policy network:
PPO clipped surrogate (``actor_loss_func``) + bounds loss, minus the entropy bonus, plus the value loss only if
the agent has no separate central value (the Factory config has one, so there it is the actor loss alone).
Each sample's loss is computed as rl_games does, then averaged per task (task id read from the one-hot in the
raw observation) and differentiated per task. Computed in float32 without autocast.

The probe observes only: it does not touch ``.grad``, the optimiser, the dataset or the global RNG, and it
switches the input normaliser to eval mode while it runs so its statistics are not updated. It is run in
chunks of sequences so 4096 envs fit in memory; its time is recorded so it can be taken out of wall-clock
comparisons.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch


def _flat(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return None
    return float(torch.dot(a, b) / (na * nb))


class GradientProbe:
    def __init__(self, out_dir, task_names, onehot_slice: slice, every: int = 10, chunk_seqs: int = 64,
                 max_seqs: int | None = None, seed: int = 0):
        self.path = Path(out_dir) / "grad_diag.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.task_names = tuple(task_names)
        self.onehot_slice = onehot_slice
        self.every = int(every)
        self.chunk_seqs = int(chunk_seqs)
        self.max_seqs = max_seqs
        self._gen_seed = seed
        self._force = False
        self._force_left = 0
        self._prev = None  # {"epoch", "total", "tasks": {t: grad}}
        self._prev_params = None
        self._prev_delta = None
        self.last_summary = {}
        self.time_total = 0.0

    # ------------------------------------------------------------------ schedule
    def should_probe(self, epoch: int) -> bool:
        """Pairs of consecutive updates: epochs 1-2 (start of training), then k*every and k*every+1."""
        if self._force:
            return True
        return self.every > 0 and (epoch in (1, 2) or epoch % self.every in (0, 1))

    def force_next(self, n: int = 2):
        """Probe the next ``n`` updates whatever the schedule (e.g. around a homo task switch)."""
        self._force_left = n
        self._force = True

    # ------------------------------------------------------------------ entry point
    def on_update_start(self, algo):
        """Call from ``AlgoObserver.after_steps``."""
        epoch = int(algo.epoch_num)
        self.last_summary = {}
        t0 = time.perf_counter()
        self._param_delta(algo, epoch)
        if self.should_probe(epoch):
            self._probe(algo, epoch)
            if self._force:
                self._force_left -= 1
                self._force = self._force_left > 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.time_total += dt
        self.last_summary["grad_diag_time"] = dt

    # ------------------------------------------------------------------ cheap, every epoch
    def _param_delta(self, algo, epoch):
        params = torch.cat([p.detach().reshape(-1) for p in algo.model.parameters() if p.requires_grad]).float()
        rec = None
        if self._prev_params is not None:
            delta = params - self._prev_params
            rec = dict(epoch=epoch, kind="delta", delta_norm=float(delta.norm()))
            if self._prev_delta is not None:
                rec["cos_prev_delta"] = _cos(delta, self._prev_delta)
            self._prev_delta = delta
            self.last_summary.update({"param_delta_norm": rec["delta_norm"],
                                      "param_delta_cos_prev": rec.get("cos_prev_delta")})
            self._write(rec)
        self._prev_params = params.clone()

    # ------------------------------------------------------------------ per-task gradients
    def _sample_losses(self, algo, idx, rnn_seq_idx):
        """Per-sample main-network loss on samples ``idx`` (sequences ``rnn_seq_idx``), as calc_gradients."""
        d = algo.dataset.values_dict
        obs = d["obs"][idx]
        batch = {"is_train": True, "prev_actions": d["actions"][idx], "obs": algo._preproc_obs(obs)}
        if algo.is_rnn:
            batch["rnn_states"] = [s[:, rnn_seq_idx, :].contiguous() for s in d["rnn_states"]]
            batch["seq_length"] = algo.seq_length
            if getattr(algo, "zero_rnn_on_done", False):
                batch["dones"] = d["dones"][idx]
        res = algo.model(batch)
        loss = algo.actor_loss_func(d["old_logp_actions"][idx], res["prev_neglogp"], d["advantages"][idx],
                                    algo.ppo, algo.e_clip).reshape(-1)
        if getattr(algo, "has_value_loss", False):
            from rl_games.common import common_losses

            c = common_losses.critic_loss(algo.model, d["old_values"][idx], res["values"], algo.e_clip,
                                          d["returns"][idx], algo.clip_value)
            loss = loss + 0.5 * algo.critic_coef * c.reshape(loss.shape[0], -1).sum(-1)
        if algo.entropy_coef:
            loss = loss - algo.entropy_coef * res["entropy"].reshape(-1)
        bounds_coef = getattr(algo, "bounds_loss_coef", None)
        if bounds_coef:
            btype = getattr(algo, "bound_loss_type", "bound")
            if btype == "regularisation":
                b = algo.reg_loss(res["mus"])
            elif btype == "bound":
                b = algo.bound_loss(res["mus"])
            else:
                b = None
            if b is not None:
                loss = loss + bounds_coef * b.reshape(-1)
        mask = d.get("rnn_masks")
        mask = mask[idx].reshape(-1).float() if mask is not None else torch.ones_like(loss)
        return loss, mask, obs[:, self.onehot_slice].argmax(-1)

    def _probe(self, algo, epoch):
        model = algo.model
        d = algo.dataset.values_dict
        obs = d["obs"]
        assert isinstance(obs, torch.Tensor), "dict observations are not supported"
        n_samples = obs.shape[0]
        seq = algo.seq_length if algo.is_rnn else 1
        n_seq = n_samples // seq
        seq_ids = torch.arange(n_seq, device=obs.device)
        if self.max_seqs is not None and n_seq > self.max_seqs:
            gen = torch.Generator(device="cpu").manual_seed(self._gen_seed + epoch)  # private RNG
            seq_ids = seq_ids[torch.randperm(n_seq, generator=gen)[: self.max_seqs].to(obs.device)].sort().values

        params = [p for p in model.parameters() if p.requires_grad]
        T = len(self.task_names)
        acc = [None] * T
        cnt = [0.0] * T
        rms = getattr(model, "running_mean_std", None)
        rms_training = rms.training if rms is not None else None
        if rms is not None:
            rms.eval()
        try:
            with torch.autocast(device_type="cuda", enabled=False):
                for c0 in range(0, len(seq_ids), self.chunk_seqs):
                    chunk = seq_ids[c0:c0 + self.chunk_seqs]
                    idx = (chunk[:, None] * seq + torch.arange(seq, device=obs.device)).reshape(-1)
                    loss, mask, task = self._sample_losses(algo, idx, chunk)
                    present = [t for t in range(T) if bool(((task == t) & (mask > 0)).any())]
                    for j, t in enumerate(present):
                        w = (task == t).float() * mask
                        g = torch.autograd.grad((loss * w).sum(), params, retain_graph=j < len(present) - 1,
                                                allow_unused=True)
                        g = _flat(g, params).float()
                        acc[t] = g if acc[t] is None else acc[t] + g
                        cnt[t] += float(w.sum())
        finally:
            if rms is not None and rms_training:
                rms.train()

        present = [t for t in range(T) if acc[t] is not None and cnt[t] > 0]
        grads = {t: acc[t] / cnt[t] for t in present}
        total = sum(acc[t] for t in present) / sum(cnt[t] for t in present)
        rec = dict(epoch=epoch, kind="probe", n_seq_used=int(len(seq_ids)), n_seq_total=int(n_seq),
                   total_grad_norm=float(total.norm()), tasks={})
        for t in present:
            name = self.task_names[t]
            rec["tasks"][name] = dict(n_samples=int(cnt[t]), grad_norm=float(grads[t].norm()),
                                      cos_to_total=_cos(grads[t], total))
        rec["pair_cos"] = {f"{self.task_names[a]}|{self.task_names[b]}": _cos(grads[a], grads[b])
                           for i, a in enumerate(present) for b in present[i + 1:]}
        if self._prev is not None:
            rec["prev_epoch"] = self._prev["epoch"]
            rec["adjacent_to_prev"] = self._prev["epoch"] == epoch - 1
            rec["cos_total_prev"] = _cos(total, self._prev["total"])
            for t in present:
                if t in self._prev["tasks"]:
                    rec["tasks"][self.task_names[t]]["cos_prev"] = _cos(grads[t], self._prev["tasks"][t])
        self._prev = {"epoch": epoch, "total": total, "tasks": grads}
        self._write(rec)

        norms = [rec["tasks"][self.task_names[t]]["grad_norm"] for t in present]
        pair = list(rec["pair_cos"].values())
        self.last_summary.update(
            grad_total_norm=rec["total_grad_norm"],
            grad_task_norm_ratio=(max(norms) / min(norms)) if len(norms) > 1 and min(norms) > 0 else None,
            grad_pair_cos_min=min(pair) if pair else None,
            grad_pair_cos_mean=sum(pair) / len(pair) if pair else None,
            grad_cos_total_prev=rec.get("cos_total_prev") if rec.get("adjacent_to_prev") else None,
        )

    def _write(self, rec):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
