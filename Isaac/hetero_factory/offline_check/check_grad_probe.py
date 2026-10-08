"""Offline checks of grad_probe.GradientProbe against a small LSTM policy shaped like rl_games' (no rl_games).

1. Per-task gradients equal a brute-force computation (one forward over all samples, masked mean per task),
   for any chunk size; the whole-rollout gradient is the sample-weighted mean of the task gradients.
2. The probe has no side effects: global RNG, input-normaliser statistics and train/eval flag, parameter
   .grad, and the model's outputs are unchanged.
3. Schedule: probes on epochs k*every and k*every+1, consecutive probes marked adjacent; parameter-change
   cosine every epoch; homo-style rollouts (one task) give no pair cosines; JSON lines written.
"""
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1])]

import torch  # noqa: E402
from torch import nn  # noqa: E402

from hetero_factory.grad_probe import GradientProbe  # noqa: E402

fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


OBS, ACT, HID, L = 22, 6, 32, 8  # obs dim (one-hot at 13:16), action dim, LSTM size, sequence length
ONEHOT = slice(13, 16)


class RunningMeanStd(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("var", torch.ones(d))

    def forward(self, x):
        if self.training:  # updates its statistics in train mode, like rl_games'
            self.mean.mul_(0.9).add_(0.1 * x.mean(0))
            self.var.mul_(0.9).add_(0.1 * x.var(0))
        return (x - self.mean) / torch.sqrt(self.var + 1e-5)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.running_mean_std = RunningMeanStd(OBS)
        self.lstm = nn.LSTM(OBS, HID, num_layers=2)
        self.mu = nn.Linear(HID, ACT)
        self.value = nn.Linear(HID, 1)  # unused by the actor loss, like with a central value
        self.logstd = nn.Parameter(torch.zeros(ACT) - 0.5)

    def forward(self, bd):
        obs = self.running_mean_std(bd["obs"])
        n = obs.shape[0] // bd["seq_length"]
        x = obs.view(n, bd["seq_length"], -1).transpose(0, 1)
        out, _ = self.lstm(x, tuple(bd["rnn_states"]))
        out = out.transpose(0, 1).reshape(obs.shape[0], -1)
        mu = self.mu(out)
        sigma = self.logstd.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, sigma)
        return dict(prev_neglogp=-dist.log_prob(bd["prev_actions"]).sum(-1), entropy=dist.entropy().sum(-1),
                    mus=mu, sigmas=sigma, values=self.value(out))


def actor_loss(old_neglogp, neglogp, adv, is_ppo, clip):  # rl_games common_losses.actor_loss
    ratio = torch.exp(old_neglogp - neglogp)
    return torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - clip, 1.0 + clip))


def bound_loss(mu):  # rl_games A2CAgent.bound_loss
    return (torch.clamp_min(mu - 1.1, 0.0) ** 2 + torch.clamp_max(mu + 1.1, 0.0) ** 2).sum(-1)


def make_algo(n_envs, tasks_of_env, seed=0):
    g = torch.Generator().manual_seed(seed)
    model = Net()
    model.train()
    B = n_envs * L * 2  # two sequences per env (horizon = 2 * L)
    obs = torch.randn(B, OBS, generator=g)
    env_of_sample = torch.arange(B) // (2 * L)
    obs[:, ONEHOT] = torch.nn.functional.one_hot(torch.tensor(tasks_of_env)[env_of_sample], 3).float()
    with torch.no_grad():
        model.running_mean_std.eval()
        old = model({"obs": obs, "seq_length": L, "prev_actions": torch.zeros(B, ACT),
                     "rnn_states": [torch.zeros(2, B // L, HID), torch.zeros(2, B // L, HID)]})
        model.running_mean_std.train()
    actions = old["mus"] + 0.3 * torch.randn(B, ACT, generator=g)
    dist = torch.distributions.Normal(old["mus"], old["sigmas"])
    values = dict(
        obs=obs, actions=actions, old_logp_actions=-dist.log_prob(actions).sum(-1),
        advantages=torch.randn(B, generator=g), dones=torch.zeros(B), rnn_masks=None,
        rnn_states=[0.1 * torch.randn(2, B // L, HID, generator=g), 0.1 * torch.randn(2, B // L, HID, generator=g)],
        old_values=torch.zeros(B, 1), returns=torch.zeros(B, 1),
    )
    return SimpleNamespace(
        model=model, dataset=SimpleNamespace(values_dict=values), is_rnn=True, seq_length=L, zero_rnn_on_done=False,
        ppo=True, e_clip=0.2, entropy_coef=0.0, bounds_loss_coef=1e-4, bound_loss_type="bound",
        has_value_loss=False, actor_loss_func=actor_loss, bound_loss=bound_loss, _preproc_obs=lambda o: o,
        epoch_num=10,
    )


def brute_force(algo, task):
    """One forward over every sample, masked mean of the per-sample loss for one task (or all: task=None)."""
    d = algo.dataset.values_dict
    params = [p for p in algo.model.parameters() if p.requires_grad]
    rms = algo.model.running_mean_std
    rms.eval()
    res = algo.model({"obs": d["obs"], "seq_length": L, "prev_actions": d["actions"], "rnn_states": d["rnn_states"]})
    rms.train()
    loss = actor_loss(d["old_logp_actions"], res["prev_neglogp"], d["advantages"], True, 0.2) + 1e-4 * bound_loss(res["mus"])
    tid = d["obs"][:, ONEHOT].argmax(-1)
    w = torch.ones_like(loss) if task is None else (tid == task).float()
    g = torch.autograd.grad((loss * w).sum() / w.sum(), params, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(g, params)])


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


names = ("peg_insert", "gear_mesh", "nut_thread")
tasks_of_env = [0, 1, 2, 0, 1, 2, 0]

# ---- 1. correctness for several chunk sizes
for chunk in (1, 3, 1000):
    algo = make_algo(7, tasks_of_env)
    with tempfile.TemporaryDirectory() as d:
        probe = GradientProbe(d, names, ONEHOT, every=10, chunk_seqs=chunk)
        probe.on_update_start(algo)
        rec = [r for r in read(Path(d) / "grad_diag.jsonl") if r["kind"] == "probe"][0]
    for t, name in enumerate(names):
        ref = brute_force(algo, t)
        check(abs(rec["tasks"][name]["grad_norm"] - float(ref.norm())) < 1e-4 * max(1.0, float(ref.norm())),
              f"chunk {chunk}: {name} gradient norm {rec['tasks'][name]['grad_norm']:.6f} = brute force {float(ref.norm()):.6f}")
    g = [brute_force(algo, t) for t in range(3)]
    cos01 = float(torch.dot(g[0], g[1]) / (g[0].norm() * g[1].norm()))
    check(abs(rec["pair_cos"]["peg_insert|gear_mesh"] - cos01) < 1e-4, f"chunk {chunk}: pair cosine {cos01:.4f}")
    total = brute_force(algo, None)
    check(abs(rec["total_grad_norm"] - float(total.norm())) < 1e-4 * max(1.0, float(total.norm())),
          f"chunk {chunk}: whole-rollout gradient = sample-weighted mean of task gradients")
    counts = {n: rec["tasks"][n]["n_samples"] for n in names}
    check(counts == {"peg_insert": 3 * 2 * L, "gear_mesh": 2 * 2 * L, "nut_thread": 2 * 2 * L}, f"chunk {chunk}: samples per task {counts}")

# ---- 2. no side effects
algo = make_algo(7, tasks_of_env, seed=1)
for p in algo.model.parameters():
    p.grad = torch.full_like(p, 7.0)
rng_before = torch.get_rng_state()
rms_before = (algo.model.running_mean_std.mean.clone(), algo.model.running_mean_std.var.clone())
d = algo.dataset.values_dict
with torch.no_grad():
    algo.model.running_mean_std.eval()
    out_before = algo.model({"obs": d["obs"], "seq_length": L, "prev_actions": d["actions"], "rnn_states": d["rnn_states"]})["mus"]
    algo.model.running_mean_std.train()
with tempfile.TemporaryDirectory() as tmp:
    GradientProbe(tmp, names, ONEHOT, every=10, chunk_seqs=2, max_seqs=6).on_update_start(algo)
check(torch.equal(torch.get_rng_state(), rng_before), "global RNG untouched (subsampling uses a private generator)")
check(torch.equal(algo.model.running_mean_std.mean, rms_before[0]) and torch.equal(algo.model.running_mean_std.var, rms_before[1]),
      "input-normaliser statistics not updated")
check(algo.model.running_mean_std.training, "input normaliser back in train mode")
check(all(torch.all(p.grad == 7.0) for p in algo.model.parameters()), "parameter .grad untouched")
with torch.no_grad():
    algo.model.running_mean_std.eval()
    out_after = algo.model({"obs": d["obs"], "seq_length": L, "prev_actions": d["actions"], "rnn_states": d["rnn_states"]})["mus"]
    algo.model.running_mean_std.train()
check(torch.equal(out_before, out_after), "model outputs unchanged")

# ---- 3. schedule, update-to-update, parameter change, homo rollouts
algo = make_algo(7, tasks_of_env, seed=2)
opt = torch.optim.SGD(algo.model.parameters(), lr=0.01)
with tempfile.TemporaryDirectory() as tmp:
    probe = GradientProbe(tmp, names, ONEHOT, every=10, chunk_seqs=4)
    for epoch in range(1, 23):
        algo.epoch_num = epoch
        probe.on_update_start(algo)
        opt.zero_grad()  # a stand-in update so the parameters change between epochs
        brute = brute_force(algo, None)
        i = 0
        for p in algo.model.parameters():
            n = p.numel()
            p.grad = brute[i:i + n].view_as(p).clone()
            i += n
        opt.step()
    recs = read(Path(tmp) / "grad_diag.jsonl")
probes = [r for r in recs if r["kind"] == "probe"]
deltas = [r for r in recs if r["kind"] == "delta"]
check([r["epoch"] for r in probes] == [1, 2, 10, 11, 20, 21], f"probes on epochs {[r['epoch'] for r in probes]}")
check([r.get("adjacent_to_prev") for r in probes] == [None, True, False, True, False, True],
      "consecutive probes marked adjacent")
check(all(isinstance(r["tasks"]["nut_thread"].get("cos_prev"), float) for r in probes[1:]), "per-task update-to-update cosine")
check(len(deltas) == 21 and all("cos_prev_delta" in r for r in deltas[1:]), "parameter change + cosine every epoch")
cosd = [r["cos_prev_delta"] for r in deltas[1:]]
check(all(c > 0.5 for c in cosd), f"plain gradient steps on a fixed batch point the same way (min cos {min(cosd):.3f})")

homo = make_algo(4, [2, 2, 2, 2], seed=3)
with tempfile.TemporaryDirectory() as tmp:
    pr = GradientProbe(tmp, names, ONEHOT, every=10)
    pr.on_update_start(homo)
    rec = [r for r in read(Path(tmp) / "grad_diag.jsonl") if r["kind"] == "probe"][0]
check(list(rec["tasks"]) == ["nut_thread"] and rec["pair_cos"] == {}, "homo rollout: one task, no pair cosines")
check(pr.last_summary["grad_pair_cos_min"] is None and "grad_diag_time" in pr.last_summary, "summary for the metrics line")

print(f"\n{len(fails)} failures")
sys.exit(1 if fails else 0)
