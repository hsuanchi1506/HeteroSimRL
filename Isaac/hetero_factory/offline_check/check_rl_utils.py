"""Offline checks of rl_utils (no Isaac Lab, no rl_games) and a syntax check of the training scripts."""
import json
import py_compile
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO_ISAAC = HERE.parents[1]
sys.path[:0] = [str(HERE / "stubs"), str(REPO_ISAAC)]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from hetero_factory import rl_utils  # noqa: E402

fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ---- the official agent config
cfg_path = HERE / "stubs" / "isaaclab_tasks" / "direct" / "factory" / "agents" / "rl_games_ppo_cfg.yaml"
official = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
check(official["params"]["config"]["minibatch_size"] == 512 and official["params"]["config"]["horizon_length"] == 128,
      "official config: minibatch 512, horizon 128")


def changes_for(n):
    import copy

    a = copy.deepcopy(official)
    ch = rl_utils.fix_minibatch(a, n)
    return a, ch


a, ch = changes_for(3)
check(a["params"]["config"]["minibatch_size"] == 384 and a["params"]["config"]["central_value_config"]["minibatch_size"] == 384
      and len(ch) == 2, f"3 envs: minibatch 512 -> 384 for PPO and the central value ({len(ch)} changes recorded)")
for n in (128, 4096):
    a, ch = changes_for(n)
    check(ch == [] and a["params"]["config"]["minibatch_size"] == 512, f"{n} envs: official minibatch kept")
a, ch = changes_for(5)  # 640 samples: 512 does not divide, 128 does
check(a["params"]["config"]["minibatch_size"] == 128, f"5 envs: minibatch -> {a['params']['config']['minibatch_size']} (multiple of seq 128)")
check(rl_utils.epochs_for_env_steps(3_276_800, 128, 128) == 200, "128 envs x 128 steps x 200 = official 3.28 M steps")
check(rl_utils.epochs_for_env_steps(3_276_800, 4096, 128) == 7, "same budget at 4096 envs: 7 iterations")


# ---- StepTimer
class DummyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,))
    action_space = gym.spaces.Box(-1, 1, (1,))

    def step(self, a):
        return 0, 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        return 0, {}


t = rl_utils.StepTimer(DummyEnv(), sync_cuda=True)
for _ in range(5):
    t.step(0)
tt, n = t.pop()
check(n == 5 and tt >= 0 and t.pop() == (0.0, 0), "StepTimer counts calls and clears on pop")

# ---- MetricsObserver
with tempfile.TemporaryDirectory() as d:
    env = SimpleNamespace(task_ids=torch.tensor([0, 1, 2, 0, 1, 2, 0]))
    obs = rl_utils.MetricsObserver(d, ("peg_insert", "gear_mesh", "nut_thread"), step_timer=t, gpu=None,
                                   warmup_epochs=1, env=env)
    probe_calls = []
    fake_probe = SimpleNamespace(on_update_start=lambda algo: probe_calls.append(algo.epoch_num),
                                 last_summary={"grad_pair_cos_min": -0.2}, time_total=0.5)
    obs.probe = fake_probe
    obs.after_init(SimpleNamespace(epoch_num=1))
    obs.process_infos({"successes_nut_thread": torch.tensor(0.5), "return_peg_insert": torch.tensor(12.0),
                       "time_outs": torch.ones(7), "reset_ik_attempts": 2}, None)
    for ep in (1, 2):
        obs.algo.epoch_num = ep
        obs.after_steps()
        t.step(0)
        # the rl_games fork calls after_print_stats twice per epoch
        obs.after_print_stats(frame=ep * 7 * 128, epoch_num=ep, total_time=float(ep))
        obs.after_print_stats(frame=ep * 7 * 128, epoch_num=ep, total_time=float(ep))
    lines = [json.loads(x) for x in (Path(d) / "metrics.jsonl").read_text().splitlines()]
    check(len(lines) == 2, "one JSON line per epoch although after_print_stats runs twice")
    check(probe_calls == [1, 2], "after_steps runs the gradient probe every epoch (it decides itself when to probe)")
    check(lines[0]["grad_pair_cos_min"] == -0.2 and abs(lines[0]["wall_time_excl_diag"] - (lines[0]["wall_time"] - 0.5)) < 1e-9,
          "probe summary and wall time without the probe in the epoch line")
    r = lines[1]
    check(r["env_steps"] == 1792 and r["epoch_env_steps"] == 896, "env steps total and per epoch")
    check(r["successes_nut_thread"] == 0.5 and r["return_peg_insert"] == 12.0 and r["reset_ik_attempts"] == 2.0,
          "per-task values carried into the epoch line")
    check("time_outs" not in r, "per-env vectors are not logged")
    check(r["envs_per_task_at_end"] == {"peg_insert": 3, "gear_mesh": 2, "nut_thread": 2}, "envs per task recorded")
    check(lines[0]["wall_time_after_warmup"] == 0 or lines[0]["wall_time_after_warmup"] is not None,
          "warm-up mark set at the end of epoch 1")
    check(r["env_step_calls"] == 1, "env-step time per epoch")

# ---- GpuSampler without NVML must not crash
g = rl_utils.GpuSampler(0)
check(isinstance(g.pop(), dict), f"GpuSampler without NVML degrades gracefully (nvml ok: {g.ok})")
g.close()

# ---- the training scripts at least compile
for p in sorted((REPO_ISAAC / "scripts").glob("*.py")):
    try:
        py_compile.compile(str(p), doraise=True)
        check(True, f"{p.name} compiles")
    except py_compile.PyCompileError as e:
        check(False, f"{p.name}: {e}")

print(f"\n{len(fails)} failures")
sys.exit(1 if fails else 0)
