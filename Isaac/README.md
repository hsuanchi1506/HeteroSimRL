# Isaac / PhysX

Homo and Hetero versions of the official Isaac Lab Factory tasks (PegInsert, GearMesh, NutThread), following `../RULES.md`. Chinese version: [README.zh-TW.md](README.zh-TW.md). Results: [results/README.md](results/README.md).

## Status (2026-10-08)

- **Done**:
  - The union-scene environment, shared by Homo and Hetero.
  - Training, evaluation, gradient diagnostics, video, timing and profiling tools.
  - Hetero/Homo training at 128 envs: official Hetero with 3 seeds, the other two groups with 5 seeds each.
  - RULES section 6 with trained policies at 3 / 128 / 4096 envs: contacts, contact phases and waiting.
  - Section 7: reading the PhysX source.
- **Running** on gputw1:
  - The RULES "run once" set (`rules_*`): 128 envs done, 3 and 4096 envs in progress.
  - The TODOs from the 2026-10-07 meeting: advantage normalisation for Homo, splitting the simulation over processes, finer profiling.
- **Not yet**: Homo/Hetero on Play2Perfect. Its official example already runs on gputw1.

## Setup

| | |
|---|---|
| Machine | gputw1: RTX PRO 6000 Blackwell Workstation (96 GB), driver 595.84, 500 W power cap, shared |
| Software | Isaac Sim 5.1.0, Isaac Lab v2.3.2 (source), PyTorch 2.7.0+cu128, rl_games 1.6.1 (the version Isaac Lab installs), Python 3.11 (conda) |
| Running | From the IsaacLab folder, with HeteroSimRL next to it: `./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/<script>.py --headless ...` |
| PhysX settings (official) | physics dt 1/120 s; 8 physics steps per policy step; 192 position iterations; official GPU buffers except the collision stack at >= 1024 envs (see below) |

Note: on gputw1 the 595 driver makes Isaac Sim 5.1's RTX renderer crash at start-up, even with an empty scene. Headless simulation and training are unaffected; videos are made by recording poses in PhysX and drawing them with MuJoCo (see "Videos").

## Approach: union scene

The official Factory env runs one task. For Hetero, the same batch of envs must run all three tasks:

- **Parts**: every env holds all 8 parts: peg and hole, medium gear and gear base, small and large gear, nut and bolt. An env runs one task at a time; the other tasks' parts are parked 0.6 m behind the robot, out of its reach.
- **Part configs**: every part is the official `ArticulationCfg` with only its prim path renamed, since two tasks cannot both own `/FixedAsset`.
- **Inherited settings**: the robot, controller, simulation settings and observation noise are inherited from the official `FactoryEnvCfg`.
- **Task table**: the per-task parameters (initial states, noise, reward coefficients, success thresholds, ...) are read from the official configs and functions into a table, never copied by hand (`TASK_PARAMS.md`). Each env indexes the table by its task id.
- **Observations**: they gain a task one-hot (policy 22 dims, critic 46 dims).
- **Homo and Hetero** use the same scene; they differ only in which task each env gets.

Checked so far:

- In Homo mode with the same seed, poses, observations and rewards match the official `FactoryEnv` for all three tasks (offline check).
- The union scene and the big-list scene (each env holds only its own parts) give identical poses at every step, so the parked parts do not change the physics.

`MultiUsdFileCfg` was not used because it needs identical prim names in every file, and the official parts' names differ.

## Files

| File | Contents |
|---|---|
| `hetero_factory/hetero_factory_env_cfg.py` | Union scene: 8 part slots, parking poses, `task_onehot`, task assignment settings |
| `hetero_factory/hetero_factory_env.py` | The env: subclasses the official `FactoryEnv` and overrides only task-dependent methods, each change marked `HETERO`. Gym id `Isaac-HeteroFactory-Direct-v0` |
| `hetero_factory/hetero_factory_list_env.py` | Big-list scene (each env holds only its task's parts). Gym id `Isaac-HeteroFactoryList-Direct-v0` |
| `hetero_factory/task_table.py`, `TASK_PARAMS.md`, `print_task_table.py` | Per-task parameter table, its contents and sources, and a script to regenerate it on the training machine |
| `hetero_factory/assignment.py` | Which task each env runs: Hetero even split (leftover envs rotate every episode), Homo all on one task; fixed counts for analysis runs |
| `hetero_factory/train_common.py`, `train_common_args.py` | Shared by both training scripts: official PPO config, the changes RULES allows, logging |
| `hetero_factory/rl_utils.py` | `env.step` timing, GPU sampling, one log line per epoch, per-task / pooled advantage normalisation, Homo switching |
| `hetero_factory/grad_probe.py` | Gradient diagnostics (RULES section 5) |
| `hetero_factory/profiling.py` | Groups CUDA kernels (contact detection, solver, articulation, ...) |
| `hetero_factory/pose_record.py`, `video_utils.py` | Video helpers |
| `hetero_factory/offline_check/` | Checks that run without Isaac Lab (stub modules); see its README |
| `scripts/train_hetero.py`, `train_homo.py` | Training, adapted from Isaac Lab's official rl_games `train.py` |
| `scripts/eval_policy.py` | Evaluation: each task separately, deterministic actions |
| `scripts/analyze_runs.py`, `plot_groups.py` | Success and return vs env steps and wall-clock; groups of seeds as mean and range; gradient figures |
| `scripts/record_video.py`, `render_offline.py` | Videos: poses from PhysX, drawn with MuJoCo |
| `scripts/bench_scene.py` | Random-action timing: union vs big list, per-stage times, PhysX statistics |
| `scripts/bench_policy.py` | Trained-policy timing (RULES section 6): whole episodes, contact points and phases, profiling windows, fixed task counts, synchronised start of several processes |
| `scripts/profile_step.py` | Kernel profile with the PyTorch profiler, or a capture range for Nsight Systems |
| `scripts/check_active_pose.py` | Checks that the one-read pose of the active parts equals the per-task read |
| `scripts/smoke_env.py` | First smoke test on a new machine |

## Running

```bash
# training (official PPO, 200 epochs)
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/train_hetero.py --num_envs 128 --seed 0 --headless
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/train_homo.py --num_envs 128 --seed 0 --switch rollout --block 4 --headless
# evaluation (an nn/ directory picks its last checkpoint)
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/eval_policy.py --headless \
    --checkpoint logs/rl_games/Factory/hetero_n128_s0/nn --out runs/eval/hetero_n128_s0.json
# figures (no Isaac Sim needed)
python ../HeteroSimRL/Isaac/scripts/plot_groups.py --out runs/analysis \
    --group "Hetero" tab:blue logs/rl_games/Factory/hetero_n128_s0 --group "Homo" tab:orange logs/rl_games/Factory/homo_rollout4_n128_s0
# trained-policy timing (section 6)
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/bench_policy.py --headless --num_envs 128 \
    --checkpoint logs/rl_games/Factory/hetero_n128_s1_ptadv/nn --contacts --profile_windows 5,300 --out runs/sec6/n128_hetero.json
```

Common training options:

- `--num_envs`: 3, 128 or 4096.
- `--seed`.
- `--adv_norm global|per_task|pooled`: advantage normalisation; see below.
- `--save_every N`: checkpoint every N epochs, for periodic evaluation and videos. Official 100; this changes only the checkpoint frequency, not PPO.
- `--collision_stack_size`: PhysX collision stack in bytes.
- `--grad_every`: how often to run the gradient diagnostics.

Output in `logs/rl_games/Factory/<run>/`:

- `run_info.json`: settings, changes to the official config, versions.
- `metrics.jsonl`: one line per epoch: times, `env.step` time, GPU, samples per task, advantage statistics.
- `episodes.jsonl`: one line per episode and task.
- `grad_diag.jsonl`: gradient diagnostics.

## Changes to the official settings

Required or allowed by RULES:

| Change | Why |
|---|---|
| Minibatch 512 → 384 at 3 envs | A rollout holds only 3 × 128 = 384 samples (RULES allows changes for divisibility) |
| Larger collision stack at >= 1024 envs (Hetero 1 GB, Homo 2 GB) | The official 2^28 bytes fit 128 envs only; beyond that PhysX drops contacts. A simulator buffer, not a task or PPO setting |
| Every episode 30 s | Official episodes are 10 / 20 / 30 s (peg / gear / nut), but the official reset resets all envs together; the longest is used until the team decides |
| `save_frequency` 20 (RULES runs only) | Section 8 asks for periodic evaluation and videos |

Extra analysis, each behind a switch; off means the official setting:

| Item | Why |
|---|---|
| `--adv_norm per_task` | The official setting normalises the advantages of the whole rollout together. A Hetero rollout mixes three tasks, so the nut task, with the smallest rewards, is pushed negative as a whole and some seeds never learn it. Normalising each task separately fixes it: this run shows the cause |
| `--adv_norm pooled` | From the 2026-10-07 meeting. A Homo rollout holds one task, so the official normalisation is already per task for Homo. This option normalises with statistics pooled over the latest rollout of every task: the Homo counterpart of what the official setting does in Hetero, and identical to the official setting in Hetero |
| One-read pose of the active parts (`fast_active_pose`, on by default) | The old code read the parts once per task every physics step, three times in Hetero: about 2% of a step. One read gives identical values (`check_active_pose.py`) |
| `bench_policy.py`, `profile_step.py`, Nsight | Section 6 asks for per-task step time, contacts and waiting. Random actions never mate the parts and underestimate the cost of mixing, so trained policies are used |
| `--task_counts` (timing only) | From the 2026-10-07 meeting: uneven envs per task, to fill the waiting time or group tasks by difficulty |

## Big-list scene (`--scene list`)

Each env holds only its task's parts; all envs are still in one PhysX scene.

- It measures what the parked parts of the union scene cost:
  - 2% at 3 and 128 envs;
  - about 1.8× slower union at 1024 and 4096 envs, because the parked gears sit on the parked base and their contacts are solved every step.
- Each env is built on its own (`replicate_physics=False`), because cloning env 0 would put its parts into every env.
- Each part is spawned only in its task's envs; all reads and writes go through a global-env-id → row table.
- Hetero only; each env's task is fixed when the scene is built.

## Advantage normalisation

rl_games normalises the advantages of the whole rollout to mean 0 and std 1.

- `per_task`: after rl_games builds the dataset and before the update (`after_steps`), the samples are split by the task one-hot in the observation and each task is normalised separately.
- `pooled`: the advantages from before rl_games normalises them are kept; each task's latest (count, mean, variance) is stored, and the advantages are normalised with the pooled mean and std of all tasks.
- Neither changes rl_games itself. `metrics.jsonl` logs each task's `adv_mean_<task>` and `adv_std_<task>` in every mode.

## Homo task switching

rl_games updates every 128 steps and an episode is 449 steps, so "one task per update" and "no cut episodes" cannot both hold.

- `--switch rollout` (default, RULES as written): the task changes every `--block` updates. All envs time out on the last step; the cut episodes are bootstrapped with the critic and logged as `_truncated`.
- `--switch episode`: the task changes every `--block` complete episodes; the update that spans a switch mixes two tasks.

Either way, after one cycle (a block of each task) Homo and Hetero have the same samples per task (`samples_per_task` in `metrics.jsonl`).

## Gradient diagnostics (RULES section 5)

Measured at the start of a PPO update, with the parameters before the update:

- per-task actor gradient norm;
- task-pair gradient cosine;
- cosine of each task to the total;
- update-to-update cosine: at epochs 1 and 2, then every `--grad_every` epochs on a pair of consecutive updates;
- the actual parameter change of every epoch.

Observation only: it does not touch `.grad`, the optimiser, the data or the global RNG. Its time is logged separately; `wall_time_excl_diag` is the wall time without it.

## Videos

```bash
# 1. run the physics and record every rigid body's pose (no cameras)
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/record_video.py --headless \
    --checkpoint logs/rl_games/Factory/hetero_n128_s0/nn --seconds 30 --out runs/video/hetero_n128_s0
# 2. draw it with MuJoCo (a small separate env: numpy, mujoco, imageio-ffmpeg, pillow)
MUJOCO_GL=egl python ../HeteroSimRL/Isaac/scripts/render_offline.py runs/video/hetero_n128_s0.npz
```

- One row per task, 3 tiles per row (240 × 180 each); a tile turns green on success.
- On a machine with a working driver, `--renderer rtx --enable_cameras` renders in Isaac Sim directly.
- Videos are never recorded during timing or profiling.

## Open team questions

- **Episode length**: 30 s for every task, or each task's official length?
- **Homo switching**: how and when the task changes.
- **Per-task advantage normalisation for Hetero**: should it be used? The official setting penalises Hetero.
- **Timing**: use trained policies. Random actions underestimate the cost of mixing (6-9% with random actions vs 12% during training).
- **From the 2026-10-07 meeting**: more tasks (e.g. 128), off-policy RL, reward changes. All of these go beyond RULES.
