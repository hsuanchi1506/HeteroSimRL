# HeteroSimRL

Analyze the efficiency bottlenecks of **contact-rich multi-task RL with GPU-parallel simulation**.

We compare homogeneous and heterogeneous task execution to determine whether differences in task/contact workload cause simulator waiting, GPU idle time, or solver imbalance, and whether these system-level effects influence RL learning efficiency.

---

# 1. Task / Scene Definition

We will first evaluate two separate task families. **The two families should remain completely independent and should not be mixed within the same experiment.**

First determine which family can be ported most cleanly across all three simulators. If both are practical, implement both.

## Option A — Isaac Factory

Tasks:

- PegInsert
- GearMesh
- NutThread

All **scene, robot, asset, controller, reward, reset, and RL configurations** should follow the official `isaac-sim/IsaacLab` `direct/factory` implementation as the canonical definition. Do not manually retune these settings.

The three tasks already share the same family of task configurations and control settings.

Reference:

`isaac-sim/IsaacLab`  
`source/isaaclab_tasks/isaaclab_tasks/direct/factory/`

## Option B — Play2Perfect

Tasks:

- Tight Insertion
- Beam Assembly
- Screwing

All settings should follow the official:

`kushal2000/play2perfect`

including:

`isaacsimenvs/cfg/`  
`isaacsimenvs/tasks/precise_assembly/`  
`assets/urdf/`

Do not mix Factory controller/reward/config settings into Play2Perfect. The official repository already provides the training code, task configurations, and assets.

---

# 2. Three Physics Backends

Implement the same canonical task family on:

- **Isaac / PhysX**
- **Genesis / RigidSolver**
- **MuJoCo / MJWarp**

All experiments should use GPU simulation.

Configuration principle:

> **Task-level settings should strictly follow the canonical repository, while solver-level settings should remain as close as possible to each backend's official defaults.**

Geometry, mass/inertia, friction, gravity, timestep, controller, and initial states should therefore be ported from Factory or Play2Perfect.

However, solver-specific parameters such as iterations, tolerances, and contact-buffer settings should **not** be forced to use numerically identical values across PhysX, Genesis, and MJWarp. Instead, keep the native/default settings and record them clearly.

Implementation references:

- **Isaac:** use the canonical Isaac implementation directly.
- **Genesis:** follow the official `Genesis-Embodied-AI/genesis-world` manipulation and parallel-environment implementation.
- **MJWarp:** follow the official `google-deepmind/mujoco_warp` `benchmarks/` and MJCF workflow; use the official definitions of `nworld`, `nconmax`, and `njmax`.

---

# 3. Unified RL Setup

For each task family, use **its official RL implementation** and port the same setup across all three physics backends.

### Factory

Use the maintained **`rl_games` PPO agent configuration** registered for the corresponding Isaac Lab Factory task.

Do not redesign or retune PPO from scratch.

### Play2Perfect

Follow:

`isaacsimenvs/train.py`  
the vendored `rl_games` implementation in the repository.

Play2Perfect pretrained checkpoints can be downloaded with:

```bash
python download_checkpoints.py
```

For example:

```text
pretrained_assembly/tight_insertion/model.pth
```

The repository also provides:

`evaluation/eval_offline.py`

All three backends should use the same:

- network architecture
- PPO hyperparameters
- rollout length
- reward / success definition
- task sampling
- shared policy

Use the same `task_id` representation in the multi-task policy observation.

Do not additionally tune PPO except for changes required by `num_envs` or minibatch divisibility. Any necessary changes should be documented.

---

# 4. Main Experiments

Evaluate:

`3 / 128 / 4096 environments`

## A. Homogeneous Scaling

Each rollout/update contains only one task, while the three tasks are alternated and trained using the same shared policy:

```text
Peg × N
→ Gear × N
→ Nut × N
→ Peg × N
→ ...
```

## B. Heterogeneous Parallel

Each rollout simultaneously contains all three tasks:

```text
Peg + Gear + Nut
```

For `N = 3`, use one environment per task.

For larger `N`, distribute environments as evenly as possible across tasks. Any extra environments should rotate across tasks so that long-term samples per task remain approximately equal.

Compare:

- success vs. environment steps
- success vs. wall-clock time
- reward vs. environment steps
- reward vs. wall-clock time
- total training time

After one complete `Peg → Gear → Nut` cycle, Homo and Hetero should have accumulated approximately the same number of samples for each task.

---

# 5. Gradient / Learning Analysis

Test the following hypothesis:

> Homo sees only one task in each PPO update, while Hetero sees all tasks in every update. Hetero may therefore provide a more informative multi-task gradient, but at the cost of lower simulation throughput.

Record only:

- **per-task actor gradient norm**
- **task-pair gradient cosine similarity**
- **update-to-update gradient cosine similarity**
- **per-task success / reward**

Main questions:

1. Does the gradient direction change sharply when Homo switches tasks?
2. Do gradients from different tasks conflict?
3. Does Hetero produce more stable or balanced updates?
4. Is any learning benefit large enough to compensate for slower heterogeneous simulation?

Gradient diagnostics only need to be sampled every fixed number of iterations, rather than computed for every update.

---

# 6. Profiling During RL Training

The main question is:

> **Do different task/contact workloads cause some environments to finish earlier but wait for slower environments, leading to GPU idle time or slower global simulation steps?**

Divide trajectories approximately into:

- **Pre-contact**
- **Contact**
- **Heavy-contact / insertion / jamming**

Record only the following core metrics.

### Simulation Workload

- **active contacts per task/environment**
- **physics step time per task**
- **contact / constraint solving time**

### Parallel Efficiency

- **overall SPS**
- **GPU utilization**
- **waiting / synchronization time**
- **slowest-task vs. average-task step time**

Define the **Straggler Ratio** as:

`Straggler Ratio = slowest [task/env] step time / average [task/env] step time`

A higher ratio indicates stronger workload imbalance and more potential waiting between parallel environments.

We specifically want to observe behavior such as:

```text
fast task finishes
        ↓
waits for heavy-contact task
        ↓
next global simulation step
```

Compare Homo and Hetero under `3 / 128 / 4096 environments` and determine whether waiting overhead increases with task heterogeneity.

Heavy profilers such as Nsight should only be enabled during selected training windows to avoid distorting normal training performance.

---

# 7. Heterogeneous Execution Analysis

Inspect the source code and documentation of all three simulators, focusing only on:

> **How different tasks/environments are represented and executed together, especially when their contact workloads differ.**

For each backend, determine:

1. How are bodies, joints, and contacts from different environments batched or stored?
2. When contact workloads differ, are environments processed independently or through shared/batched kernels or solvers?
3. Do smaller/faster environments need to wait for larger or heavier-contact environments?
4. Do heterogeneous scenes introduce overhead from padding, inactive threads, divergence, or synchronization?
5. Is the main bottleneck caused by **contact handling, constraint solving, or cross-environment synchronization**?

Do not assume that MJWarp, Genesis, or PhysX uses a particular execution strategy. Verify the behavior through source code, profiling results, and execution traces.

---

# 8. Validation / Reproducibility

For the initial study, run each experiment **once**.

Keep fixed:

- same GPU
- same software commit/version
- rendering disabled during normal training and profiling
- same task sampling / environment-step budget
- exclude compilation / JIT warm-up from timing

During training, periodically run lightweight evaluation and save a small number of **low-resolution videos** for qualitative validation.

Save:

- representative evaluation videos for each task
- periodic Homo evaluation videos
- periodic Hetero evaluation videos

The videos are only used to verify that:

1. scenes and initial states are visually consistent across backends
2. object/contact behavior does not show obvious errors
3. the policy is learning and executing the intended task
4. porting has not introduced obvious physics failures

Video recording should **not** be enabled during normal profiling runs.

If `4096 environments` causes OOM or exceeds simulator capacity, report the maximum feasible environment (for example, discount to 2048 env instead) count instead of reducing task complexity.

---

# Expected Outcome

The goal is to separate two effects:

```text
Heterogeneous tasks
        ↓

Simulation side:
contact / solver imbalance
→ waiting / GPU idle / lower throughput

Learning side:
more task-diverse data per PPO update
→ potentially more informative / stable multi-task gradients
```

Ultimately, determine:

> **Whether the learning benefit of heterogeneous parallelism is sufficient to compensate for its simulation slowdown, and whether the main optimization opportunity lies in task scheduling, contact batching, or solver execution.**