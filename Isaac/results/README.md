# Isaac / PhysX results (by RULES section)

Status 2026-10-10. Chinese version: [README.zh-TW.md](README.zh-TW.md). How to run: [../README.md](../README.md). Raw logs, checkpoints and videos are on gputw1 and are not in the repository.

## 1. Task families

- **Factory**: done on PhysX (everything below).
- **Play2Perfect**: the official example runs on gputw1 in its own environment (Isaac Lab 2.3.2.post1, the repository's vendored rl_games). Official offline evaluation of the released policies, 1024 envs, first episode per env:

  | Task | Success |
  |---|---|
  | Tight insertion | 96.9% |
  | Beam assembly, step 1 | 97.1% |
  | Beam assembly, step 2 | 92.2% |
  | Screwing | 61.4% |

- Every released checkpoint has 6 SAPG blocks. The play policy was trained with 24,576 envs (blocks of 4,096); the four stage-2 policies with 12,288 envs in blocks of 2,048, with epochs counted from 0 (read from the checkpoints themselves). The repository's default config (8,192 envs, blocks of 4,096) gives 2 blocks, so stage-2 finetuning from the released play policy fails to load; 12,288 envs with blocks of 2,048 load it.
- Homo/Hetero on Play2Perfect (2026-10-13; one seed, official advantage normalisation, 12,288 envs, union scene validated by the released policies: 95.9 / 97.3 / 92.3 / 61.0% vs official 96.9 / 97.1 / 92.2 / 61.4%). Deterministic evaluation, tight insertion / beam step 1 / beam step 2 / screwing:

  | Epoch | Hetero | Homo, run 1 | Homo, run 2 |
  |---|---|---|---|
  | 2,000 | 30 / 88 / 66 / 0% | 3 / 66 / 52 / 0% | 2 / 60 / 54 / 0% |
  | 6,000 | 71 / 96 / 83 / 46% | 32 / 88 / 75 / 0% | 21 / 86 / 72 / 0% |
  | 10,000 | 83 / 96 / 88 / 62% | crashed | crashed |

  - **Homo does not fit in GPU memory at the official scale.** Both Homo runs died (epochs 7,254 and 6,135) when PhysX could not grow its contact buffers: in a Homo block all 12,288 envs run the same task, so the contact-heavy tasks put all their contacts in the same steps (up to 92.6 GB; Hetero at most 48.8 GB). Kept as a result (2026-10-13); not rerun.
  - Per sample, Hetero learns every task faster, and only Hetero learns screwing (Homo had more than 300 M screwing steps before crashing, about twice what Hetero needed to start). Per hour, tight insertion and the beam steps are about even (Homo epochs take about 5.5 s, Hetero about 10 s).
  - Speed: a mixed step costs about as much as screwing alone (574 vs 717 ms per step at 12,288 envs); 768 screwing envs among tight-insertion envs already raise a step from 70 to 485 ms. Screwing legs touch their fixture from the reset on, so lining up contact phases in time cannot make steps cheaper here ([figures/p2p_alignment.png](figures/p2p_alignment.png)).

  ![Play2Perfect](figures/p2p_success.png)

## 2. Backend settings

- Isaac Sim 5.1.0 (PhysX 5.6.1 / omni.physx 107.3), Isaac Lab v2.3.2, RTX PRO 6000 Blackwell Workstation, driver 595.84.
- Official Factory simulation settings, unchanged:
  - physics dt 1/120 s;
  - 8 physics steps per policy step;
  - 192 solver position iterations;
  - `gpu_max_num_partitions` 1;
  - official GPU buffers, except the collision stack: 1 GB (Hetero) or 2 GB (Homo) at >= 1024 envs, because the official 2^28 bytes drop contacts there.

## 3. RL setup

- Official rl_games PPO config of the Factory tasks:
  - LSTM, asymmetric critic;
  - horizon 128, minibatch 512 (384 at 3 envs);
  - 200 epochs.
- One shared policy; the task enters the observation as a one-hot.
- Changes to the official settings are listed in [../README.md](../README.md#changes-to-the-official-settings).

## 4. Main experiments (128 envs)

Final success, deterministic evaluation (2 episodes × 128 envs per task); mean of seeds, peg / gear / nut:

| Group | Seeds | Success |
|---|---|---|
| Hetero, official advantage normalisation | 3 | 87 / 88 / **42**% (nut per seed: 65, 6, 55) |
| Hetero, per-task advantage normalisation | 5 | 92 / 83 / 90% |
| Homo (task switch every 4 updates) | 5 | 91 / 83 / 92% |

Training success during the first 1 M env steps (peg / gear / nut):

| Group | Success |
|---|---|
| Hetero (official) | 0.53 / 0.52 / 0.06 |
| Hetero (per-task) | **0.60 / 0.56 / 0.64** |
| Homo | 0.29 / 0.41 / 0.52 |

Wall-clock time (200 epochs, old pose read):

| Group | Total | Simulation |
|---|---|---|
| Hetero | 2.44-2.45 h | 2.25 h |
| Homo | 2.21 h | 2.02 h |

![success vs steps and time](figures/n128_success.png)

Line: mean of the seeds; band: lowest to highest seed. Return vs steps and time: [figures/n128_return.png](figures/n128_return.png).

- **With the official setting, Hetero loses on the nut task.** rl_games normalises the advantages of the whole rollout together. In a Hetero rollout the nut task, whose early rewards are small, is pushed negative as a whole: mean about -1.2, std 0.14-0.35 in the first epochs. One seed never learns it.
- **Per-task normalisation removes the problem.** Hetero then learns all three tasks faster than Homo, per sample and per hour, and ends equal.
- **Normalising the tasks together also hurts Homo** (meeting TODO1 below), so official Homo's edge on the nut task comes from the normalisation.
- **RULES "run once" set** (`rules_*`, current code, official advantage normalisation, seed 0, checkpoints every 20 epochs). Success at epoch 200:

  | Group | Peg / gear / nut | Total time |
  |---|---|---|
  | Hetero | 84 / 94 / 62% | 2.41 h |
  | Homo | 97 / 78 / 96% | 2.20 h |

  These match the earlier seed-0 runs. Curves: [figures/rules_n128_success.png](figures/rules_n128_success.png).
- **3 envs** (RULES runs, official 200 epochs): neither group learns. 200 epochs at 3 envs are 77k env steps, 1/43 of the 128-env runs; success is 0-8% at every checkpoint. Only the time is usable:

  | Group | Total | Simulation |
  |---|---|---|
  | Hetero | 1.66 h | 1.63 h |
  | Homo | 1.60 h | 1.57 h |

  Hetero simulates 4% longer, less than the 7% of section 6, because the untrained policies rarely insert the parts. Homo's time excludes a 7.1 h pause (the GPU guard stopped the run while another user's job started).
- **4096 envs** (RULES runs, official 200 epochs, 105 M env steps): **Hetero clearly beats Homo**, unlike at 128 envs.

  | Epoch | Hetero (official), deterministic eval | Homo |
  |---|---|---|
  | 40 | 28 / 62 / 9% | 27 / 0 / 18% |
  | 80 | 97 / 86 / 89% | 52 / 20 / 46% |
  | 200 | 97 / 83 / 97% | 85 / 42 / 96% |

  | Group | Total | Simulation | GPU memory (max) |
  |---|---|---|---|
  | Hetero | 23.0 h | 17.5 h | 37.5 GB |
  | Homo | 21.3 h | 15.8 h | 32.7 GB |

  - Hetero learns peg and gear much faster, per sample and per hour, although it simulates 11% longer (as at 128 envs). Homo's gear stays low (training success 65% in the last fifth, deterministic 42% at epoch 200). Nut is similar.
  - Hetero's nut did not drop in the deterministic evaluation here (97%), unlike seed 0 at 128 envs (62%). One seed each: scale or luck cannot be told apart.
  - Why Homo falls behind at 4096 envs is not established. An inference: the minibatch stays 512, so an epoch at 4096 envs has 32 times the gradient steps, and a Homo block (4 epochs on one task) becomes thousands of single-task steps, which favours forgetting. Homo's gear curve keeps rising and falling, consistent with this, but it is not tested.
  - Homo's update-to-update cosine is 0.04 across a switch and 0.14 within a task (128 envs: -0.30 and +0.27).

  ![4096 envs](figures/rules_n4096_success.png)

## 5. Gradients (128 envs)

1. **Does the direction flip when Homo switches tasks?** Yes. Consecutive updates across a switch have cosine -0.21 to -0.32 (5 seeds); within a task, +0.23 to +0.35.
2. **Do task gradients conflict?**
   - Peg and gear mostly agree (+0.2 to +0.35).
   - Nut against either of the others is near orthogonal on average, and negative 30-60% of the time.
3. **Is Hetero more balanced?** Not with the official normalisation: the nut signal is suppressed. With per-task normalisation, all three tasks get signals of the same size.
4. **Does the learning benefit pay for the slower simulation?** Not with the official normalisation; yes with per-task normalisation.

![gradients](figures/n128_gradients.png)

## 6. Profiling with trained policies

`scripts/bench_policy.py` with the trained policy (Hetero, per-task normalisation, seed 1):

- whole episodes, timed like training;
- contacts of each env's held part with its fixed part and with the fingers, read from PhysX every step;
- three phases per env:
  - pre-contact;
  - contact;
  - heavy contact / insertion (the official "engaged" check);
- kernel profiles at step 5 (pre-contact) and step 300 (insertion).

| | 3 envs | 128 envs | 4096 envs |
|---|---|---|---|
| Straggler Ratio (slowest Homo task / mean of the three) | 1.06 | 1.10 | 1.20 |
| Hetero / Homo mean (training schedules) | 1.07 | 1.09 | 1.08 |
| Hetero / slowest single task | 1.02 | 1.00 | 0.90 |
| GPU idle, waiting for the CPU (profiled steps) | 26-34% | 23-31% | 5-7% |
| SPS, Homo peg / gear / nut | 14 / 12 / 12 | 522 / 401 / 411 | 2458 / 1463 / 1588 |
| SPS, Hetero | 12 | 403 | 1610 |

![step time relative to the Homo mean](figures/sec6_relative_step_time.png)

- **Small scenes wait for the heaviest contact.** At 3 and 128 envs the GPU is not full, and a mixed step takes as long as the slowest single task. At 4096 envs the GPU is saturated (95% busy), and a mixed step falls between the mean and the slowest task.
- **The extra time comes from insertion contacts.**
  - Before contact the three tasks take about the same time per step.
  - Once the gear and the nut are engaged, a step costs about 40 ms (3 envs), 70 ms (128) or 1.2 s (4096) more. The peg costs almost nothing extra.
  - Contact points per env while engaged: peg about 85, gear about 106, nut about 250.
- **The cost of mixing stays at 7-9% across scales**, while the imbalance between tasks (Straggler Ratio) grows from 1.06 to 1.20.
- **Contact detection is sized for the most complex part.** At 128 envs, before any insertion, a mixed scene spends 25 ms on contact detection, close to the nut alone (28 ms), while peg and gear alone take 14-15 ms. Nsight agrees: the convex-vs-mesh midphase is 1.5-2.5× slower in mixed scenes.
- **Bottleneck**: contact and constraint solving (inside the articulation kernels at small scale, an inference since the parts are one-link articulations; in the solver kernels at 4096 envs), then contact detection. About 2,250 kernel launches per physics step at every scale give a fixed cost that leaves the GPU waiting for the CPU at small scale.
- **Random actions underestimate mixing.** Random actions never mate the parts, so the three tasks cost about the same: 6-9% slowdown with random actions vs 12% in training. A trained policy reproduces the 12% of training ([figures/policy_vs_random_step_time.png](figures/policy_vs_random_step_time.png): Hetero follows the slowest task with the trained policy; with random actions the three tasks stay close).

![step time and contact phases](figures/sec6_step_time_phases.png)

![kernel time during insertion](figures/sec6_kernels.png)

## 7. How PhysX runs heterogeneous envs (source: PhysX 5.6.1)

1. **Storage**: the whole GPU scene is one island. Bodies, articulations and contacts are batched by type, not by env. Articulations are processed in blocks of 32, padded to the largest link count (Franka: 11 links).
2. **Processing**: every env goes through the same shared kernels. Every env does the same 192 position iterations; the iteration count is the scene-wide maximum, and there is no early exit on convergence.
3. **Do small envs wait for large ones?** Yes. Each kernel ends when its heaviest work finishes, and several buffer sizes and launch grids follow scene-wide maxima.
4. **Padding, divergence, synchronisation**:
   - Padding exists (articulation blocks, maxima-sized buffers).
   - Warp-level divergence is not the main cost: putting each task's envs in contiguous blocks changed step time by less than 1%.
   - Each physics step makes about 2,300 kernel launches regardless of env count (no CUDA graphs).
5. **Main bottleneck**: contact and constraint solving over the 192 iterations, and the fixed launch cost at small scale. Cross-env synchronisation itself is not the main cost.

## 8. Validation and reproducibility

- **Hardware and software**: same GPU and versions for every run (recorded in each `run_info.json`).
- **Rendering**: disabled during training and timing.
- **Budgets**: Homo and Hetero have the same env-step budget and the same samples per task after each cycle.
- **Warm-up**: excluded from timing.
- **RULES "run once" set**:
  - checkpoints every 20 epochs;
  - deterministic evaluation (1 episode × 128 envs per task) at epochs 40, 80, 120, 160 and 200;
  - a 30 s low-resolution video at each of those epochs, drawn offline because the RTX renderer crashes on this driver.
- **Raw data**: in the owner's notes and on gputw1.

## Beyond RULES (and why)

| Item | Why |
|---|---|
| More seeds at 128 envs (3-5) | Seed 1 of official Hetero never learned the nut task; one seed could not show whether that was typical |
| Per-task advantage normalisation | To test, and confirm, why official Hetero loses the nut task |
| Trained-policy timing, Nsight | Random actions underestimate the cost of mixing |
| One-read pose of the active parts | Removed 2% of a Hetero step that came from our code, not PhysX |
| Big-list scene | Measures what the union scene's parked parts cost |

## Meeting TODOs (2026-10-07)

- **TODO1** (fair normalisation for Homo): with `--adv_norm pooled`, Homo gets what the official setting does in Hetero: each rollout is normalised with statistics pooled over the latest rollout of every task. Homo, 128 envs, 3 seeds. Final success, peg / gear / nut:

  | | All tasks normalised together | Each task normalised on its own |
  |---|---|---|
  | Hetero | 87 / 88 / **42**% (official, 3 seeds) | 92 / 83 / 90% (5 seeds) |
  | Homo | 86 / **70** / 82% (pooled, 3 seeds) | 91 / 83 / 92% (official, 5 seeds) |

  - **Normalising the tasks together hurts both.** Hetero loses the nut task. Homo learns peg and gear more slowly, and on gear all three seeds end below official Homo's worst seed (64-74% vs 75%).
  - **With per-task normalisation, Hetero and Homo end equal**, and Hetero learns faster early. When both normalise per task, mixing the tasks costs nothing in learning.
  - **Measured shift**: in the first 50 epochs, pooling moves the mean advantage of a Homo rollout by +0.25 (peg), +0.37 (gear) and -0.44 (nut); later it stays near 0. Why Hetero and Homo lose different tasks is still open.

  ![2x2 of normalisation](figures/todo1_success.png)
- **TODO2** (split the simulation over several processes on one GPU): **no gain; one mixed process is fastest** at every scale. Env steps per second, all tasks together:

  | Total envs | One mixed process | Three single-task processes at once | Three single-task processes one after another |
  |---|---|---|---|
  | 3 | 12 | 4.9 (0.41×) | 4.5 (0.38×) |
  | 128 | 413 | 179 (0.43×) | 167 (0.40×) |
  | 4096 | 1620 | 1458 (0.90×) | 1419 (0.88×) |

  - **The GPU serves the processes in turns.** Without MPS it time-slices them: under concurrency each process's step takes about the sum of the three processes' steps alone.
  - **At small scale most of a step is a fixed cost.** About 2,300 kernel launches per physics step, whatever the env count. Splitting the envs pays that cost three times; mixing pays it once.

  ![split compute](figures/todo2_split_compute.png)
- **TODO3** (finer profiling during insertion). The mixed step (N envs, profiled at step 300) is compared with each task alone at N/3 envs:

  | Envs | Mixed vs slowest task alone | Mixed vs sum of the three | Share of the mixed step that peg / gear / nut envs wait |
  |---|---|---|---|
  | 3 | 1.07× | 0.39× | 26 / 8 / 7% |
  | 128 | 1.16× | 0.42× | 34 / 15 / 14% |
  | 4096 | 2.27× | 0.90× | no idle wait (the work adds up) |

  - **Side by side at small scale.** At 3 and 128 envs the tasks run side by side, so cheap tasks wait for the heaviest.
  - **Filling the wait.** Adding cheap peg envs to the 128-env mixed scene:

    | Peg / gear / nut envs | Step time | Total env steps per second |
    |---|---|---|
    | 42 / 43 / 43 (even) | baseline | baseline |
    | 128 / 43 / 43 | +2% | 1.63× |
    | 256 / 43 / 43 | +11% | 2.41× |

  - **Grouping by difficulty.** Gear + nut mixed (309 ms) costs the same as the full three-task mix (307 ms). Running "peg alone" and "gear + nut" in turns, with equal samples per task, costs 283 ms per 128 env steps: Homo 277, full mix 307.

  ![waiting and filling](figures/todo3_waiting.png)
