# learnability

GPU MiniGrid-style world (`gridworld.py`) → practice policies, coverage metric, coverage pre-training and finetuning on 23 downstream tasks (`learn.py`) → figures, tables and videos for STORY.md (`analyze.py`).

## Env

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd learnability
uv sync
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Run

```bash
# STORY.md step 2: which pre-trained policy learns the downstream tasks fastest?
uv run python learn.py pretrain                # 14 practice tasks x 3 seeds, 2M steps, checkpoints 50k-2M in ckpt/
uv run python learn.py gt                      # ground truth: 252 checkpoints x 23 downstream tasks, critic reset, 2M steps
uv run python learn.py pretrain_test           # 8 held-out practice tasks x 3 seeds, 1M steps
uv run python learn.py gt_test                 # ground truth for the held-out practice checkpoints

# STORY.md step 3: coverage of the noisy policy over 30 abstract states predicts learnability
uv run python learn.py metrics                 # coverage and every other reward-free metric on every checkpoint
uv run python learn.py metrics_test            # same, held-out practice checkpoints
uv run python learn.py lprobe                  # short learning probe, the strongest non-coverage baseline
uv run python learn.py mixsweep                # random-action ratio and episode budget of the coverage metric

# STORY.md step 4: coverage as the pre-training objective
uv run python learn.py pretrain_cov            # reward-free pre-training: coverage (1/q on first visit) vs SMM
uv run python learn.py gt_cov                  # their ground truth + full transfer on 3 door tasks
uv run python learn.py cover_pos               # coverage over (carry x coarse position), no door variable, with ground truth

# STORY.md figures, tables, videos
uv run python analyze.py story                 # figs/fig_story_*.png, results/story.txt
uv run python analyze.py pot_videos            # videos/5_*.mp4, videos/6_*.mp4
uv run python analyze.py cov_video             # videos/7_*.mp4

# POTENTIAL.md: every metric, mechanism and failure case
uv run python learn.py metrics2                # pseudo-reward policy-gradient coherence, head Fisher
uv run python learn.py metrics2_test
uv run python learn.py lprobe_test
uv run python learn.py mixopen                 # coverage in the door-open world
uv run python learn.py early                   # every practice task at 50k, 18 finetune runs, critic reset and full transfer
uv run python learn.py rehearse                # 100k zero-reward finetuning of every checkpoint, then coverage
uv run python analyze.py pot_report            # results/potential.txt, figs/fig_pot*.png

# REPORT.md: earlier study (critic interference, real MiniGrid)
./run_all.sh                                   # every REPORT.md experiment + figures + videos (~6 h on one RTX 5090 + 24-core CPU)
uv run python learn.py scratch                 # 22 tasks x 4 seeds from scratch, 10M steps
uv run python learn.py probe                   # zero-shot hit rates / behavior / representation stats of every checkpoint
uv run python learn.py finetune                # every practiced agent x 11 downstream tasks, 6M steps
uv run python learn.py mech                    # full / soft / reset-critic / behavior-only / representation-only / permuted
uv run python learn.py ckpt                    # learnability vs pretraining step
uv run python learn.py reps --seed 1000        # 12 finetune seeds per (condition, target)
uv run python learn.py residual                # environment solves the prefix (key given / door open)
uv run python learn.py doorfree                # hard practice tasks that never touch the door
uv run python learn.py once                    # practice without terminating at success
uv run python learn.py drift                   # PPO with / without advantage normalization
uv run python learn.py nocritic                # PPO with V = 0
uv run python learn.py ncsweep                 # 17 practice conditions x 6 targets, critic head reset vs full transfer
uv run python learn.py ckptnc                  # learnability vs pretraining step, critic head reset
uv run python learn.py rep_only                # identical initial behavior: pretrained trunk vs permuted trunk
uv run python learn.py critic                  # stale critic with warm-up / no gradient into trunk
uv run python learn.py optimism                # mean-centered stale critic, optimistic V init for scratch
uv run python learn.py sep                     # separate actor / critic networks
SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_mg.sh        # real MiniGrid: Unlock / Empty-8x8 -> DoorKey-8x8 (CPU)
uv run python analyze.py all                   # REPORT.md figures into figs/ and tables into results/summary.txt
uv run python learn.py demo2                   # 12 seeds x (scratch, open_door, face_door, face_door + critic reset) on goal, checkpoints for videos
uv run python learn.py mg --seed 0 --job ft:unlock-reset --steps 102400   # one real-MiniGrid video checkpoint (run_all.sh makes all six)
uv run python analyze.py videos                # videos/0_*.mp4 - 4_*.mp4
```
