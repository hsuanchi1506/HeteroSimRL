#!/bin/bash
set -e
cd "$(dirname "$0")"
UV=~/.local/bin/uv
mkdir -p logs
for exp in scratch pretrain probe finetune drift nocritic floor mech ckpt proxy demo residual doorfree once critic optimism damage vreset demo2; do
    $UV run python learn.py $exp > logs/$exp.log 2>&1
done
$UV run python learn.py reps --seed 1000 > logs/reps.log 2>&1
for exp in ncsweep ckptnc rep_only sep; do
    $UV run python learn.py $exp > logs/$exp.log 2>&1
done
SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_mg.sh
for st in 102400 307200; do for j in ft:scratch ft:unlock-full ft:unlock-reset; do
    $UV run python learn.py mg --seed 0 --job $j --steps $st > logs/mg/demo_${j}_$st.log 2>&1
done; done
$UV run python analyze.py all
$UV run python analyze.py videos
