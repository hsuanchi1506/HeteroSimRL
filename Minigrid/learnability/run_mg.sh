#!/bin/bash
cd "$(dirname "$0")"
UV=~/.local/bin/uv
mkdir -p logs/mg
for s in ${SEEDS:-0 1 2 3 4}; do for j in pre:unlock pre:empty; do echo "$s $j"; done; done | \
    xargs -P 10 -n 2 sh -c "$UV run python learn.py mg --seed \$0 --job \$1 > logs/mg/\$1_\$0.log 2>&1"
for s in ${SEEDS:-0 1 2 3 4}; do for j in ft:scratch ft:unlock-full ft:unlock-reset ft:empty-full ft:empty-reset; do echo "$s $j"; done; done | \
    xargs -P 16 -n 2 sh -c "$UV run python learn.py mg --seed \$0 --job \$1 > logs/mg/\$1_\$0.log 2>&1"
