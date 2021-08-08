#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a env_names=(
  assembly-v2
  basketball-v2
  bin-picking-v2
  button-press-topdown-wall-v2
  coffee-button-v2
#  coffee-pull-v2
#  coffee-push-v2
#  disassemble-v2
#  hammer-v2
#  hand-insert-v2
#  handle-press-side-v2
#  handle-press-v2
#  handle-pull-side-v2
#  handle-pull-v2
#  lever-pull-v2
#  peg-insert-side-v2
#  peg-unplug-side-v2
#  pick-out-of-hole-v2
#  pick-place-v2
#  plate-slide-back-side-v2
#  plate-slide-back-v2
#  plate-slide-side-v2
#  plate-slide-v2
#  push-back-v2
#  push-v2
#  push-wall-v2
#  reach-wall-v2
#  shelf-place-v2
#  soccer-v2
#  stick-pull-v2
#  stick-push-v2
#  sweep-into-v2
#  sweep-v2
#  window-open-v2
)

declare -a seeds=(0 1 2 3)

for env_name in "${env_names[@]}"; do
  for seed in "${seeds[@]}"; do
    export CUDA_VISIBLE_DEVICES=$seed
    nohup \
    python $PROJECT_DIR/main_metaworld.py \
      --env_name $env_name \
      --seed $seed \
      --save_dir $PROJECT_DIR/logs/td3_official_single \
    > $PROJECT_DIR/terminal_logs/td3-$env_name-seed-$seed.log 2>&1 &
  done
done
