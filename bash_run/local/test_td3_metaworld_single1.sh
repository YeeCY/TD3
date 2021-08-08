#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/../../..")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a env_names=(
  button-press-topdown-wall-v2
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
