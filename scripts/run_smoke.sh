#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${PYTHON:=/home/qjh/miniconda3/envs/mhc_env/bin/python}"

"${PYTHON}" train.py configs/vanilla.py \
  --device=cpu --dtype=float32 --compile=False \
  --n_layer=2 --n_head=2 --n_embd=64 --block_size=32 \
  --batch_size=2 --gradient_accumulation_steps=1 \
  --max_iters=2 --eval_interval=1 --eval_iters=1 \
  --out_dir=outputs/smoke-vanilla --wandb_log=False

"${PYTHON}" train.py configs/hc.py \
  --device=cpu --dtype=float32 --compile=False \
  --n_layer=2 --n_head=2 --n_embd=64 --block_size=32 \
  --batch_size=2 --gradient_accumulation_steps=1 \
  --max_iters=2 --eval_interval=1 --eval_iters=1 \
  --out_dir=outputs/smoke-hc --wandb_log=False

"${PYTHON}" train.py configs/mhc.py \
  --device=cpu --dtype=float32 --compile=False \
  --n_layer=2 --n_head=2 --n_embd=64 --block_size=32 \
  --batch_size=2 --gradient_accumulation_steps=1 \
  --max_iters=2 --eval_interval=1 --eval_iters=1 \
  --out_dir=outputs/smoke-mhc --wandb_log=False
