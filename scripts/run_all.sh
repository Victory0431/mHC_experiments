#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${NPROC_PER_NODE:=2}"
: "${TORCHRUN:=/home/qjh/miniconda3/envs/mhc_env/bin/torchrun}"

"${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py configs/vanilla.py
"${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py configs/hc.py
"${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py configs/mhc.py
