#!/usr/bin/env bash
#
# Multi-node torchrun launcher for AngelSlim QAT + DeepSpeed ZeRO-3.
#
# Usage:
#   bash scripts/qat/run_qat_multinode.sh
#
# Run the SAME command on each of the N nodes (same container image, same
# shared filesystem). The script auto-detects the current node's rank from
# ``NODE_IP_LIST`` / ``hostname -I`` so you do NOT have to edit per-node.
#
# Environment variables you can override:
#   CONFIG        QAT yaml path
#   MODEL_PATH    overrides ``model.model_path`` in the yaml
#   FROM_PTQ_CKPT overrides ``compression.QAT.from_ptq_ckpt`` in the yaml
#   SAVE_PATH     overrides ``global.save_path`` in the yaml
#   MASTER_PORT   default 29500
#
# The following are auto-detected from the taiji environment:
#   NODE_IP_LIST   e.g. "28.58.246.88:8,28.48.3.150:8,28.48.6.124:8,28.59.20.59:8"
#   HOST_NUM       total nodes
#   HOST_GPU_NUM   GPUs per node
#   LOCAL_HOSTNAME hostname of the current container
#
set -euo pipefail

# ---- conda env + NCCL tuning for multi-host RDMA ----
source /apdcephfs_zwfy2/share_301053287/brunosu/init_scripts/init_conda.sh
pkill -f "python3 -"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_TIMEOUT=600

# ---- Detect rank / world from environment ----
if [ -z "${NODE_IP_LIST:-}" ]; then
    echo "ERROR: NODE_IP_LIST env var is empty. This script expects a"
    echo "       multi-node taiji job (e.g. NODE_IP_LIST=ip1:8,ip2:8,...)."
    exit 1
fi

# Split NODE_IP_LIST into arrays of IPs and slot counts.
IFS=',' read -r -a NODE_ENTRIES <<< "${NODE_IP_LIST}"
NNODES_AUTO=${#NODE_ENTRIES[@]}

# Collect all local IPs on this container (space-separated).
LOCAL_IPS=$(hostname -I 2>/dev/null || echo "")

# First IP in NODE_IP_LIST is the master.
FIRST_ENTRY=${NODE_ENTRIES[0]}
MASTER_ADDR_AUTO=${FIRST_ENTRY%%:*}

# Figure out which entry in NODE_ENTRIES matches the current container.
NODE_RANK_AUTO=-1
for i in "${!NODE_ENTRIES[@]}"; do
    ip=${NODE_ENTRIES[$i]%%:*}
    if [[ " ${LOCAL_IPS} " == *" ${ip} "* ]]; then
        NODE_RANK_AUTO=$i
        break
    fi
done

if [ "${NODE_RANK_AUTO}" -lt 0 ]; then
    echo "ERROR: Could not match any local IP (${LOCAL_IPS}) to NODE_IP_LIST entries:"
    for e in "${NODE_ENTRIES[@]}"; do echo "  - ${e}"; done
    exit 1
fi

# GPUs per node: take from the first entry (slots after ``:``).
GPUS_PER_NODE=${FIRST_ENTRY##*:}
if [[ "${GPUS_PER_NODE}" == "${FIRST_ENTRY}" ]] || ! [[ "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]]; then
    GPUS_PER_NODE=${HOST_GPU_NUM:-8}
fi

# Allow explicit overrides.
NNODES=${NNODES:-${NNODES_AUTO}}
NPROC_PER_NODE=${NPROC_PER_NODE:-${GPUS_PER_NODE}}
NODE_RANK=${NODE_RANK:-${NODE_RANK_AUTO}}
MASTER_ADDR=${MASTER_ADDR:-${MASTER_ADDR_AUTO}}
MASTER_PORT=${MASTER_PORT:-29500}

# ---- Training config ----
CONFIG=${CONFIG:-hy3_a3b_fp8_static_end2end_learn_scale_zero3.yaml}
MODEL_PATH=${MODEL_PATH:-}
FROM_PTQ_CKPT=${FROM_PTQ_CKPT:-}
SAVE_PATH=${SAVE_PATH:-}

# Move to project root (the directory containing tools/run.py).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

echo "=== Multi-node launch config ==="
echo "  project_root  : ${PROJECT_ROOT}"
echo "  nnodes        : ${NNODES}"
echo "  nproc_per_node: ${NPROC_PER_NODE}"
echo "  node_rank     : ${NODE_RANK}   (auto-detected: ${NODE_RANK_AUTO})"
echo "  master_addr   : ${MASTER_ADDR} (auto: ${MASTER_ADDR_AUTO})"
echo "  master_port   : ${MASTER_PORT}"
echo "  local_ips     : ${LOCAL_IPS}"
echo "  NODE_IP_LIST  : ${NODE_IP_LIST}"
echo "  CONFIG        : ${CONFIG}"
echo "================================="

# Compose the model override flags.
EXTRA_ARGS=()
if [ -n "${MODEL_PATH}" ]; then
    EXTRA_ARGS+=(--model-path "${MODEL_PATH}")
fi
if [ -n "${FROM_PTQ_CKPT}" ]; then
    EXTRA_ARGS+=(--from-ptq-ckpt "${FROM_PTQ_CKPT}")
fi
if [ -n "${SAVE_PATH}" ]; then
    EXTRA_ARGS+=(--save-path "${SAVE_PATH}")
fi

torchrun \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    tools/run.py \
    -c "${CONFIG}" \
    "${EXTRA_ARGS[@]}"
