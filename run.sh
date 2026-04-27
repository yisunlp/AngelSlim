#!/usr/bin/env bash
#
# Multi-node launcher built on top of the ``deepspeed`` runner.
#
# Behaviour:
#   * Translates ``NODE_IP_LIST`` (taiji format ``ip:slots,ip:slots,...``)
#     into a DeepSpeed-compatible ``hostfile``.
#   * Runs ``prelude_multinode.sh`` (pkill of stale workers) on every node
#     via SSH, BEFORE the deepspeed launcher fans out — otherwise the new
#     workers would compete with leftovers from a previous crashed run.
#   * Invokes ``deepspeed --hostfile=hostfile ...`` once on the master;
#     DeepSpeed itself takes care of SSH-spawning on every other node.
#
# Run THIS SCRIPT ONLY ON THE MASTER NODE:
#     bash launch_qat_deepspeed_multinode.sh
#
# Environment overrides:
#     CONFIG          QAT yaml
#     MODEL_PATH      overrides ``model.model_path``
#     FROM_PTQ_CKPT   overrides ``compression.QAT.from_ptq_ckpt``
#     SAVE_PATH       overrides ``global.save_path``
#     MASTER_PORT     default 29500
#     LOG_DIR         where stdout gets written; defaults to ./logs/ds_<ts>
#
set -euo pipefail

# ---- conda + NCCL/IB tuning (mirrors the FSDP run.sh) ----
# init_conda.sh references PYTHONPATH which may be unbound in a fresh
# shell; relax ``-u`` while sourcing.
set +u
source /apdcephfs_zwfy2/share_301053287/brunosu/init_scripts/init_conda.sh
set -u

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
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

# ---- node list / paths ----
if [ -z "${NODE_IP_LIST:-}" ]; then
    echo "ERROR: NODE_IP_LIST is not set."
    exit 1
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_DIR}"

LOG_DIR=${LOG_DIR:-"${PROJECT_DIR}/logs/ds_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${LOG_DIR}"

# Translate NODE_IP_LIST -> hostfile (one ``ip slots=N`` per line).
HOSTFILE="${LOG_DIR}/hostfile"
echo "${NODE_IP_LIST}" | tr ',' '\n' | sed 's/:/ slots=/g' > "${HOSTFILE}"
echo "=== hostfile ==="
cat "${HOSTFILE}"
echo

MASTER_IP=$(head -1 "${HOSTFILE}" | awk '{print $1}')
MASTER_PORT=${MASTER_PORT:-29500}

# ---- per-node prelude (pkill stale workers) ----
PRELUDE="${PROJECT_DIR}/prelude_multinode.sh"
if [ ! -x "${PRELUDE}" ]; then
    chmod +x "${PRELUDE}" 2>/dev/null || true
fi
echo "=== prelude on every node ==="
while read -r ip _; do
    [ -z "${ip}" ] && continue
    echo "[prelude] ssh ${ip}"
    ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR "${ip}" \
        "bash ${PRELUDE}" 2>&1 | sed "s|^|  [${ip}] |"
done < "${HOSTFILE}"
echo

# ---- training config ----
CONFIG=${CONFIG:-hy3_a3b_fp8_static_end2end_learn_scale_zero3.yaml}

EXTRA_ARGS=()
if [ -n "${MODEL_PATH:-}" ];    then EXTRA_ARGS+=(--model-path "${MODEL_PATH}");       fi
if [ -n "${FROM_PTQ_CKPT:-}" ]; then EXTRA_ARGS+=(--from-ptq-ckpt "${FROM_PTQ_CKPT}"); fi
if [ -n "${SAVE_PATH:-}" ];     then EXTRA_ARGS+=(--save-path "${SAVE_PATH}");         fi

echo "=== launch config ==="
echo "  master   : ${MASTER_IP}:${MASTER_PORT}"
echo "  hostfile : ${HOSTFILE}"
echo "  config   : ${CONFIG}"
echo "  extras   : ${EXTRA_ARGS[*]:-}"
echo "  log_dir  : ${LOG_DIR}"
echo

# ---- forward env vars to remote workers ----
# DeepSpeed's PDSH/SSH launcher does NOT inherit the master's environment
# on remote nodes; SSH-spawned worker shells start clean. The supported
# way to forward env vars is via ``.deepspeed_env`` (one ``KEY=VAL`` per
# line, in CWD or ``$HOME``). DeepSpeed reads the file and re-exports
# every entry into each worker process.
#
# We forward all NCCL/IB tuning we set above so RDMA actually works on
# every worker, plus PYTHONPATH (so ``angelslim`` resolves).
FORWARD_VARS=(
    PYTHONPATH
    PYTORCH_CUDA_ALLOC_CONF
    PYTORCH_ALLOC_CONF
    NCCL_IB_GID_INDEX
    NCCL_IB_SL
    NCCL_CHECK_DISABLE
    NCCL_P2P_DISABLE
    NCCL_IB_DISABLE
    NCCL_LL_THRESHOLD
    NCCL_IB_CUDA_SUPPORT
    NCCL_SOCKET_IFNAME
    UCX_NET_DEVICES
    NCCL_IB_HCA
    NCCL_NET_GDR_LEVEL
    NCCL_IB_QPS_PER_CONNECTION
    NCCL_IB_TC
    NCCL_PXN_DISABLE
    NCCL_IB_TIMEOUT
    NCCL_SOCKET_TIMEOUT
)
DS_ENV_FILE="${PROJECT_DIR}/.deepspeed_env"
: > "${DS_ENV_FILE}"
for var in "${FORWARD_VARS[@]}"; do
    if [ -n "${!var:-}" ]; then
        printf '%s=%s\n' "${var}" "${!var}" >> "${DS_ENV_FILE}"
    fi
done
echo "=== .deepspeed_env (forwarded to every worker) ==="
cat "${DS_ENV_FILE}"
echo

# ---- launch ----
deepspeed \
    --hostfile="${HOSTFILE}" \
    --master_addr="${MASTER_IP}" \
    --master_port="${MASTER_PORT}" \
    tools/run.py \
    -c "${CONFIG}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/main.log"
