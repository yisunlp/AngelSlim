#!/usr/bin/env bash
#
# One-shot launcher: runs on the MASTER node only and SSH-spawns the
# training script on every other node listed in ``NODE_IP_LIST``.
#
# Prerequisites:
#   * All nodes share the same workspace via shared FS (apdcephfs).
#   * SSH between nodes works passwordlessly (it does in the taiji
#     container environment).
#   * ``NODE_IP_LIST`` is set (taiji auto-sets this).
#
# Usage (on master only):
#   bash scripts/qat/launch_qat_multinode.sh
#
# Environment overrides (forwarded to every node):
#   CONFIG          QAT yaml path
#   MODEL_PATH      overrides ``model.model_path``
#   FROM_PTQ_CKPT   overrides ``compression.QAT.from_ptq_ckpt``
#   SAVE_PATH       overrides ``global.save_path``
#   MASTER_PORT     default 29500
#   LOG_DIR         where per-node stdout logs get written (default: ./logs/multinode_$(date))
#   PROJECT_DIR     project root; default auto-detected from this script location
#
set -euo pipefail

if [ -z "${NODE_IP_LIST:-}" ]; then
    echo "ERROR: NODE_IP_LIST is not set."
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd "${SCRIPT_DIR}" && pwd)}
LOG_DIR=${LOG_DIR:-"${PROJECT_DIR}/logs/multinode_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${LOG_DIR}"

RUN_SCRIPT="${PROJECT_DIR}/run_qat_multinode.sh"
if [ ! -f "${RUN_SCRIPT}" ]; then
    echo "ERROR: ${RUN_SCRIPT} not found."
    exit 1
fi

IFS=',' read -r -a NODE_ENTRIES <<< "${NODE_IP_LIST}"
echo "Launching across ${#NODE_ENTRIES[@]} nodes, logs under ${LOG_DIR}"
for e in "${NODE_ENTRIES[@]}"; do
    echo "  - ${e}"
done

# Environment the remote shell needs (NODE_IP_LIST is inherited through
# the container image, but we forward the training overrides explicitly).
REMOTE_ENV=""
for var in CONFIG MODEL_PATH FROM_PTQ_CKPT SAVE_PATH MASTER_PORT NNODES NPROC_PER_NODE; do
    if [ -n "${!var:-}" ]; then
        REMOTE_ENV+="${var}=$(printf %q "${!var}") "
    fi
done

PIDS=()
for entry in "${NODE_ENTRIES[@]}"; do
    ip=${entry%%:*}
    LOG_FILE="${LOG_DIR}/node_${ip//./_}.log"
    echo "[spawn] ${ip} -> ${LOG_FILE}"
    ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR "${ip}" \
        "bash -lc 'cd ${PROJECT_DIR} && ${REMOTE_ENV} bash ${RUN_SCRIPT}'" \
        > "${LOG_FILE}" 2>&1 &
    PIDS+=($!)
done

echo
echo "All nodes spawned. Tail logs with:"
echo "  tail -f ${LOG_DIR}/node_*.log"
echo

# Wait for all remote sessions; exit with the first non-zero status.
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAIL=1
    fi
done
exit "${FAIL}"
