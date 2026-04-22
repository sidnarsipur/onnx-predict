#!/usr/bin/env bash
#SBATCH --job-name=onnx-inference
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=16G

set -uo pipefail

SCRIPT_DIR="$(pwd)"
if [[ ! -f "${SCRIPT_DIR}/run_inference.py" ]]; then
  echo "Could not find run_inference.py in the current directory." >&2
  echo "Run this script from the inference directory." >&2
  exit 1
fi

RUN_LABEL="${1:-}"
if [[ -z "${RUN_LABEL}" ]]; then
  echo "Usage: sbatch slurm_inference_job.sh <run_id>" >&2
  echo "Example: sbatch -p interactive -t 12:00:00 slurm_inference_job.sh 1" >&2
  exit 1
fi

ENV_NAME="${ENV_NAME:-onnx-inference}"
JOB_ID="${SLURM_JOB_ID:-local}"
ORT_INTRA_OP_NUM_THREADS="${ORT_INTRA_OP_NUM_THREADS:-0}"
RUN_DIR="${SCRIPT_DIR}/logs/${RUN_LABEL}"

NODE_COUNTS_CSV="${RUN_DIR}/node_counts.csv"
RESULTS_CSV="${RUN_DIR}/inference_results.csv"
RESULTS_LOG="${RUN_DIR}/inference_results.log"
ARTIFACTS_TXT="${RUN_DIR}/artifacts.txt"
ENV_TXT="${RUN_DIR}/env.txt"

mkdir -p "${RUN_DIR}"
rm -rf "${RUN_DIR}/download_tmp"

PREVIOUS_INFERENCE_SECONDS="$(
  awk -F= '$1 == "total_inference_seconds" { value = $2 } END { print value }' "${ENV_TXT}" 2>/dev/null || true
)"
if [[ ! "${PREVIOUS_INFERENCE_SECONDS}" =~ ^[0-9]+$ ]]; then
  PREVIOUS_INFERENCE_SECONDS=0
fi
CURRENT_INFERENCE_SECONDS=0
INFERENCE_STARTED=0
INFERENCE_START_EPOCH=0

if [[ ! -f "${NODE_COUNTS_CSV}" ]]; then
  if [[ ! -f "${SCRIPT_DIR}/node_counts.csv" ]]; then
    echo "node_counts.csv not found in ${SCRIPT_DIR}." >&2
    exit 1
  fi
  cp "${SCRIPT_DIR}/node_counts.csv" "${NODE_COUNTS_CSV}"
fi

exec > "${RUN_DIR}/${JOB_ID}.out" 2> "${RUN_DIR}/${JOB_ID}.err"

append_section() {
  local title="$1"
  shift
  {
    echo
    echo "===== ${title} ====="
    "$@"
  } >> "${ARTIFACTS_TXT}" 2>&1 || true
}

format_seconds() {
  local total_seconds="$1"
  printf "%02d:%02d:%02d" \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

collect_artifacts() {
  local exit_code=$?
  set +e
  local now_epoch
  local total_inference_seconds

  if [[ "${INFERENCE_STARTED}" == "1" ]]; then
    now_epoch="$(date +%s)"
    CURRENT_INFERENCE_SECONDS=$((now_epoch - INFERENCE_START_EPOCH))
    if (( CURRENT_INFERENCE_SECONDS < 0 )); then
      CURRENT_INFERENCE_SECONDS=0
    fi
  fi
  total_inference_seconds=$((PREVIOUS_INFERENCE_SECONDS + CURRENT_INFERENCE_SECONDS))

  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "job_id=${JOB_ID}"
    echo "run_label=${RUN_LABEL}"
    echo "exit_code=${exit_code}"
    echo "memory_from_slurm=${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-${SLURM_MEM_PER_GPU:-}}}"
    echo "slurm_mem_per_node=${SLURM_MEM_PER_NODE:-}"
    echo "slurm_mem_per_cpu=${SLURM_MEM_PER_CPU:-}"
    echo "slurm_mem_per_gpu=${SLURM_MEM_PER_GPU:-}"
    echo "cores_requested=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-}}"
    echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
    echo "slurm_cpus_on_node=${SLURM_CPUS_ON_NODE:-}"
    echo "intra_threads=${ORT_INTRA_OP_NUM_THREADS}"
    echo "previous_inference_seconds=${PREVIOUS_INFERENCE_SECONDS}"
    echo "current_inference_seconds=${CURRENT_INFERENCE_SECONDS}"
    echo "total_inference_seconds=${total_inference_seconds}"
    echo "total_inference_time=$(format_seconds "${total_inference_seconds}")"
  } > "${ENV_TXT}"

  : > "${ARTIFACTS_TXT}"
  append_section "run_context" bash -c 'echo "hostname=$(hostname)"; echo "job_id='"${JOB_ID}"'"; echo "job_name=${SLURM_JOB_NAME:-}"; echo "run_label='"${RUN_LABEL}"'"; echo "exit_code='"${exit_code}"'"; echo "script_dir='"${SCRIPT_DIR}"'"; echo "run_dir='"${RUN_DIR}"'"; echo "conda_env='"${ENV_NAME}"'"; echo "python_bin=$(command -v python 2>/dev/null || true)"; echo "node_counts_csv='"${NODE_COUNTS_CSV}"'"; echo "results_csv='"${RESULTS_CSV}"'"; echo "results_log='"${RESULTS_LOG}"'"'
  append_section "uname" uname -a
  append_section "lscpu" lscpu
  append_section "cpuinfo" cat /proc/cpuinfo
  append_section "meminfo" cat /proc/meminfo
  append_section "disk_usage" df -h
  append_section "python_version" python --version
  append_section "pip_freeze" python -m pip freeze

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    append_section "scontrol_show_job" scontrol show job "${SLURM_JOB_ID}"
    append_section "sacct" sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
    append_section "seff" seff "${SLURM_JOB_ID}"
  fi

  rm -rf "${RUN_DIR}/download_tmp"

  echo "Artifacts written to ${ARTIFACTS_TXT}"
  exit "${exit_code}"
}

trap collect_artifacts EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "Starting ONNX inference job at $(date --iso-8601=seconds)"
echo "Run label: ${RUN_LABEL}"
echo "Run directory: ${RUN_DIR}"
if [[ -f "${RESULTS_LOG}" ]]; then
  echo "Resuming from ${RESULTS_LOG}"
  echo "Existing completed/failed entries: $(wc -l < "${RESULTS_LOG}")"
else
  echo "Starting a fresh run log at ${RESULTS_LOG}"
fi

PYTHON_BIN="python"
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
elif [[ -x "${ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN="${ENV_NAME}/bin/python"
else
  echo "conda was not found on PATH and ENV_NAME is not a usable env path: ${ENV_NAME}" >&2
  echo "Set ENV_NAME=/path/to/conda/env or load the conda module before running this script." >&2
  exit 1
fi

export HF_HOME="${RUN_DIR}/.hf_home"
export HF_HUB_CACHE="${RUN_DIR}/.hf_cache"
export ORT_INTRA_OP_NUM_THREADS

INFERENCE_STARTED=1
INFERENCE_START_EPOCH="$(date +%s)"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_inference.py" \
  --node-counts "${NODE_COUNTS_CSV}" \
  --output "${RESULTS_CSV}" \
  --progress-log "${RESULTS_LOG}" \
  --hf-cache "${RUN_DIR}/download_tmp" \
  --intra-op-num-threads "${ORT_INTRA_OP_NUM_THREADS}"
INFERENCE_STATUS=$?
CURRENT_INFERENCE_SECONDS=$(($(date +%s) - INFERENCE_START_EPOCH))
if (( CURRENT_INFERENCE_SECONDS < 0 )); then
  CURRENT_INFERENCE_SECONDS=0
fi
INFERENCE_STARTED=0

echo "Finished ONNX inference job at $(date --iso-8601=seconds)"
exit "${INFERENCE_STATUS}"
