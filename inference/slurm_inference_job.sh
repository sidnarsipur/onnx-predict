#!/usr/bin/env bash
#SBATCH --job-name=onnx-inference
#SBATCH --output=logs/onnx_inference_%j.out
#SBATCH --error=logs/onnx_inference_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=16G

set -uo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME:-onnx-inference}"
NODE_COUNTS_CSV="${NODE_COUNTS_CSV:-${SCRIPT_DIR}/node_counts.csv}"
RUN_ID="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="${SCRIPT_DIR}/artifacts/run_${RUN_ID}"
LOG_DIR="${ARTIFACT_DIR}/logs"
HARDWARE_DIR="${ARTIFACT_DIR}/hardware"
DATA_DIR="${ARTIFACT_DIR}/data"

mkdir -p "${LOG_DIR}" "${HARDWARE_DIR}" "${DATA_DIR}" "${SCRIPT_DIR}/.hf_home" "${SCRIPT_DIR}/.hf_cache"

collect_artifacts() {
  local exit_code=$?
  set +e

  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "job_id=${SLURM_JOB_ID:-}"
    echo "job_name=${SLURM_JOB_NAME:-}"
    echo "exit_code=${exit_code}"
    echo "project_dir=${PROJECT_DIR}"
    echo "script_dir=${SCRIPT_DIR}"
    echo "conda_env=${ENV_NAME}"
    echo "python_bin=$(command -v python 2>/dev/null || true)"
    echo "node_counts_csv=${NODE_COUNTS_CSV}"
    echo "artifact_dir=${ARTIFACT_DIR}"
  } > "${ARTIFACT_DIR}/run_summary.txt"

  uname -a > "${HARDWARE_DIR}/uname.txt" 2>&1 || true
  cat /proc/cpuinfo > "${HARDWARE_DIR}/cpuinfo.txt" 2>&1 || true
  cat /proc/meminfo > "${HARDWARE_DIR}/meminfo.txt" 2>&1 || true
  lscpu > "${HARDWARE_DIR}/lscpu.txt" 2>&1 || true
  df -h > "${HARDWARE_DIR}/disk_usage.txt" 2>&1 || true
  env | sort > "${LOG_DIR}/environment.txt" 2>&1 || true

  if command -v python >/dev/null 2>&1; then
    python --version > "${LOG_DIR}/python_version.txt" 2>&1 || true
    python -m pip freeze > "${LOG_DIR}/pip_freeze.txt" 2>&1 || true
  fi

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job "${SLURM_JOB_ID}" > "${LOG_DIR}/scontrol_show_job.txt" 2>&1 || true
    sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize > "${LOG_DIR}/sacct.txt" 2>&1 || true
    seff "${SLURM_JOB_ID}" > "${LOG_DIR}/seff.txt" 2>&1 || true
    cp "${SLURM_SUBMIT_DIR:-${PROJECT_DIR}}/onnx_inference_${SLURM_JOB_ID}.out" "${LOG_DIR}/" 2>/dev/null || true
    cp "${SLURM_SUBMIT_DIR:-${PROJECT_DIR}}/onnx_inference_${SLURM_JOB_ID}.err" "${LOG_DIR}/" 2>/dev/null || true
  fi

  cp "${SCRIPT_DIR}/inference_results.csv" "${DATA_DIR}/" 2>/dev/null || true
  cp "${SCRIPT_DIR}/inference_progress.log" "${DATA_DIR}/" 2>/dev/null || true
  cp "${SCRIPT_DIR}/inference_errors.log" "${DATA_DIR}/" 2>/dev/null || true
  cp "${NODE_COUNTS_CSV}" "${DATA_DIR}/" 2>/dev/null || true

  echo "Artifacts collected in ${ARTIFACT_DIR}"
  exit "${exit_code}"
}

trap collect_artifacts EXIT

exec > >(tee -a "${LOG_DIR}/job_stdout.log") 2> >(tee -a "${LOG_DIR}/job_stderr.log" >&2)

echo "Starting ONNX inference job at $(date --iso-8601=seconds)"
echo "Artifacts: ${ARTIFACT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Load your conda module before submitting this job." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [[ ! -f "${NODE_COUNTS_CSV}" ]]; then
  echo "node_counts.csv not found: ${NODE_COUNTS_CSV}" >&2
  echo "Set NODE_COUNTS_CSV=/path/to/node_counts.csv when submitting if needed." >&2
  exit 1
fi

export HF_HOME="${SCRIPT_DIR}/.hf_home"
export HF_HUB_CACHE="${SCRIPT_DIR}/.hf_cache"

python "${SCRIPT_DIR}/run_inference.py" \
  --node-counts "${NODE_COUNTS_CSV}" \
  --output "${SCRIPT_DIR}/inference_results.csv" \
  --progress-log "${SCRIPT_DIR}/inference_progress.log" \
  --error-log "${SCRIPT_DIR}/inference_errors.log" \
  --hf-cache "${HF_HUB_CACHE}"

echo "Finished ONNX inference job at $(date --iso-8601=seconds)"
