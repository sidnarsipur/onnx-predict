#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-onnx-inference}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
ONNX_VERSION="${ONNX_VERSION:-1.16.1}"
ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.19.0}"
HF_HOME_DIR="${SCRIPT_DIR}/.hf_home"
HF_CACHE_DIR="${SCRIPT_DIR}/.hf_cache"
LOG_FILE="${SCRIPT_DIR}/logs/setup_inference_env.log"
LOCK_DIR="${SCRIPT_DIR}/.setup_lock"

mkdir -p "${SCRIPT_DIR}/artifacts" "${SCRIPT_DIR}/logs" "${HF_HOME_DIR}" "${HF_CACHE_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Another setup run appears to be active: ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Load your conda module first, then rerun this script." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Using existing conda environment: ${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install --upgrade --only-binary=:all: \
  "numpy==${NUMPY_VERSION}" \
  "onnx==${ONNX_VERSION}" \
  "onnxruntime==${ONNXRUNTIME_VERSION}" \
  huggingface_hub

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/onnx_inference_env.sh" <<EOF
export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_CACHE_DIR}"
EOF

export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_CACHE_DIR}"

echo "Hugging Face cache:"
echo "  HF_HOME=${HF_HOME}"
echo "  HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "NumPy version: ${NUMPY_VERSION}"
echo "ONNX version: ${ONNX_VERSION}"
echo "ONNX Runtime version: ${ONNXRUNTIME_VERSION}"

python - <<PY
import numpy
import onnx
import onnxruntime as ort

expected = {
    "numpy": "${NUMPY_VERSION}",
    "onnx": "${ONNX_VERSION}",
    "onnxruntime": "${ONNXRUNTIME_VERSION}",
}
actual = {
    "numpy": numpy.__version__,
    "onnx": onnx.__version__,
    "onnxruntime": ort.__version__,
}
for name, expected_version in expected.items():
    actual_version = actual[name]
    print(f"{name} {actual_version}")
    if actual_version != expected_version:
        raise SystemExit(f"Expected {name}=={expected_version}, got {actual_version}")
PY

if command -v hf >/dev/null 2>&1 && hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face is already logged in."
elif command -v hf >/dev/null 2>&1; then
  hf auth login
elif huggingface-cli whoami >/dev/null 2>&1; then
  echo "Hugging Face is already logged in."
else
  huggingface-cli login
fi

echo "Environment ready. Activate it with:"
echo "  conda activate ${ENV_NAME}"
