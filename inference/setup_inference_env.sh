#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-onnx-inference}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
ONNX_VERSION="${ONNX_VERSION:-1.16.1}"
ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.20.0}"
HUGGINGFACE_HUB_VERSION="${HUGGINGFACE_HUB_VERSION:-0.30.2}"
LOG_FILE="${SCRIPT_DIR}/logs/setup_inference_env.log"
LOCK_DIR="${SCRIPT_DIR}/.setup_lock"

mkdir -p "${SCRIPT_DIR}/logs"
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

if [[ "${ENV_NAME}" == */* ]]; then
  CONDA_ENV_ARGS=(-p "${ENV_NAME}")
  if [[ -d "${ENV_NAME}" ]]; then
    echo "Using existing conda environment: ${ENV_NAME}"
  else
    conda create -y -p "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
else
  CONDA_ENV_ARGS=(-n "${ENV_NAME}")
  if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Using existing conda environment: ${ENV_NAME}"
  else
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
fi

conda run "${CONDA_ENV_ARGS[@]}" python -m pip install --upgrade pip
conda run "${CONDA_ENV_ARGS[@]}" python -m pip install --upgrade --only-binary=:all: \
  "numpy==${NUMPY_VERSION}" \
  "onnx==${ONNX_VERSION}" \
  "onnxruntime==${ONNXRUNTIME_VERSION}" \
  certifi \
  charset-normalizer \
  filelock \
  fsspec \
  idna \
  packaging \
  pyyaml \
  requests \
  tqdm \
  typing_extensions \
  urllib3 \
  "huggingface_hub[cli]==${HUGGINGFACE_HUB_VERSION}"

echo "NumPy version: ${NUMPY_VERSION}"
echo "ONNX version: ${ONNX_VERSION}"
echo "ONNX Runtime version: ${ONNXRUNTIME_VERSION}"
echo "Hugging Face Hub version: ${HUGGINGFACE_HUB_VERSION}"

conda run "${CONDA_ENV_ARGS[@]}" python - <<PY
import certifi
import huggingface_hub
import huggingface_hub.hf_api
import numpy
import onnx
import onnxruntime as ort

expected = {
    "numpy": "${NUMPY_VERSION}",
    "onnx": "${ONNX_VERSION}",
    "onnxruntime": "${ONNXRUNTIME_VERSION}",
    "huggingface_hub": "${HUGGINGFACE_HUB_VERSION}",
}
actual = {
    "numpy": numpy.__version__,
    "onnx": onnx.__version__,
    "onnxruntime": ort.__version__,
    "huggingface_hub": huggingface_hub.__version__,
}
for name, expected_version in expected.items():
    actual_version = actual[name]
    print(f"{name} {actual_version}")
    if actual_version != expected_version:
        raise SystemExit(f"Expected {name}=={expected_version}, got {actual_version}")
print(f"certifi {certifi.where()}")
PY

conda run "${CONDA_ENV_ARGS[@]}" bash -c 'command -v hf || command -v huggingface-cli' >/dev/null

if conda run "${CONDA_ENV_ARGS[@]}" hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face is already logged in."
elif conda run "${CONDA_ENV_ARGS[@]}" hf auth login; then
  :
elif conda run "${CONDA_ENV_ARGS[@]}" huggingface-cli whoami >/dev/null 2>&1; then
  echo "Hugging Face is already logged in."
else
  conda run "${CONDA_ENV_ARGS[@]}" huggingface-cli login
fi

echo "Environment ready. Activate it with:"
echo "  conda activate ${ENV_NAME}"
