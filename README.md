# ONNX-Predict: Predict the latency of ONNX Models using static features

This was a course project for [ECE 208: The Art of Machine Learning](https://hajim.rochester.edu/ece/sites/zduan/teaching/ece408/index.html) at the University of Rochester.

The project trains machine learning models to predict ONNX Runtime inference latency for models on x86 Linux machines.

# Structure

- data_collection: Downloads ONNX models and extracts static graph features.
- inference: Runs ONNX Runtime latency benchmarks locally or through Slurm jobs.
- training: Builds datasets and trains latency prediction models.
- tool: Predicts ONNX Runtime latency for a local ONNX model using the trained checkpoint.

# Hugging Face Resources

- Dataset: https://huggingface.co/datasets/sidnarsipur/onnx-inference
- Trained model: https://huggingface.co/sidnarsipur/onnx-predict

# Requirements

- Python 3.12 or newer
- Python packages:
  - `numpy`
  - `torch`
  - `scikit-learn`
  - `onnx`
  - `onnxruntime`


# Running the Tool

Run the latency predictor from the repo root:

```bash
./tool/predict_latency.py path/to/model.onnx \
  --cpu-provider amd \
  --l1d-cache-kb 32 \
  --l1i-cache-kb 32 \
  --l2-cache-kb 512 \
  --base-clock-mhz 2450 \
  --num-cores 16 \
  --memory-bandwidth-gbs 205 \
  --memory-mb 65536
```

The tool creates ONNX Runtime optimization variants for the input model and predicts `average_ms` for each variant:

- `disable_all`
- `basic`
- `extended`
- `enable_all`

To predict only one or more specific variants, repeat `--variant`:

```bash
./tool/predict_latency.py path/to/model.onnx \
  --variant basic \
  --variant extended \
  --cpu-provider intel \
  --l1d-cache-kb 48 \
  --l1i-cache-kb 32 \
  --l2-cache-kb 1280 \
  --base-clock-mhz 2200 \
  --num-cores 8 \
  --memory-bandwidth-gbs 102 \
  --memory-mb 32768
```

If a hardware argument is omitted, the script will prompt for it interactively.

Use `--json` for machine-readable output:

```bash
./tool/predict_latency.py path/to/model.onnx \
  --json \
  --cpu-provider amd \
  --l1d-cache-kb 32 \
  --l1i-cache-kb 32 \
  --l2-cache-kb 512 \
  --base-clock-mhz 2450 \
  --num-cores 16 \
  --memory-bandwidth-gbs 205 \
  --memory-mb 65536
```
