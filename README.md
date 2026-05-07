# ONNX-Predict: Predict the latency of ONNX Models using static features

This was a course project for [ECE 208: The Art of Machine Learning](https://hajim.rochester.edu/ece/sites/zduan/teaching/ece408/index.html) at the University of Rochester.

The project involves training a series of machine learning model to predict the latency of ONNX Models on x86 Linux machine using the ONNX Runtime CPU Provider.

# Structure

- data_collection: code to download ONNX models from the ONNX Model Zoo and collect static features
- inference: run slurm jobs or python script to collect inference data
- training: scripts that were used to train models
- tool: python script that performs inference and predicts latency given an ONNX Model

# Requirements

- Python 3.12 or newer
- A local ONNX model file
- The trained latency model checkpoint. The tool downloads `tool/model.pt` from Hugging Face on first run if it is missing.
- Python packages:
  - `numpy`
  - `pandas`
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
  --memory-bandwidth-gbs 205
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
  --memory-bandwidth-gbs 102
```

If a hardware argument is omitted, the script will prompt for it interactively.

By default, the tool also reports a simple confidence label based on similar rows in the held-out training data. Disable that extra lookup with:

```bash
./tool/predict_latency.py path/to/model.onnx \
  --no-confidence \
  --cpu-provider amd \
  --l1d-cache-kb 32 \
  --l1i-cache-kb 32 \
  --l2-cache-kb 512 \
  --base-clock-mhz 2450 \
  --num-cores 16 \
  --memory-bandwidth-gbs 205
```
