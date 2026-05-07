#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import checker, version_converter
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
ort.set_default_logger_severity(3)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data_collection"))
from onnx_node_metrics import collect_model_row 

TARGET_OPSET = 21
BYTES_PER_MB = 1024 * 1024
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "model.pt"
DEFAULT_CHECKPOINT_URL = "https://huggingface.co/sidnarsipur/onnx-predict/resolve/main/model.pt"

OPTIMIZATION_LEVELS = {
    "disable_all": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "enable_all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
CPU_PROVIDER_MAP = {"intel": 0.0, "amd": 1.0}


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class SimpleLatencyMLP(nn.Module):
    def __init__(self, n_model_features: int, n_hw_features: int):
        super().__init__()
        model_dim = 192
        hw_dim = 64

        self.model_tower = nn.Sequential(
            nn.Linear(n_model_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            ResidualBlock(model_dim, dropout=0.08),
        )
        self.hw_tower = nn.Sequential(
            nn.Linear(n_hw_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, hw_dim),
            nn.LayerNorm(hw_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(model_dim + hw_dim + model_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 1),
        )

    def forward(self, model_x: torch.Tensor, hw_x: torch.Tensor) -> torch.Tensor:
        model_emb = self.model_tower(model_x)
        hw_emb = self.hw_tower(hw_x)
        hw_interaction = torch.nn.functional.pad(hw_emb, (0, model_emb.shape[1] - hw_emb.shape[1]))
        return self.fusion(torch.cat([model_emb, hw_emb, model_emb * hw_interaction], dim=1)).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ONNX Runtime CPU latency for an ONNX model.")
    parser.add_argument("onnx_model", help="Path to a local .onnx model.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Run 17 checkpoint path. The default downloads from Hugging Face if missing or stale.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(OPTIMIZATION_LEVELS),
        help="Optimization variant to predict. Repeat for multiple. Defaults to all variants.",
    )
    parser.add_argument("--cpu-provider", choices=sorted(CPU_PROVIDER_MAP))
    parser.add_argument("--l1d-cache-kb", type=float)
    parser.add_argument("--l1i-cache-kb", type=float)
    parser.add_argument("--l2-cache-kb", type=float)
    parser.add_argument("--base-clock-mhz", type=float)
    parser.add_argument("--num-cores", type=float)
    parser.add_argument("--memory-bandwith-gbs", type=float)
    parser.add_argument("--memory-mb", type=float)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    return parser.parse_args()


def download_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".download")
    print(f"Downloading Run 17 checkpoint to {path}...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(DEFAULT_CHECKPOINT_URL, temp_path)
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_run17_checkpoint(checkpoint: dict[str, Any]) -> bool:
    state = checkpoint.get("model_state_dict", {})
    hardware_cols = checkpoint.get("hardware_feature_cols", [])
    return (
        "model_tower.0.weight" in state
        and "hw_tower.0.weight" in state
        and "memory_mb" in hardware_cols
    )


def load_checkpoint(path: Path, is_default: bool) -> dict[str, Any]:
    if not path.exists():
        if not is_default:
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        download_checkpoint(path)

    checkpoint = torch_load(path)
    if is_run17_checkpoint(checkpoint):
        return checkpoint
    if not is_default:
        raise ValueError(f"Checkpoint is not a Run 17 SimpleLatencyMLP checkpoint: {path}")

    print(f"{path} is not a Run 17 checkpoint; downloading the current checkpoint.", file=sys.stderr)
    download_checkpoint(path)
    checkpoint = torch_load(path)
    if not is_run17_checkpoint(checkpoint):
        raise ValueError(f"Downloaded checkpoint is not compatible with Run 17: {DEFAULT_CHECKPOINT_URL}")
    return checkpoint


def prompt_float(label: str, value: float | None) -> float:
    if value is not None:
        return float(value)
    while True:
        try:
            return float(input(f"{label}: ").strip())
        except ValueError:
            print(f"Enter a number for {label}.")


def prompt_provider(value: str | None) -> str:
    if value:
        return value
    while True:
        provider = input("cpu_provider [amd/intel]: ").strip().lower()
        if provider in CPU_PROVIDER_MAP:
            return provider
        print("Enter 'amd' or 'intel'.")


def hardware_features(args: argparse.Namespace) -> dict[str, float]:
    provider = prompt_provider(args.cpu_provider)
    return {
        "l1d_cache_kb": prompt_float("L1d cache KB", args.l1d_cache_kb),
        "l1i_cache_kb": prompt_float("L1i cache KB", args.l1i_cache_kb),
        "l2_cache_kb": prompt_float("L2 cache KB", args.l2_cache_kb),
        "base_clock_mhz": prompt_float("Base clock MHz", args.base_clock_mhz),
        "num_cores": prompt_float("Number of cores/threads", args.num_cores),
        "memory_bandwith_gbs": prompt_float("Memory bandwidth GB/s", args.memory_bandwith_gbs),
        "memory_mb": prompt_float("Memory MB", args.memory_mb),
        "cpu_provider_binary": CPU_PROVIDER_MAP[provider],
    }


def converted_model(path: Path) -> onnx.ModelProto:
    model = onnx.load(path)
    main_opset = next((opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")), None)
    if main_opset != TARGET_OPSET:
        model = version_converter.convert_version(model, TARGET_OPSET)
    checker.check_model(model)
    return model


def optimized_row(model_path: Path, variant: str, temp_dir: Path) -> dict[str, str]:
    input_path = temp_dir / model_path.name
    output_path = temp_dir / f"{model_path.stem}_{variant}.onnx"
    options = ort.SessionOptions()
    options.graph_optimization_level = OPTIMIZATION_LEVELS[variant]
    options.optimized_model_filepath = str(output_path)
    ort.InferenceSession(str(input_path), sess_options=options, providers=["CPUExecutionProvider"])
    model = onnx.load(output_path)
    output_path.unlink(missing_ok=True)
    return collect_model_row(model, output_path.name)


def collect_rows(model_path: Path, variants: list[str]) -> list[dict[str, str]]:
    with tempfile.TemporaryDirectory() as temp_name:
        temp_dir = Path(temp_name)
        onnx.save(converted_model(model_path), temp_dir / model_path.name)
        return [optimized_row(model_path, variant, temp_dir) for variant in variants]


def numeric_features(row: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in row.items():
        if key == "model" or key.endswith("_dimensions") or key.endswith("_dtypes"):
            continue
        try:
            number = float(value or 0)
        except ValueError:
            number = 0.0
        values[f"{key[:-6]}_mb" if key.endswith("_bytes") else key] = (
            number / BYTES_PER_MB if key.endswith("_bytes") else number
        )
    for key in ("elementwise_mb", "reduction_mb", "normalization_mb", "movement_mb"):
        values[key] = round(values.get(key, 0.0), 0)
    return values


def predict(row: dict[str, str], hardware: dict[str, float], checkpoint: dict[str, Any], model: nn.Module) -> dict[str, Any]:
    model_features = numeric_features(row)
    x_model = np.array(
        [[model_features.get(col, 0.0) for col in checkpoint["model_feature_cols"]]],
        dtype=np.float32,
    )
    x_hw = np.array(
        [[hardware[col] for col in checkpoint["hardware_feature_cols"]]],
        dtype=np.float32,
    )
    x_model = np.log1p(np.clip(x_model, 0.0, None)).astype(np.float32)
    x_model = checkpoint["model_scaler"].transform(x_model).astype(np.float32)
    x_hw = checkpoint["hw_scaler"].transform(x_hw).astype(np.float32)

    with torch.no_grad():
        pred_log_ms = model(torch.from_numpy(x_model), torch.from_numpy(x_hw)).item()
    return {"variant": row["model"], "average_ms": float(np.exp(pred_log_ms))}


def print_table(results: list[dict[str, Any]]) -> None:
    print(f"{'variant':<60} {'average_ms':>12}")
    print("-" * 73)
    for result in results:
        print(f"{result['variant']:<60} {result['average_ms']:>12.6f}")


def main() -> int:
    args = parse_args()
    model_path = Path(args.onnx_model).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()

    if not model_path.exists() or model_path.suffix.lower() != ".onnx":
        print(f"Expected a local .onnx file: {model_path}", file=sys.stderr)
        return 1

    try:
        checkpoint = load_checkpoint(checkpoint_path, checkpoint_path.resolve() == DEFAULT_CHECKPOINT.resolve())
        hardware = hardware_features(args)
        rows = collect_rows(model_path, list(dict.fromkeys(args.variant or OPTIMIZATION_LEVELS)))
        model = SimpleLatencyMLP(checkpoint["n_model_features"], checkpoint["n_hw_features"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        results = [predict(row, hardware, checkpoint, model) for row in rows]
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
