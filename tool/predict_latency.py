#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from onnx import checker, version_converter
from pandas.errors import PerformanceWarning
from sklearn.exceptions import InconsistentVersionWarning


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)
ort.set_default_logger_severity(3)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_COLLECTION_DIR = REPO_ROOT / "data_collection"
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))

from onnx_node_metrics import collect_model_row 


TARGET_OPSET = 21
BYTES_PER_MB = 1024 * 1024
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "model.pt"
DEFAULT_CALIBRATION_DATA = REPO_ROOT / "training" / "test_set.csv"
OPTIMIZATION_LEVELS = {
    "disable_all": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "enable_all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
CPU_PROVIDER_MAP = {
    "intel": 0.0,
    "amd": 1.0,
}


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class GatedFiLMLatencyMLP(nn.Module):
    def __init__(self, n_model_features: int, n_hw_features: int):
        super().__init__()
        emb_dim = 192

        self.model_in = nn.Sequential(
            nn.Linear(n_model_features, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(384, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )
        self.model_blocks = nn.Sequential(
            ResidualBlock(emb_dim, dropout=0.10),
            ResidualBlock(emb_dim, dropout=0.08),
            ResidualBlock(emb_dim, dropout=0.05),
        )
        self.hw_tower = nn.Sequential(
            nn.Linear(n_hw_features, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(96, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )
        self.hw_film = nn.Linear(emb_dim, emb_dim * 2)
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 5, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.12),
            ResidualBlock(384, dropout=0.08),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(192, 1),
        )

    def forward(self, model_x: torch.Tensor, hw_x: torch.Tensor) -> torch.Tensor:
        model_emb = self.model_in(model_x)
        model_emb = self.model_blocks(model_emb)
        hw_emb = self.hw_tower(hw_x)

        gamma, beta = self.hw_film(hw_emb).chunk(2, dim=1)
        film_model = model_emb * (1 + gamma) + beta
        gate = self.gate(torch.cat([model_emb, hw_emb], dim=1))
        conditioned_model = gate * film_model + (1 - gate) * model_emb

        interaction = conditioned_model * hw_emb
        diff = torch.abs(conditioned_model - hw_emb)
        ratio_proxy = conditioned_model / (torch.abs(hw_emb) + 1.0)
        fused = torch.cat([conditioned_model, hw_emb, interaction, diff, ratio_proxy], dim=1)

        return self.fusion(fused).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict latency for disable_all/basic/extended variants of an ONNX model."
    )
    parser.add_argument("onnx_model", help="Path to a local .onnx model.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to model.pt generated from 13.ipynb.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(OPTIMIZATION_LEVELS),
        help="Variant to predict. Repeat to select multiple. Defaults to all variants.",
    )
    parser.add_argument("--l1d-cache-kb", type=float)
    parser.add_argument("--l1i-cache-kb", type=float)
    parser.add_argument("--l2-cache-kb", type=float)
    parser.add_argument("--base-clock-mhz", type=float)
    parser.add_argument("--num-cores", type=float)
    parser.add_argument("--memory-bandwith-gbs", type=float)
    parser.add_argument("--cpu-provider", choices=sorted(CPU_PROVIDER_MAP))
    parser.add_argument(
        "--calibration-data",
        default=str(DEFAULT_CALIBRATION_DATA),
        help="Held-out CSV used for group-based confidence. Defaults to training/test_set.csv.",
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Do not compute group-based confidence.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(model_path: Path) -> onnx.ModelProto:
    try:
        return onnx.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load ONNX model '{model_path}': {exc}") from exc


def convert_to_target_opset(model: onnx.ModelProto, source_name: str) -> onnx.ModelProto:
    main_opset = next(
        (opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")),
        None,
    )
    if main_opset == TARGET_OPSET:
        converted_model = model
    else:
        try:
            converted_model = version_converter.convert_version(model, TARGET_OPSET)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to convert '{source_name}' to opset {TARGET_OPSET}: {exc}"
            ) from exc

    try:
        checker.check_model(converted_model)
    except Exception as exc:
        raise RuntimeError(f"Converted model '{source_name}' is not valid ONNX: {exc}") from exc

    return converted_model


def optimize_model(
    converted_model_path: Path,
    variant_name: str,
    source_name: str,
) -> onnx.ModelProto:
    optimized_model_path = converted_model_path.with_name(
        f"{converted_model_path.stem}_{variant_name}.onnx"
    )
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = OPTIMIZATION_LEVELS[variant_name]
    session_options.optimized_model_filepath = str(optimized_model_path)

    try:
        ort.InferenceSession(
            str(converted_model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to create {variant_name} variant for '{source_name}': {exc}") from exc

    if not optimized_model_path.exists():
        raise RuntimeError(f"ONNX Runtime did not write the {variant_name} variant.")

    try:
        return load_model(optimized_model_path)
    finally:
        optimized_model_path.unlink(missing_ok=True)


def collect_variant_rows(model_path: Path, variants: list[str]) -> list[dict[str, str]]:
    source_model = load_model(model_path)
    converted_model = convert_to_target_opset(source_model, model_path.name)

    rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        converted_model_path = temp_dir / model_path.name
        onnx.save(converted_model, converted_model_path)

        for variant_name in variants:
            optimized_model = optimize_model(converted_model_path, variant_name, model_path.name)
            variant_model_name = f"{model_path.stem}_{variant_name}{model_path.suffix}"
            rows.append(collect_model_row(optimized_model, variant_model_name))

    return rows


def prompt_float(name: str, display: str, value: float | None) -> float:
    if value is not None:
        return float(value)
    while True:
        raw = input(f"{display}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print(f"Enter a number for {name}.")


def prompt_provider(value: str | None) -> str:
    if value:
        return value.lower()
    while True:
        raw = input("cpu_provider [amd/intel]: ").strip().lower()
        if raw in CPU_PROVIDER_MAP:
            return raw
        print("Enter 'amd' or 'intel'.")


def hardware_values(args: argparse.Namespace) -> dict[str, float]:
    provider = prompt_provider(args.cpu_provider)
    values = {
        "l1d_cache_kb": prompt_float("l1d_cache_kb", "L1d cache KB", args.l1d_cache_kb),
        "l1i_cache_kb": prompt_float("l1i_cache_kb", "L1i cache KB", args.l1i_cache_kb),
        "l2_cache_kb": prompt_float("l2_cache_kb", "L2 cache KB", args.l2_cache_kb),
        "base_clock_mhz": prompt_float("base_clock_mhz", "Base clock MHz", args.base_clock_mhz),
        "num_cores": prompt_float("num_cores", "Number of cores/threads", args.num_cores),
        "memory_bandwith_gbs": prompt_float(
            "memory_bandwith_gbs",
            "Memory bandwidth GB/s",
            args.memory_bandwith_gbs,
        ),
        "cpu_provider": provider,
        "cpu_provider_binary": CPU_PROVIDER_MAP[provider],
    }
    return values


def row_to_features(row: dict[str, str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for key, value in row.items():
        if key == "model" or key.endswith("_dimensions") or key.endswith("_dtypes"):
            continue
        output_key = f"{key[:-6]}_mb" if key.endswith("_bytes") else key
        try:
            numeric = float(value or 0)
        except ValueError:
            numeric = 0.0
        if key.endswith("_bytes"):
            numeric = numeric / BYTES_PER_MB
        features[output_key] = numeric

    for column in ["elementwise_mb", "reduction_mb", "normalization_mb", "movement_mb"]:
        if column in features:
            features[column] = round(features[column], 0)
    return features


def predict_row(
    row: dict[str, str],
    hardware: dict[str, float],
    checkpoint: dict[str, Any],
    model: GatedFiLMLatencyMLP,
) -> dict[str, float | str]:
    model_feature_cols = checkpoint["model_feature_cols"]
    hardware_feature_cols = checkpoint["hardware_feature_cols"]

    extracted_features = row_to_features(row)
    model_values = np.array(
        [[float(extracted_features.get(column, 0.0)) for column in model_feature_cols]],
        dtype=np.float32,
    )
    model_values = np.log1p(np.clip(model_values, a_min=0.0, a_max=None)).astype(np.float32)

    hw_values = np.array(
        [[float(hardware[column]) for column in hardware_feature_cols]],
        dtype=np.float32,
    )

    model_values = checkpoint["model_scaler"].transform(model_values).astype(np.float32)
    hw_values = checkpoint["hw_scaler"].transform(hw_values).astype(np.float32)

    with torch.no_grad():
        pred_log = model(
            torch.from_numpy(model_values),
            torch.from_numpy(hw_values),
        ).item()

    return {
        "variant": row["model"],
        "pred_log_average_ms": pred_log,
        "pred_average_ms": float(np.exp(pred_log)),
    }


def optimization_variant(model_name: str) -> str:
    stem = Path(model_name).stem
    for variant_name in sorted(OPTIMIZATION_LEVELS, key=len, reverse=True):
        suffix = f"_{variant_name}"
        if stem.endswith(suffix):
            return variant_name
    return "default"


def run13_preprocess(df: pd.DataFrame, checkpoint: dict[str, Any]) -> pd.DataFrame:
    df = df.dropna(subset=["average_ms"]).copy()
    df["cv"] = df["stddev_ms"] / df["average_ms"]
    df = df[df["cv"] <= 0.1].copy()
    df["log_average_ms"] = np.log(df["average_ms"])

    for col in ["elementwise_mb", "reduction_mb", "normalization_mb", "movement_mb"]:
        if col in df.columns:
            df[col] = df[col].round(0)

    for col in checkpoint["model_feature_cols"]:
        df[col] = np.log1p(df[col].clip(lower=0))

    df["cpu_provider_binary"] = df["cpu_provider"].map(CPU_PROVIDER_MAP)
    if df["cpu_provider_binary"].isna().any():
        bad_values = df.loc[df["cpu_provider_binary"].isna(), "cpu_provider"].unique()
        raise ValueError(f"Unknown cpu_provider values in calibration data: {bad_values}")

    df["optimization_variant"] = df["model"].map(optimization_variant)
    return df


def predict_dataframe(
    df: pd.DataFrame,
    checkpoint: dict[str, Any],
    model: GatedFiLMLatencyMLP,
    batch_size: int = 4096,
) -> np.ndarray:
    model_feature_cols = checkpoint["model_feature_cols"]
    hardware_feature_cols = checkpoint["hardware_feature_cols"]

    x_model = df[model_feature_cols].astype("float32").to_numpy()
    x_hw = df[hardware_feature_cols].astype("float32").to_numpy()

    x_model = checkpoint["model_scaler"].transform(x_model).astype(np.float32)
    x_hw = checkpoint["hw_scaler"].transform(x_hw).astype(np.float32)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            end = start + batch_size
            pred = model(
                torch.from_numpy(x_model[start:end]),
                torch.from_numpy(x_hw[start:end]),
            )
            preds.append(pred.numpy())
    return np.concatenate(preds)


def calibration_summary(g: pd.DataFrame) -> pd.Series:
    rel_err = g["relative_error"].to_numpy()
    return pd.Series(
        {
            "count": len(g),
            "within_10pct": float(np.mean(rel_err <= 0.10)),
            "within_25pct": float(np.mean(rel_err <= 0.25)),
            "median_percent_error": float(np.median(rel_err) * 100),
            "p90_percent_error": float(np.percentile(rel_err, 90) * 100),
            "p95_percent_error": float(np.percentile(rel_err, 95) * 100),
        }
    )


def build_confidence_calibration(
    path: Path,
    checkpoint: dict[str, Any],
    model: GatedFiLMLatencyMLP,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df = run13_preprocess(df, checkpoint)
    pred_log = predict_dataframe(df, checkpoint, model)
    true_ms = np.exp(df["log_average_ms"].to_numpy())
    pred_ms = np.exp(pred_log)
    df["relative_error"] = np.abs(pred_ms - true_ms) / true_ms

    exact = df.groupby(["cpu_provider", "num_cores", "optimization_variant"]).apply(
        calibration_summary,
        include_groups=False,
    )
    hardware = df.groupby(["cpu_provider", "num_cores"]).apply(
        calibration_summary,
        include_groups=False,
    )
    provider = df.groupby(["cpu_provider"]).apply(
        calibration_summary,
        include_groups=False,
    )
    overall = calibration_summary(df)

    reference = hardware[hardware["count"] >= 30] if not hardware.empty else hardware
    if reference.empty:
        reference = hardware

    return {
        "exact": exact,
        "hardware": hardware,
        "provider": provider,
        "overall": overall,
        "within10_high": float(reference["within_10pct"].quantile(0.67)),
        "within10_low": float(reference["within_10pct"].quantile(0.33)),
        "p90_high": float(reference["p90_percent_error"].quantile(0.67)),
        "p90_low": float(reference["p90_percent_error"].quantile(0.33)),
    }


def lookup_group_stats(
    calibration: dict[str, Any] | None,
    hardware: dict[str, float],
    variant_name: str,
) -> tuple[pd.Series | None, str]:
    if calibration is None:
        return None, "none"

    provider = str(hardware["cpu_provider"])
    cores = float(hardware["num_cores"])
    variant = optimization_variant(variant_name)

    exact = calibration["exact"]
    exact_key = (provider, cores, variant)
    if exact_key in exact.index:
        return exact.loc[exact_key], "provider+cores+variant"

    hardware_groups = calibration["hardware"]
    hardware_key = (provider, cores)
    if hardware_key in hardware_groups.index:
        return hardware_groups.loc[hardware_key], "provider+cores"

    provider_groups = calibration["provider"]
    if provider in provider_groups.index:
        return provider_groups.loc[provider], "provider"

    return calibration["overall"], "overall"


def confidence_label(stats: pd.Series | None, calibration: dict[str, Any] | None) -> tuple[str, str]:
    if stats is None or calibration is None:
        return "unknown", "no calibration data"

    count = int(stats["count"])
    within_10 = float(stats["within_10pct"])
    p90 = float(stats["p90_percent_error"])

    if count < 30:
        return "low", f"only {count} similar calibration samples"
    if within_10 >= calibration["within10_high"] and p90 <= calibration["p90_high"]:
        return "high", f"historical within_10={within_10:.1%}, p90_error={p90:.1f}%"
    if within_10 < calibration["within10_low"] or p90 > calibration["p90_high"]:
        return "low", f"historical within_10={within_10:.1%}, p90_error={p90:.1f}%"
    return "medium", f"historical within_10={within_10:.1%}, p90_error={p90:.1f}%"


def add_confidence(
    result: dict[str, float | str],
    hardware: dict[str, float],
    calibration: dict[str, Any] | None,
) -> dict[str, float | str]:
    stats, group = lookup_group_stats(calibration, hardware, str(result["variant"]))
    label, reason = confidence_label(stats, calibration)
    result["confidence"] = label
    result["confidence_group"] = group
    result["confidence_reason"] = reason
    if stats is not None:
        result["confidence_samples"] = int(stats["count"])
    return result


def print_table(results: list[dict[str, float | str]]) -> None:
    has_confidence = any("confidence" in result for result in results)
    if has_confidence:
        print(f"{'variant':<60} {'average_ms':>12} {'confidence':>11}")
        print("-" * 86)
    else:
        print(f"{'variant':<60} {'average_ms':>12}")
        print("-" * 73)
    for result in results:
        if has_confidence:
            print(
                f"{str(result['variant']):<60} "
                f"{float(result['pred_average_ms']):>12.6f} "
                f"{str(result.get('confidence', 'unknown')):>11}"
            )
        else:
            print(f"{str(result['variant']):<60} {float(result['pred_average_ms']):>12.6f}")


def main() -> int:
    args = parse_args()
    model_path = Path(args.onnx_model).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()

    if not model_path.exists():
        print(f"ONNX model not found: {model_path}", file=sys.stderr)
        return 1
    if model_path.suffix.lower() != ".onnx":
        print(f"Expected an .onnx file, got: {model_path}", file=sys.stderr)
        return 1
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    checkpoint = load_checkpoint(checkpoint_path)
    hardware = hardware_values(args)
    variants = list(dict.fromkeys(args.variant or ["disable_all", "basic", "extended", "enable_all"]))
    rows = collect_variant_rows(model_path, variants)

    latency_model = GatedFiLMLatencyMLP(
        n_model_features=checkpoint["n_model_features"],
        n_hw_features=checkpoint["n_hw_features"],
    )
    latency_model.load_state_dict(checkpoint["model_state_dict"])
    latency_model.eval()

    results = [predict_row(row, hardware, checkpoint, latency_model) for row in rows]
    if not args.no_confidence:
        calibration = build_confidence_calibration(
            Path(args.calibration_data).expanduser(),
            checkpoint,
            latency_model,
        )
        results = [add_confidence(result, hardware, calibration) for result in results]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
