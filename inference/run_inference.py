#!/usr/bin/env python3
"""Run ONNX Runtime inference for model variants listed in node_counts.csv."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import tempfile
import time
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import HfApi, hf_hub_url
from onnx import checker, version_converter

TARGET_OPSET = 21
MODEL_COLUMN = "model"
DEFAULT_REPO_PREFIX = "onnxmodelzoo"
VARIANT_SUFFIXES = {
    "_disable_all.onnx": ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
    "_basic.onnx": ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
    "_extended.onnx": ("extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
    "_enable_all.onnx": ("enable_all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    "_all.onnx": ("enable_all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
}
CPU_PROVIDER = "CPUExecutionProvider"
SAMPLE_COLUMNS = [f"sample_{index}_ms" for index in range(1, 11)]
OUTPUT_COLUMNS = [
    "model",
    "base_model",
    "repo_id",
    "repo_file",
    "optimization_variant",
    "provider",
    "warmup_runs",
    "sample_runs",
    "average_ms",
    "stddev_ms",
    "min_ms",
    "max_ms",
    *SAMPLE_COLUMNS,
    "status",
    "error",
]


@dataclass(frozen=True)
class ModelEntry:
    model_name: str
    base_model_name: str
    repo_id: str
    variant_name: str
    optimization_level: ort.GraphOptimizationLevel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ONNX model zoo models and collect ONNX Runtime inference timings."
    )
    parser.add_argument(
        "--node-counts",
        default="../node_counts.csv",
        help="Path to node_counts.csv.",
    )
    parser.add_argument(
        "--output",
        default="inference_results.csv",
        help="CSV file for inference timing results.",
    )
    parser.add_argument(
        "--progress-log",
        default="inference_results.log",
        help="Progress log used to resume interrupted runs.",
    )
    parser.add_argument(
        "--hf-cache",
        default="",
        help="Temporary download directory. Defaults to a temporary directory.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up inference runs.")
    parser.add_argument("--samples", type=int, default=10, help="Timed inference samples.")
    parser.add_argument(
        "--repo-prefix",
        default=DEFAULT_REPO_PREFIX,
        help="Hugging Face org/user that owns the model repos.",
    )
    return parser.parse_args()


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_log(log_path: Path, status: str, model_name: str, message: str = "") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{timestamp()}\t{status}\t{model_name}"
    if message:
        line = f"{line}\t{one_line(message)}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def one_line(message: str) -> str:
    return " ".join(str(message).replace("\r", "\n").split())


def read_completed_models(progress_log: Path) -> set[str]:
    if not progress_log.exists():
        return set()

    completed: set[str] = set()
    for line in progress_log.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t", 3)
        if len(parts) >= 3 and parts[1] in {"DONE", "FAILED"}:
            completed.add(parts[2])
    return completed


def parse_model_name(model_name: str, repo_prefix: str) -> ModelEntry:
    for suffix, (variant_name, optimization_level) in VARIANT_SUFFIXES.items():
        if model_name.endswith(suffix):
            base_model_name = f"{model_name[: -len(suffix)]}.onnx"
            repo_id = f"{repo_prefix}/{Path(base_model_name).stem}"
            return ModelEntry(
                model_name=model_name,
                base_model_name=base_model_name,
                repo_id=repo_id,
                variant_name=variant_name,
                optimization_level=optimization_level,
            )
    else:
        raise ValueError(f"Skipping non-optimized model row: {model_name}")


def read_model_entries(node_counts_path: Path, repo_prefix: str) -> list[ModelEntry]:
    with node_counts_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or MODEL_COLUMN not in reader.fieldnames:
            raise RuntimeError(f"Expected '{MODEL_COLUMN}' column in {node_counts_path}")
        entries: list[ModelEntry] = []
        for row in reader:
            model_name = row.get(MODEL_COLUMN)
            if not model_name:
                continue
            try:
                entries.append(parse_model_name(model_name, repo_prefix))
            except ValueError:
                continue
        return entries


def group_entries(entries: list[ModelEntry]) -> OrderedDict[str, list[ModelEntry]]:
    grouped: OrderedDict[str, list[ModelEntry]] = OrderedDict()
    for entry in entries:
        grouped.setdefault(entry.base_model_name, []).append(entry)
    return grouped


def find_repo_file(api: HfApi, entry: ModelEntry) -> str:
    repo_files = api.list_repo_files(entry.repo_id, repo_type="model")
    if entry.base_model_name in repo_files:
        return entry.base_model_name

    basename_matches = [repo_file for repo_file in repo_files if Path(repo_file).name == entry.base_model_name]
    if basename_matches:
        return basename_matches[0]

    onnx_files = [repo_file for repo_file in repo_files if repo_file.lower().endswith(".onnx")]
    if len(onnx_files) == 1:
        return onnx_files[0]

    raise RuntimeError(f"Could not find {entry.base_model_name} in {entry.repo_id}")


def download_model(entry: ModelEntry, repo_file: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    destination_path = download_dir / entry.base_model_name
    temp_path = destination_path.with_suffix(f"{destination_path.suffix}.part")
    url = hf_hub_url(repo_id=entry.repo_id, filename=repo_file, repo_type="model")

    temp_path.unlink(missing_ok=True)
    destination_path.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url) as response, temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temp_path.replace(destination_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        destination_path.unlink(missing_ok=True)
        raise

    return destination_path


def delete_downloaded_model(model_path: Path) -> None:
    model_path.unlink(missing_ok=True)


def convert_to_target_opset(model_path: Path, output_path: Path) -> Path:
    model = onnx.load(model_path)
    main_opset = next(
        (opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")),
        None,
    )
    if main_opset != TARGET_OPSET:
        model = version_converter.convert_version(model, TARGET_OPSET)
    checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


def choose_providers() -> list[str]:
    available = ort.get_available_providers()
    if CPU_PROVIDER not in available:
        raise RuntimeError(f"{CPU_PROVIDER} is not available in this ONNX Runtime installation")
    return [CPU_PROVIDER]


def create_session(
    model_path: Path,
    optimization_level: ort.GraphOptimizationLevel,
    providers: list[str],
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = optimization_level
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers,
    )


def clean_dim(dim: object) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    return 1


def input_shape(input_meta: ort.NodeArg) -> tuple[int, ...]:
    return tuple(clean_dim(dim) for dim in input_meta.shape)


def random_tensor(input_meta: ort.NodeArg, rng: np.random.Generator) -> np.ndarray:
    shape = input_shape(input_meta)
    input_type = input_meta.type

    if input_type == "tensor(float16)":
        return rng.standard_normal(shape).astype(np.float16)
    if input_type == "tensor(float)":
        return rng.standard_normal(shape).astype(np.float32)
    if input_type == "tensor(double)":
        return rng.standard_normal(shape).astype(np.float64)
    if input_type == "tensor(bool)":
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    if input_type == "tensor(string)":
        return np.full(shape, "input", dtype=object)

    dtype = {
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(int16)": np.int16,
        "tensor(uint16)": np.uint16,
        "tensor(int32)": np.int32,
        "tensor(uint32)": np.uint32,
        "tensor(int64)": np.int64,
        "tensor(uint64)": np.uint64,
    }.get(input_type)
    if dtype is not None:
        return rng.integers(0, 10, size=shape).astype(dtype)

    raise RuntimeError(f"Unsupported input type {input_type} for input {input_meta.name}")


def make_inputs(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {input_meta.name: random_tensor(input_meta, rng) for input_meta in session.get_inputs()}


def time_inference(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    warmup_runs: int,
    sample_runs: int,
) -> list[float]:
    for _ in range(warmup_runs):
        session.run(None, inputs)

    samples: list[float] = []
    for _ in range(sample_runs):
        start = time.perf_counter_ns()
        session.run(None, inputs)
        end = time.perf_counter_ns()
        samples.append((end - start) / 1_000_000)
    return samples


def format_float(value: float) -> str:
    return f"{value:.6f}"


def success_row(
    entry: ModelEntry,
    repo_file: str,
    provider: str,
    warmup_runs: int,
    samples: list[float],
) -> dict[str, str]:
    row = base_output_row(entry, repo_file, provider, warmup_runs, len(samples), "DONE", "")
    row["average_ms"] = format_float(statistics.fmean(samples))
    row["stddev_ms"] = format_float(statistics.stdev(samples) if len(samples) > 1 else 0.0)
    row["min_ms"] = format_float(min(samples))
    row["max_ms"] = format_float(max(samples))
    for column, sample in zip(SAMPLE_COLUMNS, samples):
        row[column] = format_float(sample)
    return row


def failure_row(entry: ModelEntry, repo_file: str, error: str) -> dict[str, str]:
    return base_output_row(entry, repo_file, "", 0, 0, "FAILED", one_line(error))


def base_output_row(
    entry: ModelEntry,
    repo_file: str,
    provider: str,
    warmup_runs: int,
    sample_runs: int,
    status: str,
    error: str,
) -> dict[str, str]:
    row = {column: "" for column in OUTPUT_COLUMNS}
    row.update(
        {
            "model": entry.model_name,
            "base_model": entry.base_model_name,
            "repo_id": entry.repo_id,
            "repo_file": repo_file,
            "optimization_variant": entry.variant_name,
            "provider": provider,
            "warmup_runs": str(warmup_runs),
            "sample_runs": str(sample_runs),
            "status": status,
            "error": error,
        }
    )
    return row


def append_output_row(output_path: Path, row: dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def process_entry(
    entry: ModelEntry,
    source_model_path: Path,
    repo_file: str,
    args: argparse.Namespace,
    providers: list[str],
    output_path: Path,
    progress_log: Path,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir_name:
        converted_path = Path(temp_dir_name) / entry.base_model_name
        convert_to_target_opset(source_model_path, converted_path)
        session = create_session(converted_path, entry.optimization_level, providers)
        inputs = make_inputs(session)
        samples = time_inference(session, inputs, args.warmup, args.samples)

    row = success_row(
        entry,
        repo_file,
        ",".join(session.get_providers()),
        args.warmup,
        samples,
    )
    append_output_row(output_path, row)
    append_log(progress_log, "DONE", entry.model_name)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    args = parse_args()
    node_counts_path = resolve_path(args.node_counts, script_dir)
    output_path = resolve_path(args.output, script_dir)
    progress_log = resolve_path(args.progress_log, script_dir)
    download_dir = resolve_path(args.hf_cache, script_dir) if args.hf_cache else Path(tempfile.mkdtemp())

    os.environ.setdefault("HF_HOME", str(download_dir / ".hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(download_dir / ".hf_cache"))

    entries = read_model_entries(node_counts_path, args.repo_prefix)
    completed = read_completed_models(progress_log)
    grouped_entries = group_entries([entry for entry in entries if entry.model_name not in completed])
    api = HfApi()
    providers = choose_providers()
    print(f"Using ONNX Runtime providers: {', '.join(providers)}", flush=True)

    for group in grouped_entries.values():
        repo_file = ""
        source_model_path: Path | None = None
        try:
            repo_file = find_repo_file(api, group[0])
            source_model_path = download_model(group[0], repo_file, download_dir)
        except Exception as exc:
            for entry in group:
                row = failure_row(entry, repo_file, str(exc))
                append_output_row(output_path, row)
                append_log(progress_log, "FAILED", entry.model_name, str(exc))
            continue

        for entry in group:
            print(f"Running {entry.model_name}...", flush=True)
            try:
                process_entry(
                    entry,
                    source_model_path,
                    repo_file,
                    args,
                    providers,
                    output_path,
                    progress_log,
                )
            except Exception as exc:
                row = failure_row(entry, repo_file, str(exc))
                append_output_row(output_path, row)
                append_log(progress_log, "FAILED", entry.model_name, str(exc))
                print(f"Failed {entry.model_name}: {one_line(str(exc))}", flush=True)

        delete_downloaded_model(source_model_path)

    if not args.hf_cache:
        shutil.rmtree(download_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
