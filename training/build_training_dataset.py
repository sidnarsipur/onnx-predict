#!/usr/bin/env python3
"""Build a training dataset by joining node counts with inference run results."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, OrderedDict
from pathlib import Path


RUN_IDS = [str(run_id) for run_id in range(1, 18) if run_id != 14]
HARDWARE_COLUMNS = [
    "num_cores",
    "memory_mb",
    "l1d_cache_kb",
    "l1i_cache_kb",
    "l2_cache_kb",
    "base_clock_mhz",
    "memory_bandwith_gbs",
    "cpu_provider",
    "machine_type",
    "platform",
]
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB_DECIMAL = 1_000_000_000
DROP_INFERENCE_COLUMNS = {
    "base_model",
    "repo_id",
    "optimization_variant",
    "provider",
    "warmup_runs",
    "sample_runs",
    "status",
    "error",
}

# Decimal bytes/sec. Values are derived from CPU model, memory generation, memory
# transfer rate, channel count, and socket count observed in artifacts.txt.
PER_SOCKET_MEMORY_BANDWIDTH = {
    "INTEL(R) XEON(R) PLATINUM 8568Y+": 8 * 5600 * 1_000_000 * 8,
    "AMD EPYC 7B13": 8 * 3200 * 1_000_000 * 8,
    "INTEL(R) XEON(R) CPU @ 2.80GHZ": 6 * 2933 * 1_000_000 * 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join inference/node_counts.csv with inference/logs/*/inference_results.csv."
    )
    parser.add_argument(
        "--node-counts",
        default="inference/node_counts.csv",
        help="Path to the node-count/static-feature CSV.",
    )
    parser.add_argument(
        "--logs-dir",
        default="inference/logs",
        help="Directory containing numbered inference run folders.",
    )
    parser.add_argument(
        "--output",
        default="training/full_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--preview-output",
        default="training/training_dataset_preview.csv",
        help="Small preview CSV path for editors that cannot open the full dataset.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=0,
        help="Number of data rows to write to the preview CSV. Use 0 to disable.",
    )
    return parser.parse_args()


def parse_size_to_bytes(value: str) -> int | None:
    match = re.search(r"([0-9.]+)\s*([KMGT]?i?B|kB)?", value, re.IGNORECASE)
    if not match:
        return None

    number = float(match.group(1))
    unit = (match.group(2) or "B").lower()
    multipliers = {
        "b": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": 1024**2,
        "mib": 1024**2,
        "gb": 1024**3,
        "gib": 1024**3,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    return int(number * multipliers[unit])


def parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_labeled_value(text: str, label: str) -> str:
    pattern = re.compile(rf"^{re.escape(label)}\s*:\s*(.+)$", re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def parse_first_int(text: str, label: str) -> int | None:
    value = parse_labeled_value(text, label)
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else None


def parse_memory_bytes(text: str) -> int | None:
    match = re.search(r"^MemTotal:\s*(\d+)\s+kB$", text, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def parse_cache_bytes(text: str, label: str) -> int | None:
    value = parse_labeled_value(text, label)
    return parse_size_to_bytes(value) if value else None


def parse_base_clock_mhz(text: str) -> float | None:
    values = [float(match) for match in re.findall(r"^cpu MHz\s*:\s*([0-9.]+)$", text, re.MULTILINE)]
    if not values:
        return None

    rounded = [round(value, 3) for value in values]
    return Counter(rounded).most_common(1)[0][0]


def derive_memory_bandwidth(cpu_model: str, sockets: int | None) -> int | None:
    bandwidth = PER_SOCKET_MEMORY_BANDWIDTH.get(cpu_model.upper())
    if bandwidth is None:
        return None
    return bandwidth * (sockets or 1)


def bytes_per_second_to_gbs(value: object) -> str:
    if value in {None, ""}:
        return ""
    return str(round(float(value) / BYTES_PER_GB_DECIMAL))


def bytes_to_mb(value: object) -> str:
    if value in {None, ""}:
        return ""
    return f"{float(value) / BYTES_PER_MB:.6f}"


def bytes_to_kb(value: object) -> str:
    if value in {None, ""}:
        return ""
    return f"{float(value) / BYTES_PER_KB:.6f}"


def output_feature_name(column: str) -> str:
    return f"{column[:-6]}_mb" if column.endswith("_bytes") else column


def output_feature_value(column: str, value: str) -> str:
    return bytes_to_mb(value) if column.endswith("_bytes") else value


def parse_hardware(run_dir: Path) -> dict[str, str]:
    env = parse_key_value_file(run_dir / "env.txt")
    artifacts = (run_dir / "artifacts.txt").read_text(encoding="utf-8")
    cpu_model = parse_labeled_value(artifacts, "Model name")
    sockets = parse_first_int(artifacts, "Socket(s)")
    hostname = env.get("hostname") or parse_key_value_file(run_dir / "artifacts.txt").get("hostname", "")
    cpu_model_upper = cpu_model.upper()

    if "AMD" in cpu_model_upper:
        cpu_provider = "amd"
        machine_type = "epyc"
    elif "PLATINUM" in cpu_model_upper:
        cpu_provider = "intel"
        machine_type = "xeon_plat"
    elif "XEON" in cpu_model_upper:
        cpu_provider = "intel"
        machine_type = "xeon"
    else:
        cpu_provider = ""
        machine_type = ""

    values: dict[str, object] = {
        "num_cores": env.get("intra_threads", ""),
        "memory_mb": str(int((parse_memory_bytes(artifacts) or 0) // BYTES_PER_MB)),
        "l1d_cache_kb": bytes_to_kb(parse_cache_bytes(artifacts, "L1d cache")),
        "l1i_cache_kb": bytes_to_kb(parse_cache_bytes(artifacts, "L1i cache")),
        "l2_cache_kb": bytes_to_kb(parse_cache_bytes(artifacts, "L2 cache")),
        "base_clock_mhz": parse_base_clock_mhz(artifacts),
        "memory_bandwith_gbs": bytes_per_second_to_gbs(derive_memory_bandwidth(cpu_model, sockets)),
        "cpu_provider": cpu_provider,
        "machine_type": machine_type,
        "platform": "bluehive" if hostname.startswith("bhdrb") else "gcloud",
    }
    return {key: "" if value is None else str(value) for key, value in values.items()}


def keep_inference_column(column: str) -> bool:
    return column != "model" and column not in DROP_INFERENCE_COLUMNS and not re.fullmatch(r"sample_\d+_ms", column)


def read_node_counts(path: Path) -> tuple[list[str], OrderedDict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "model" not in reader.fieldnames:
            raise RuntimeError(f"Expected a model column in {path}")
        rows = OrderedDict((row["model"], row) for row in reader if row.get("model"))
    return list(reader.fieldnames), rows


def read_inference_results(path: Path) -> tuple[list[str], OrderedDict[str, dict[str, str]], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "model" not in reader.fieldnames:
            raise RuntimeError(f"Expected a model column in {path}")
        rows: OrderedDict[str, dict[str, str]] = OrderedDict()
        duplicate_count = 0
        for row in reader:
            model = row.get("model")
            if not model:
                continue
            if model in rows:
                duplicate_count += 1
            rows[model] = row
    return list(reader.fieldnames), rows, duplicate_count


def main() -> None:
    args = parse_args()
    node_count_columns, node_count_rows = read_node_counts(Path(args.node_counts))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inference_columns: list[str] | None = None
    output_rows: list[dict[str, str]] = []
    total_duplicates = 0

    for run_id in RUN_IDS:
        run_dir = Path(args.logs_dir) / run_id
        results_path = run_dir / "inference_results.csv"
        if not results_path.exists():
            raise RuntimeError(f"Missing results file: {results_path}")

        current_inference_columns, inference_rows, duplicate_count = read_inference_results(results_path)
        total_duplicates += duplicate_count
        if inference_columns is None:
            inference_columns = current_inference_columns
        elif inference_columns != current_inference_columns:
            raise RuntimeError(f"Inference schema mismatch in {results_path}")

        hardware = parse_hardware(run_dir)
        for model, inference_row in inference_rows.items():
            if inference_row.get("status") != "DONE" or inference_row.get("error"):
                continue

            node_count_row = node_count_rows.get(model)
            if node_count_row is None:
                continue

            output_row = {"model": model}
            output_row.update(
                {
                    output_feature_name(column): output_feature_value(column, node_count_row[column])
                    for column in node_count_columns
                    if column != "model"
                }
            )
            output_row["run_id"] = run_id
            output_row.update(hardware)
            output_row.update({column: inference_row[column] for column in current_inference_columns if keep_inference_column(column)})
            output_rows.append(output_row)

    if inference_columns is None:
        raise RuntimeError("No inference results were loaded.")

    output_columns = (
        ["model"]
        + [output_feature_name(column) for column in node_count_columns if column != "model"]
        + ["run_id"]
        + HARDWARE_COLUMNS
        + [column for column in inference_columns if keep_inference_column(column)]
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(output_rows)

    if args.preview_rows > 0:
        preview_path = Path(args.preview_output)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        with preview_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_columns)
            writer.writeheader()
            writer.writerows(output_rows[: args.preview_rows])
        print(f"wrote {min(args.preview_rows, len(output_rows))} preview rows to {preview_path}")

    print(f"wrote {len(output_rows)} rows to {output_path}")
    print(f"deduplicated {total_duplicates} duplicate run/model rows")


if __name__ == "__main__":
    main()
