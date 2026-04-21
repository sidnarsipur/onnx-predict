#!/usr/bin/env python3
"""Count ONNX nodes and simple aggregate metrics for ONNX models."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort
from onnx import checker, version_converter

from onnx_node_metrics import FIXED_COLUMNS, TEXT_COLUMNS, collect_model_row

TARGET_OPSET = 21
CSV_MODEL_COLUMN = "model"
OPTIMIZATION_LEVELS = {
    "disable_all": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ONNX models to opset 21, store node counts plus simple "
            "aggregate metrics in a CSV, and delete the processed model files."
        )
    )
    parser.add_argument("models", nargs="+", help="One or more .onnx model paths.")
    parser.add_argument("--csv", default="node_counts.csv", help="CSV file to update.")
    parser.add_argument(
        "--optimization-level",
        action="append",
        dest="optimization_levels",
        choices=sorted(OPTIMIZATION_LEVELS),
        help="Create rows for the given ONNX Runtime optimization level instead of the original model. Repeat this flag to request multiple levels.",
    )
    return parser.parse_args()


def normalize_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_model(model_path: Path) -> onnx.ModelProto:
    try:
        return onnx.load(model_path)
    except Exception as exc:  # pragma: no cover
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
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to convert '{source_name}' to opset {TARGET_OPSET}: {exc}"
            ) from exc

    try:
        checker.check_model(converted_model)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Converted model '{source_name}' is not valid ONNX: {exc}") from exc

    return converted_model


def ordered_columns(columns: list[str]) -> list[str]:
    fixed = [column for column in FIXED_COLUMNS if column in columns]
    counts = sorted(
        [column for column in columns if column not in fixed],
        key=str.casefold,
    )
    return fixed + counts


def read_existing_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not csv_path.exists():
        return [], []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return [], []
        if fieldnames[0] != CSV_MODEL_COLUMN:
            raise RuntimeError(
                f"Expected first column in '{csv_path}' to be '{CSV_MODEL_COLUMN}', found '{fieldnames[0]}'."
            )
        columns = ordered_columns(fieldnames[1:])
        rows = [normalize_row(row, columns) for row in reader]
        return columns, rows


def normalize_row(row: dict[str, str], columns: list[str]) -> dict[str, str]:
    normalized = {CSV_MODEL_COLUMN: row.get(CSV_MODEL_COLUMN, "")}
    for column in columns:
        default = "" if column in TEXT_COLUMNS else "0"
        normalized[column] = row.get(column, default) or default
    return normalized


def update_csv(csv_path: Path, new_rows: list[dict[str, str]]) -> None:
    existing_columns, existing_rows = read_existing_csv(csv_path)
    merged_columns = ordered_columns(
        list(
            dict.fromkeys(
                existing_columns
                + [column for row in new_rows for column in row if column != CSV_MODEL_COLUMN]
            )
        )
    )

    rows_by_model = {
        row[CSV_MODEL_COLUMN]: normalize_row(row, merged_columns) for row in existing_rows
    }
    for row in new_rows:
        normalized = normalize_row(row, merged_columns)
        rows_by_model[normalized[CSV_MODEL_COLUMN]] = normalized

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[CSV_MODEL_COLUMN] + merged_columns)
        writer.writeheader()
        for model_name in rows_by_model:
            writer.writerow(rows_by_model[model_name])


def optimize_model(
    converted_model_path: Path,
    optimization_level_name: str,
    source_name: str,
) -> onnx.ModelProto:
    optimized_model_path = converted_model_path.with_name(
        f"{converted_model_path.stem}_{optimization_level_name}.onnx"
    )
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = OPTIMIZATION_LEVELS[optimization_level_name]
    session_options.optimized_model_filepath = str(optimized_model_path)

    try:
        ort.InferenceSession(
            str(converted_model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to optimize '{source_name}' with {optimization_level_name}: {exc}"
        ) from exc

    if not optimized_model_path.exists():  # pragma: no cover
        raise RuntimeError(
            f"ONNX Runtime did not write the {optimization_level_name} optimized model for '{source_name}'."
        )

    try:
        return load_model(optimized_model_path)
    finally:
        optimized_model_path.unlink(missing_ok=True)


def optimized_model_name(source_name: str, optimization_level_name: str) -> str:
    source_path = Path(source_name)
    return f"{source_path.stem}_{optimization_level_name}{source_path.suffix}"


def process_model(model_path: Path, optimization_levels: list[str]) -> list[dict[str, str]]:
    source_model = load_model(model_path)
    converted_model = convert_to_target_opset(source_model, model_path.name)
    if not optimization_levels:
        return [collect_model_row(converted_model, model_path.name)]

    rows: list[dict[str, str]] = []
    unique_levels = list(dict.fromkeys(optimization_levels))
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        converted_model_path = temp_dir / model_path.name
        onnx.save(converted_model, converted_model_path)

        for optimization_level_name in unique_levels:
            optimized_model = optimize_model(
                converted_model_path,
                optimization_level_name,
                model_path.name,
            )
            rows.append(
                collect_model_row(
                    optimized_model,
                    optimized_model_name(model_path.name, optimization_level_name),
                )
            )

    return rows


def delete_models(model_paths: list[Path]) -> None:
    for model_path in dict.fromkeys(model_paths):
        model_path.unlink()


def main() -> int:
    args = parse_args()
    csv_path = normalize_path(args.csv)
    rows: list[dict[str, str]] = []
    processed_model_paths: list[Path] = []

    for model_arg in args.models:
        model_path = normalize_path(model_arg)
        if not model_path.exists():
            print(f"Model not found: {model_arg}", file=sys.stderr)
            return 1
        if model_path.suffix.lower() != ".onnx":
            print(f"Expected an .onnx file, got: {model_path}", file=sys.stderr)
            return 1
        rows.extend(process_model(model_path, args.optimization_levels or []))
        processed_model_paths.append(model_path)

    update_csv(csv_path, rows)
    delete_models(processed_model_paths)
    print(f"Updated {csv_path} with {len(rows)} row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
