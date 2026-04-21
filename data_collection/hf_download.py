#!/usr/bin/env python3
"""Download ONNX models from Hugging Face Hub and run count_onnx_nodes.py."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List, Sequence

from huggingface_hub import HfApi, hf_hub_url

SCRIPT_DIR = Path(__file__).resolve().parent
COUNT_SCRIPT = SCRIPT_DIR / "count_onnx_nodes.py"
MODELS_DIR = SCRIPT_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download ONNX models from one or more Hugging Face model repos into "
            "models/ and run count_onnx_nodes.py on them."
        )
    )
    parser.add_argument(
        "repos",
        nargs="+",
        help="One or more Hugging Face model repo IDs, for example Xenova/tiny-random-Wav2Vec2ForCTC-ONNX.",
    )
    parser.add_argument(
        "--csv",
        default="node_counts.csv",
        help="CSV output path passed to count_onnx_nodes.py. Defaults to node_counts.csv.",
    )
    parser.add_argument(
        "--optimization-level",
        action="append",
        dest="optimization_levels",
        choices=["disable_all", "basic", "extended"],
        help="Run count_onnx_nodes.py on ONNX Runtime optimized variants instead of the original model. Repeat this flag to request multiple levels.",
    )
    return parser.parse_args()


def normalize_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def is_onnx_file(repo_file: str) -> bool:
    return repo_file.lower().endswith(".onnx")


def downloaded_model_name(repo_id: str, repo_file: str) -> str:
    return Path(repo_file).name


def list_onnx_files(api: HfApi, repo_id: str) -> List[str]:
    repo_files = api.list_repo_files(repo_id, repo_type="model")
    onnx_files = [repo_file for repo_file in repo_files if is_onnx_file(repo_file)]
    if not onnx_files:
        raise RuntimeError(f"Repo '{repo_id}' does not contain any .onnx files.")
    return onnx_files


def download_model(repo_id: str, repo_file: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination_path = MODELS_DIR / downloaded_model_name(repo_id, repo_file)
    temp_path = destination_path.with_name(f"{destination_path.name}.part")

    if destination_path.exists():
        print(f"Using existing {destination_path.name}.", flush=True)
        return destination_path

    temp_path.unlink(missing_ok=True)
    try:
        download_url = hf_hub_url(
            repo_id=repo_id,
            filename=repo_file,
            repo_type="model",
        )
        with urllib.request.urlopen(download_url) as response, temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temp_path.replace(destination_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        destination_path.unlink(missing_ok=True)
        raise

    print(f"Saved {destination_path.name}.", flush=True)
    return destination_path


def run_count_script(
    model_paths: Sequence[Path],
    csv_path: Path,
    optimization_levels: Sequence[str],
) -> None:
    if not COUNT_SCRIPT.exists():
        raise RuntimeError(f"Expected count script at '{COUNT_SCRIPT}', but it was not found.")

    command = [sys.executable, str(COUNT_SCRIPT), "--csv", str(csv_path)]
    if optimization_levels:
        for optimization_level in optimization_levels:
            command.extend(["--optimization-level", optimization_level])
    command.extend(str(model_path) for model_path in model_paths)
    subprocess.run(command, check=True, cwd=SCRIPT_DIR)


def main() -> int:
    args = parse_args()
    api = HfApi()
    csv_path = normalize_path(args.csv)

    processed_models = 0
    for repo_id in args.repos:
        onnx_files = list_onnx_files(api, repo_id)
        for repo_file in onnx_files:
            model_path = download_model(repo_id, repo_file)
            try:
                run_count_script([model_path], csv_path, args.optimization_levels or [])
            except Exception:
                model_path.unlink(missing_ok=True)
                raise
            processed_models += 1

    print(f"Processed {processed_models} model(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
