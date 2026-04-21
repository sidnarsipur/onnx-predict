#!/usr/bin/env python3
"""Run hf_download.py for every repo in the onnxmodelzoo org with resume support."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HF_DOWNLOAD_SCRIPT = SCRIPT_DIR / "hf_download.py"
DEFAULT_SOURCE_LOG_PATH = SCRIPT_DIR / "onnxmodelzoo_progress.log"
DEFAULT_LOG_PATH = SCRIPT_DIR / "onnxmodelzoo_opt_progress.log"
EXCLUDED_REPO_IDS = {"onnxmodelzoo/legacy_models"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read successful repos from onnxmodelzoo_progress.log, generate basic and "
            "extended optimized rows through hf_download.py, and record second-pass "
            "progress in a separate log file so interrupted runs can resume."
        )
    )
    parser.add_argument("--csv", default="node_counts.csv", help="CSV file to update.")
    parser.add_argument(
        "--source-log",
        default=str(DEFAULT_SOURCE_LOG_PATH),
        help="Progress log from the first pass. Only repos marked DONE are processed.",
    )
    parser.add_argument(
        "--log",
        default=str(DEFAULT_LOG_PATH),
        help="Second-pass progress log file. Each line records DONE or FAILED for one repo.",
    )
    return parser.parse_args()


def normalize_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def read_logged_repos(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    repos: set[str] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry:
            continue
        parts = entry.split("\t", 3)
        if len(parts) >= 3 and parts[1] in {"DONE", "FAILED"}:
            repos.add(parts[2])
    return repos


def read_successful_repos(source_log_path: Path) -> list[str]:
    if not source_log_path.exists():
        raise RuntimeError(f"Source progress log not found: {source_log_path}")

    repos: list[str] = []
    seen: set[str] = set()
    for line in source_log_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry:
            continue

        parts = entry.split("\t", 2)
        if parts[0] == "DONE" and len(parts) >= 2:
            repo_id = parts[1]
        elif parts[0] == "FAILED":
            continue
        else:
            repo_id = parts[0]

        if repo_id in EXCLUDED_REPO_IDS or repo_id in seen:
            continue

        repos.append(repo_id)
        seen.add(repo_id)

    if not repos:
        raise RuntimeError(f"No successful repos found in {source_log_path}")

    return repos


def append_log_entry(log_path: Path, status: str, repo_id: str, message: str = "") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"{timestamp}\t{status}\t{repo_id}"
    if message:
        line = f"{line}\t{message}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def run_hf_download(repo_id: str, csv_path: Path) -> subprocess.CompletedProcess[str]:
    if not HF_DOWNLOAD_SCRIPT.exists():
        raise RuntimeError(f"Expected hf_download.py at '{HF_DOWNLOAD_SCRIPT}', but it was not found.")

    command = [
        sys.executable,
        str(HF_DOWNLOAD_SCRIPT),
        "--csv",
        str(csv_path),
        "--optimization-level",
        "disable_all",
        "--optimization-level",
        "basic",
        "--optimization-level",
        "extended",
        repo_id,
    ]
    return subprocess.run(
        command,
        check=False,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
    )


def summarize_error(output: str) -> str:
    lines = [
        " ".join(line.strip().split())
        for line in output.replace("\r", "\n").splitlines()
        if line.strip()
    ]
    if not lines:
        return "Unknown error"

    for line in reversed(lines):
        if line.startswith("RuntimeError:"):
            return line

    for line in reversed(lines):
        if "Error:" in line and "CalledProcessError" not in line:
            return line

    for line in reversed(lines):
        if "CalledProcessError" not in line and not line.startswith("Traceback"):
            return line

    return lines[-1]


def main() -> int:
    args = parse_args()
    csv_path = normalize_path(args.csv)
    source_log_path = normalize_path(args.source_log)
    log_path = normalize_path(args.log)
    logged_repos = read_logged_repos(log_path)
    repo_ids = read_successful_repos(source_log_path)
    done_count = 0
    failed_count = 0

    for repo_id in repo_ids:
        if repo_id in logged_repos:
            print(f"Skipping {repo_id} (already logged).", flush=True)
            continue

        print(f"Processing {repo_id}...", flush=True)
        result = run_hf_download(repo_id, csv_path)
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)

        if result.returncode == 0:
            append_log_entry(log_path, "DONE", repo_id)
            logged_repos.add(repo_id)
            done_count += 1
            continue

        error_message = summarize_error(f"{result.stderr}\n{result.stdout}")
        append_log_entry(log_path, "FAILED", repo_id, error_message)
        logged_repos.add(repo_id)
        failed_count += 1
        print(f"Failed {repo_id}: {error_message}", file=sys.stderr, flush=True)

    print(
        f"Run finished. New successes: {done_count}. New failures: {failed_count}. Progress log: {log_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
