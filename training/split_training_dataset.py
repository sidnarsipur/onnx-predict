#!/usr/bin/env python3
"""Split training_dataset.csv into train/test/val CSVs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible train/test/val CSV splits.")
    parser.add_argument(
        "--input",
        default="training/full_dataset.csv",
        help="Input dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="training",
        help="Directory for split CSV files.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=67,
        help="Random seed for row shuffling.",
    )
    return parser.parse_args()


def write_split(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    rng = random.Random(args.random_state)
    rng.shuffle(rows)

    total_rows = len(rows)
    train_count = int(total_rows * 0.70)
    remaining_count = total_rows - train_count
    test_count = remaining_count // 2
    val_count = remaining_count - test_count

    train_rows = rows[:train_count]
    test_rows = rows[train_count : train_count + test_count]
    val_rows = rows[train_count + test_count :]

    write_split(output_dir / "train_set.csv", header, train_rows)
    write_split(output_dir / "test_set.csv", header, test_rows)
    write_split(output_dir / "val_set.csv", header, val_rows)

    print(f"train rows: {len(train_rows)}")
    print(f"test rows: {len(test_rows)}")
    print(f"val rows: {len(val_rows)}")
    print(f"total rows: {len(train_rows) + len(test_rows) + len(val_rows)}")


if __name__ == "__main__":
    main()
