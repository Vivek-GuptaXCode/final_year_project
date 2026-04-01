from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create chronological train/val/test splits with optional leakage gap."
    )
    parser.add_argument("--input", required=True, help="Input labeled CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory for split CSV files.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70).")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (default: 0.15).")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15).")
    parser.add_argument(
        "--gap-seconds",
        type=int,
        default=120,
        help="Gap applied between train/val and val/test boundaries to avoid look-ahead leakage.",
    )
    return parser.parse_args()


def _to_int(value: str, field_name: str) -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"Invalid integer-like value for {field_name}: {value}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    if args.gap_seconds < 0:
        raise ValueError("gap_seconds must be >= 0")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV is missing header")
        fieldnames = list(reader.fieldnames)
        if "timestamp_s" not in fieldnames:
            raise ValueError("Input CSV must contain timestamp_s")
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV has no data rows")

    rows.sort(key=lambda row: _to_int(row["timestamp_s"], "timestamp_s"))
    unique_ts = sorted({_to_int(row["timestamp_s"], "timestamp_s") for row in rows})
    if len(unique_ts) < 3:
        raise ValueError("Need at least 3 distinct timestamps to build train/val/test splits")

    n = len(unique_ts)
    train_count_target = max(1, int(n * args.train_ratio))
    val_count_target = max(1, int(n * args.val_ratio))

    train_end_idx = min(train_count_target - 1, n - 3)
    train_end_ts = unique_ts[train_end_idx]

    val_start_idx = bisect.bisect_right(unique_ts, train_end_ts + args.gap_seconds)
    if val_start_idx >= n - 1:
        raise ValueError("Validation split is empty. Reduce gap_seconds or adjust split ratios.")
    val_start_ts = unique_ts[val_start_idx]

    val_end_idx = min(val_start_idx + val_count_target - 1, n - 2)
    val_end_ts = unique_ts[val_end_idx]

    test_start_idx = bisect.bisect_right(unique_ts, val_end_ts + args.gap_seconds)
    if test_start_idx >= n:
        raise ValueError("Test split is empty. Reduce gap_seconds or adjust split ratios.")

    test_start_ts = unique_ts[test_start_idx]

    train_rows = [row for row in rows if _to_int(row["timestamp_s"], "timestamp_s") <= train_end_ts]
    val_rows = [
        row
        for row in rows
        if val_start_ts <= _to_int(row["timestamp_s"], "timestamp_s") <= val_end_ts
    ]
    test_rows = [row for row in rows if _to_int(row["timestamp_s"], "timestamp_s") >= test_start_ts]

    if not val_rows:
        raise ValueError("Validation split is empty. Reduce gap_seconds or adjust split ratios.")
    if not test_rows:
        raise ValueError("Test split is empty. Reduce gap_seconds or adjust split ratios.")

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    _write_rows(train_path, fieldnames, train_rows)
    _write_rows(val_path, fieldnames, val_rows)
    _write_rows(test_path, fieldnames, test_rows)

    metadata = {
        "input": str(input_path),
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "gap_seconds": args.gap_seconds,
        "boundaries": {
            "train_end_ts": train_end_ts,
            "val_end_ts": val_end_ts,
            "test_start_ts": test_start_ts,
        },
        "rows": {
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "gap_dropped": len(rows) - len(train_rows) - len(val_rows) - len(test_rows),
        },
        "hashes": {
            "train_sha256": _sha256_file(train_path),
            "val_sha256": _sha256_file(val_path),
            "test_sha256": _sha256_file(test_path),
        },
    }

    metadata_path = output_dir / "split_manifest.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[processing] wrote splits to {output_dir}")
    print(f"[processing] rows train/val/test = {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")


if __name__ == "__main__":
    main()
