from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate temporal split integrity and leakage constraints."
    )
    parser.add_argument("--split-dir", required=True, help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument(
        "--expected-gap-seconds",
        type=int,
        default=120,
        help="Expected minimum temporal gap between split boundaries.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional output path for JSON report (default: <split-dir>/leakage_report.json)",
    )
    return parser.parse_args()


def _to_int(value: str, field_name: str) -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"Invalid integer-like value for {field_name}: {value}") from exc


def _load_split(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {path}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def _key_set(rows: list[dict[str, str]], has_rsu: bool) -> set[tuple[str, str]]:
    if has_rsu:
        return {(row["timestamp_s"], row["rsu_node"]) for row in rows}
    return {(row["timestamp_s"], "") for row in rows}


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)

    train_rows, train_fields = _load_split(split_dir / "train.csv")
    val_rows, val_fields = _load_split(split_dir / "val.csv")
    test_rows, test_fields = _load_split(split_dir / "test.csv")

    for fields, name in [(train_fields, "train"), (val_fields, "val"), (test_fields, "test")]:
        if "timestamp_s" not in fields:
            raise ValueError(f"timestamp_s missing from {name}.csv")

    has_rsu = "rsu_node" in train_fields and "rsu_node" in val_fields and "rsu_node" in test_fields

    train_ts = sorted({_to_int(row["timestamp_s"], "timestamp_s") for row in train_rows})
    val_ts = sorted({_to_int(row["timestamp_s"], "timestamp_s") for row in val_rows})
    test_ts = sorted({_to_int(row["timestamp_s"], "timestamp_s") for row in test_rows})

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    checks["non_empty_splits"] = bool(train_rows) and bool(val_rows) and bool(test_rows)

    train_max = max(train_ts) if train_ts else None
    val_min = min(val_ts) if val_ts else None
    val_max = max(val_ts) if val_ts else None
    test_min = min(test_ts) if test_ts else None

    checks["chronological_order"] = (
        train_max is not None
        and val_min is not None
        and val_max is not None
        and test_min is not None
        and train_max < val_min
        and val_max < test_min
    )

    if train_max is not None and val_min is not None and val_max is not None and test_min is not None:
        checks["gap_respected"] = (
            (val_min - train_max) >= (args.expected_gap_seconds + 1)
            and (test_min - val_max) >= (args.expected_gap_seconds + 1)
        )
        details["observed_gaps"] = {
            "train_to_val_seconds": val_min - train_max,
            "val_to_test_seconds": test_min - val_max,
        }
    else:
        checks["gap_respected"] = False
        details["observed_gaps"] = None

    train_keys = _key_set(train_rows, has_rsu)
    val_keys = _key_set(val_rows, has_rsu)
    test_keys = _key_set(test_rows, has_rsu)

    overlap_train_val = len(train_keys.intersection(val_keys))
    overlap_val_test = len(val_keys.intersection(test_keys))
    overlap_train_test = len(train_keys.intersection(test_keys))

    checks["no_row_overlap"] = overlap_train_val == 0 and overlap_val_test == 0 and overlap_train_test == 0

    details["row_overlap_counts"] = {
        "train_val": overlap_train_val,
        "val_test": overlap_val_test,
        "train_test": overlap_train_test,
    }
    details["row_counts"] = {
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
    }

    passed = all(checks.values())
    report = {
        "passed": passed,
        "checks": checks,
        "details": details,
    }

    report_path = Path(args.report) if args.report else split_dir / "leakage_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[processing] leakage report: {report_path}")
    print(f"[processing] status: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
