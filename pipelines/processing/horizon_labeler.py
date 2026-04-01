from __future__ import annotations

import argparse
import bisect
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate future congestion horizon labels from 1 Hz RSU features."
    )
    parser.add_argument(
        "--input-rsu",
        required=True,
        help="Path to input rsu_features_1hz.csv.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output labeled CSV.",
    )
    parser.add_argument(
        "--target-column",
        default="congested_global",
        help="Binary source column used to compute future labels.",
    )
    parser.add_argument(
        "--horizons",
        default="60,120",
        help="Comma-separated horizon seconds to label (example: 60,120).",
    )
    return parser.parse_args()


def _parse_horizons(raw: str) -> list[int]:
    horizons = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("All horizons must be positive integers in seconds")
        horizons.append(value)
    if not horizons:
        raise ValueError("At least one horizon must be provided")
    return sorted(set(horizons))


def _to_int(value: str, field_name: str) -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"Invalid integer-like value for {field_name}: {value}") from exc


def _to_binary(value: str, field_name: str) -> int:
    v = _to_int(value, field_name)
    return 1 if v > 0 else 0


def _label_group(rows: list[dict[str, str]], horizons: list[int], target_column: str) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: _to_int(row["timestamp_s"], "timestamp_s"))
    timestamps = [_to_int(row["timestamp_s"], "timestamp_s") for row in ordered]
    targets = [_to_binary(row[target_column], target_column) for row in ordered]

    # Prefix sum lets us query whether any positive target exists in future windows.
    prefix = [0]
    for flag in targets:
        prefix.append(prefix[-1] + flag)

    labeled: list[dict[str, str]] = []
    for idx, row in enumerate(ordered):
        ts = timestamps[idx]
        out = dict(row)

        for horizon in horizons:
            end_idx_exclusive = bisect.bisect_right(timestamps, ts + horizon)
            start_idx = idx + 1
            positives = 0
            if start_idx < end_idx_exclusive:
                positives = prefix[end_idx_exclusive] - prefix[start_idx]
            out[f"label_congestion_{horizon}s"] = "1" if positives > 0 else "0"

        labeled.append(out)

    return labeled


def main() -> None:
    args = parse_args()
    horizons = _parse_horizons(args.horizons)

    input_path = Path(args.input_rsu)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV is missing header")

        fieldnames = list(reader.fieldnames)
        required = {"timestamp_s", "rsu_node", args.target_column}
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")

        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            grouped[row["rsu_node"]].append(row)

    labeled_rows: list[dict[str, str]] = []
    for _rsu_node, rows in grouped.items():
        labeled_rows.extend(_label_group(rows, horizons, args.target_column))

    labeled_rows.sort(key=lambda row: (_to_int(row["timestamp_s"], "timestamp_s"), row["rsu_node"]))

    output_fields = fieldnames + [f"label_congestion_{h}s" for h in horizons]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(labeled_rows)

    print(f"[processing] wrote {len(labeled_rows)} labeled rows to {output_path}")


if __name__ == "__main__":
    main()
