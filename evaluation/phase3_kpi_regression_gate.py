"""
Phase 3 Routing: True KPI Regression Gate
=========================================
Compares baseline vs Phase 3 runs using SUMO output artifacts and evaluates
whether Phase 3 causes unacceptable regression in core KPIs.

Supported inputs:
  - SUMO statistic-output XML (--source-type statistics)
  - SUMO summary XML (--source-type summary)
  - SUMO tripinfo XML (--source-type tripinfo)
  - Normalized JSON (--source-type json)

Primary KPI gate (P3.4):
  - mean travel time regression percent (lower is better)
  - mean waiting time regression percent (lower is better)
  - throughput regression percent (higher is better)

The gate checks both the mean delta and a bootstrap confidence interval so
single-seed noise does not produce false positives.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import glob
import json
import math
from pathlib import Path
import random
from typing import Any
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunKpi:
    run_id: str
    source_file: str
    mean_travel_time_s: float
    mean_waiting_time_s: float
    throughput_veh_per_h: float
    completed_trips: int
    total_travel_time_s: float | None = None
    total_depart_delay_s: float | None = None


@dataclass
class PairDelta:
    pair_id: str
    baseline_run_id: str
    phase3_run_id: str
    baseline_file: str
    phase3_file: str
    baseline: RunKpi
    phase3: RunKpi
    travel_time_delta_pct: float
    waiting_time_delta_pct: float
    throughput_delta_pct: float
    fair_total_time_delta_pct: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate true P3.4 regression gate from baseline and Phase 3 KPI artifacts "
            "(travel time, waiting time, throughput)."
        )
    )
    parser.add_argument(
        "--source-type",
        choices=["statistics", "summary", "tripinfo", "json"],
        required=True,
        help="Type of input artifact for both baseline and phase3 globs.",
    )
    parser.add_argument(
        "--baseline-glob",
        required=True,
        help="Glob pattern for baseline artifacts.",
    )
    parser.add_argument(
        "--phase3-glob",
        required=True,
        help="Glob pattern for phase3 artifacts.",
    )
    parser.add_argument(
        "--pairing",
        choices=["run-id", "index"],
        default="run-id",
        help=(
            "How to pair baseline and phase3 runs: run-id matches by parsed run_id; "
            "index matches sorted files by position."
        ),
    )
    parser.add_argument(
        "--max-travel-time-regression-pct",
        type=float,
        default=2.0,
        help="Maximum allowed regression in mean travel time (percent).",
    )
    parser.add_argument(
        "--max-waiting-time-regression-pct",
        type=float,
        default=3.0,
        help="Maximum allowed regression in mean waiting time (percent).",
    )
    parser.add_argument(
        "--max-throughput-drop-pct",
        type=float,
        default=1.0,
        help="Maximum allowed throughput drop (percent).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for mean-delta confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=17,
        help="Random seed for bootstrap reproducibility.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap interval (default: 0.95).",
    )
    parser.add_argument(
        "--output",
        default="evaluation/phase3_kpi_regression_results.json",
        help="Output JSON path (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the gate fails.",
    )
    return parser.parse_args()


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _find_child(node: ET.Element, child_name: str) -> ET.Element | None:
    for child in list(node):
        if _local_name(child.tag) == child_name:
            return child
    return None


def _float_attr(node: ET.Element | None, key: str) -> float | None:
    if node is None:
        return None
    raw = node.attrib.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _int_attr(node: ET.Element | None, key: str) -> int | None:
    value = _float_attr(node, key)
    if value is None:
        return None
    try:
        return int(round(value))
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / float(len(values))


def _default_run_id(path: Path) -> str:
    # Most run artifacts are named by folder and use generic filenames.
    # Prefer folder name for stable pairing across baseline/phase3 runs.
    if path.parent and path.parent.name:
        return path.parent.name
    return path.stem


def _pct_delta(baseline_value: float, phase3_value: float) -> float:
    eps = 1e-9
    if abs(baseline_value) <= eps:
        if abs(phase3_value) <= eps:
            return 0.0
        return math.inf if phase3_value > 0 else -math.inf
    return ((phase3_value - baseline_value) / baseline_value) * 100.0


def _percentile_sorted(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    q = max(0.0, min(1.0, float(q)))
    last = len(sorted_values) - 1
    if last <= 0:
        return sorted_values[0]
    pos = q * last
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    finite_values = [v for v in values if math.isfinite(v)]
    if not finite_values:
        return (math.nan, math.nan)
    if len(finite_values) == 1:
        return (finite_values[0], finite_values[0])

    n = len(finite_values)
    sample_count = max(100, int(samples))
    rng = random.Random(seed)

    bootstrap_means: list[float] = []
    for _ in range(sample_count):
        total = 0.0
        for _ in range(n):
            total += finite_values[rng.randrange(n)]
        bootstrap_means.append(total / float(n))

    bootstrap_means.sort()
    alpha = max(0.0, min(1.0, 1.0 - float(confidence_level)))
    lo = _percentile_sorted(bootstrap_means, alpha / 2.0)
    hi = _percentile_sorted(bootstrap_means, 1.0 - (alpha / 2.0))
    return (lo, hi)


def _expand_paths(pattern: str) -> list[Path]:
    matches = sorted(Path(p).resolve() for p in glob.glob(pattern))
    if not matches:
        raise ValueError(f"No files matched pattern: {pattern}")
    return matches


def _parse_statistics_xml(path: Path) -> RunKpi:
    root = ET.parse(path).getroot()
    if _local_name(root.tag) != "statistics":
        raise ValueError(f"Expected <statistics> root in {path}")

    performance = _find_child(root, "performance")
    vehicles = _find_child(root, "vehicles")
    trip_stats = _find_child(root, "vehicleTripStatistics")

    sim_duration_s = _float_attr(performance, "duration")
    if sim_duration_s is None:
        begin = _float_attr(performance, "begin") or 0.0
        end = _float_attr(performance, "end")
        if end is not None:
            sim_duration_s = max(0.0, end - begin)
    if sim_duration_s is None or sim_duration_s <= 0:
        sim_duration_s = 1.0

    completed_trips = _int_attr(trip_stats, "count")
    if completed_trips is None:
        completed_trips = _int_attr(vehicles, "inserted") or 0

    mean_travel = _float_attr(trip_stats, "duration")
    mean_wait = _float_attr(trip_stats, "waitingTime")
    total_tt = _float_attr(trip_stats, "totalTravelTime")
    total_depart_delay = _float_attr(trip_stats, "totalDepartDelay")

    if mean_travel is None:
        if total_tt is not None and completed_trips > 0:
            mean_travel = total_tt / float(completed_trips)
        else:
            mean_travel = 0.0
    if mean_wait is None:
        mean_wait = 0.0

    throughput = (completed_trips / sim_duration_s) * 3600.0 if sim_duration_s > 0 else 0.0

    return RunKpi(
        run_id=_default_run_id(path),
        source_file=str(path),
        mean_travel_time_s=float(mean_travel),
        mean_waiting_time_s=float(mean_wait),
        throughput_veh_per_h=float(throughput),
        completed_trips=int(completed_trips),
        total_travel_time_s=total_tt,
        total_depart_delay_s=total_depart_delay,
    )


def _parse_summary_xml(path: Path) -> RunKpi:
    root = ET.parse(path).getroot()
    if _local_name(root.tag) != "summary":
        raise ValueError(f"Expected <summary> root in {path}")

    last_step: ET.Element | None = None
    last_time = -math.inf
    for child in list(root):
        if _local_name(child.tag) != "step":
            continue
        time_s = _float_attr(child, "time")
        if time_s is None:
            continue
        if time_s >= last_time:
            last_time = time_s
            last_step = child

    if last_step is None:
        raise ValueError(f"No <step> entries found in summary output: {path}")

    sim_end_s = _float_attr(last_step, "time") or 0.0
    completed_trips = _int_attr(last_step, "ended") or _int_attr(last_step, "arrived") or 0
    mean_travel = _float_attr(last_step, "meanTravelTime") or 0.0
    mean_wait = _float_attr(last_step, "meanWaitingTime") or 0.0
    throughput = (completed_trips / sim_end_s) * 3600.0 if sim_end_s > 0 else 0.0

    total_tt = mean_travel * float(completed_trips) if completed_trips > 0 else 0.0

    return RunKpi(
        run_id=_default_run_id(path),
        source_file=str(path),
        mean_travel_time_s=float(mean_travel),
        mean_waiting_time_s=float(mean_wait),
        throughput_veh_per_h=float(throughput),
        completed_trips=int(completed_trips),
        total_travel_time_s=float(total_tt),
        total_depart_delay_s=None,
    )


def _parse_tripinfo_xml(path: Path) -> RunKpi:
    root = ET.parse(path).getroot()
    if _local_name(root.tag) != "tripinfos":
        raise ValueError(f"Expected <tripinfos> root in {path}")

    durations: list[float] = []
    waits: list[float] = []
    depart_delays: list[float] = []
    arrivals: list[float] = []
    departs: list[float] = []

    for child in list(root):
        if _local_name(child.tag) != "tripinfo":
            continue
        duration = _float_attr(child, "duration")
        waiting = _float_attr(child, "waitingTime")
        depart_delay = _float_attr(child, "departDelay")
        arrival = _float_attr(child, "arrival")
        depart = _float_attr(child, "depart")

        if duration is not None:
            durations.append(duration)
        if waiting is not None:
            waits.append(waiting)
        if depart_delay is not None:
            depart_delays.append(depart_delay)
        if arrival is not None:
            arrivals.append(arrival)
        if depart is not None:
            departs.append(depart)

    completed_trips = len(durations)
    if completed_trips <= 0:
        raise ValueError(f"No <tripinfo> entries with duration found in {path}")

    mean_travel = _mean(durations)
    mean_wait = _mean(waits) if waits else 0.0

    sim_start_s = min(departs) if departs else 0.0
    sim_end_s = max(arrivals) if arrivals else max(durations)
    sim_horizon_s = max(1.0, sim_end_s - sim_start_s)
    throughput = (completed_trips / sim_horizon_s) * 3600.0

    total_tt = sum(durations)
    total_depart_delay = sum(depart_delays) if depart_delays else None

    return RunKpi(
        run_id=_default_run_id(path),
        source_file=str(path),
        mean_travel_time_s=float(mean_travel),
        mean_waiting_time_s=float(mean_wait),
        throughput_veh_per_h=float(throughput),
        completed_trips=int(completed_trips),
        total_travel_time_s=float(total_tt),
        total_depart_delay_s=(float(total_depart_delay) if total_depart_delay is not None else None),
    )


def _coalesce(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _parse_run_from_json_record(record: dict[str, Any], source_file: str, fallback_id: str) -> RunKpi:
    run_id_raw = _coalesce(record, ["run_id", "runId", "id"])
    run_id = str(run_id_raw) if run_id_raw is not None else fallback_id

    mean_travel_raw = _coalesce(record, ["mean_travel_time_s", "meanTravelTime", "travel_time_s"])
    mean_wait_raw = _coalesce(record, ["mean_waiting_time_s", "meanWaitingTime", "waiting_time_s"])
    throughput_raw = _coalesce(record, ["throughput_veh_per_h", "throughput_vph", "throughput"])

    completed_raw = _coalesce(record, ["completed_trips", "count", "trips"])
    sim_duration_raw = _coalesce(record, ["sim_duration_s", "duration_s", "duration"])

    total_tt_raw = _coalesce(record, ["total_travel_time_s", "totalTravelTime"])
    total_dd_raw = _coalesce(record, ["total_depart_delay_s", "totalDepartDelay"])

    try:
        mean_travel = float(mean_travel_raw)
    except (TypeError, ValueError):
        mean_travel = 0.0

    try:
        mean_wait = float(mean_wait_raw)
    except (TypeError, ValueError):
        mean_wait = 0.0

    completed = 0
    try:
        completed = int(float(completed_raw))
    except (TypeError, ValueError):
        completed = 0

    throughput = math.nan
    try:
        throughput = float(throughput_raw)
    except (TypeError, ValueError):
        throughput = math.nan

    if not math.isfinite(throughput):
        try:
            sim_duration = float(sim_duration_raw)
            throughput = (completed / sim_duration) * 3600.0 if sim_duration > 0 else 0.0
        except (TypeError, ValueError):
            throughput = 0.0

    total_tt: float | None
    try:
        total_tt = float(total_tt_raw)
    except (TypeError, ValueError):
        total_tt = None

    total_dd: float | None
    try:
        total_dd = float(total_dd_raw)
    except (TypeError, ValueError):
        total_dd = None

    if total_tt is None and completed > 0:
        total_tt = mean_travel * float(completed)

    return RunKpi(
        run_id=run_id,
        source_file=source_file,
        mean_travel_time_s=mean_travel,
        mean_waiting_time_s=mean_wait,
        throughput_veh_per_h=float(throughput),
        completed_trips=completed,
        total_travel_time_s=total_tt,
        total_depart_delay_s=total_dd,
    )


def _parse_json(path: Path) -> list[RunKpi]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    records: list[dict[str, Any]] = []
    if isinstance(payload, list):
        records = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("runs"), list):
            records = [row for row in payload.get("runs", []) if isinstance(row, dict)]
        elif isinstance(payload.get("scenarios"), list):
            # Compatibility with existing evaluation JSON shape.
            # We read Phase 3-side records from each scenario entry when available.
            for row in payload.get("scenarios", []):
                if not isinstance(row, dict):
                    continue
                phase3 = row.get("phase3") if isinstance(row.get("phase3"), dict) else None
                if phase3 is None:
                    continue
                records.append(
                    {
                        "run_id": row.get("scenario"),
                        "mean_travel_time_s": phase3.get("mean_travel_time_s"),
                        "mean_waiting_time_s": phase3.get("mean_waiting_time_s"),
                        "throughput_veh_per_h": phase3.get("throughput_veh_per_h"),
                    }
                )
        else:
            records = [payload]
    else:
        raise ValueError(f"Unsupported JSON shape in {path}")

    if not records:
        raise ValueError(f"No run records found in JSON file: {path}")

    runs: list[RunKpi] = []
    for idx, record in enumerate(records):
        fallback_id = f"{_default_run_id(path)}#{idx + 1}"
        runs.append(_parse_run_from_json_record(record, str(path), fallback_id))
    return runs


def _parse_one(path: Path, source_type: str) -> list[RunKpi]:
    if source_type == "statistics":
        return [_parse_statistics_xml(path)]
    if source_type == "summary":
        return [_parse_summary_xml(path)]
    if source_type == "tripinfo":
        return [_parse_tripinfo_xml(path)]
    if source_type == "json":
        return _parse_json(path)
    raise ValueError(f"Unsupported source type: {source_type}")


def _load_runs(pattern: str, source_type: str) -> list[RunKpi]:
    files = _expand_paths(pattern)
    runs: list[RunKpi] = []
    for path in files:
        runs.extend(_parse_one(path, source_type))
    return runs


def _pair_runs(
    baseline_runs: list[RunKpi],
    phase3_runs: list[RunKpi],
    pairing: str,
) -> list[tuple[RunKpi, RunKpi, str]]:
    if pairing == "run-id":
        baseline_by_id: dict[str, RunKpi] = {}
        phase3_by_id: dict[str, RunKpi] = {}

        for run in baseline_runs:
            if run.run_id in baseline_by_id:
                raise ValueError(f"Duplicate baseline run_id: {run.run_id}")
            baseline_by_id[run.run_id] = run

        for run in phase3_runs:
            if run.run_id in phase3_by_id:
                raise ValueError(f"Duplicate phase3 run_id: {run.run_id}")
            phase3_by_id[run.run_id] = run

        common = sorted(set(baseline_by_id.keys()) & set(phase3_by_id.keys()))
        if not common:
            raise ValueError(
                "No common run_id between baseline and phase3 artifacts. "
                "Use --pairing index or normalize run_id fields."
            )

        return [
            (baseline_by_id[run_id], phase3_by_id[run_id], run_id)
            for run_id in common
        ]

    baseline_sorted = sorted(baseline_runs, key=lambda r: r.source_file)
    phase3_sorted = sorted(phase3_runs, key=lambda r: r.source_file)
    n = min(len(baseline_sorted), len(phase3_sorted))
    if n <= 0:
        raise ValueError("No runs available for index-based pairing")

    pairs: list[tuple[RunKpi, RunKpi, str]] = []
    for i in range(n):
        pair_id = f"pair_{i + 1:03d}"
        pairs.append((baseline_sorted[i], phase3_sorted[i], pair_id))
    return pairs


def _build_pair_delta(baseline: RunKpi, phase3: RunKpi, pair_id: str) -> PairDelta:
    fair_delta: float | None = None
    if baseline.total_travel_time_s is not None and phase3.total_travel_time_s is not None:
        baseline_total = float(baseline.total_travel_time_s)
        phase3_total = float(phase3.total_travel_time_s)
        if baseline.total_depart_delay_s is not None and phase3.total_depart_delay_s is not None:
            baseline_total += float(baseline.total_depart_delay_s)
            phase3_total += float(phase3.total_depart_delay_s)
        fair_delta = _pct_delta(baseline_total, phase3_total)

    return PairDelta(
        pair_id=pair_id,
        baseline_run_id=baseline.run_id,
        phase3_run_id=phase3.run_id,
        baseline_file=baseline.source_file,
        phase3_file=phase3.source_file,
        baseline=baseline,
        phase3=phase3,
        travel_time_delta_pct=_pct_delta(baseline.mean_travel_time_s, phase3.mean_travel_time_s),
        waiting_time_delta_pct=_pct_delta(baseline.mean_waiting_time_s, phase3.mean_waiting_time_s),
        throughput_delta_pct=_pct_delta(baseline.throughput_veh_per_h, phase3.throughput_veh_per_h),
        fair_total_time_delta_pct=fair_delta,
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.3f}"


def _print_report(
    pair_deltas: list[PairDelta],
    *,
    travel_mean: float,
    travel_ci: tuple[float, float],
    wait_mean: float,
    wait_ci: tuple[float, float],
    throughput_mean: float,
    throughput_ci: tuple[float, float],
    travel_pass: bool,
    wait_pass: bool,
    throughput_pass: bool,
    overall_pass: bool,
    args: argparse.Namespace,
) -> None:
    print("\n" + "=" * 108)
    print("  PHASE 3 TRUE KPI REGRESSION GATE (P3.4)")
    print("=" * 108)
    print(
        "Thresholds: "
        f"travel_time<=+{args.max_travel_time_regression_pct:.2f}%  "
        f"waiting_time<=+{args.max_waiting_time_regression_pct:.2f}%  "
        f"throughput>=-{args.max_throughput_drop_pct:.2f}%"
    )
    print(
        f"Bootstrap: samples={args.bootstrap_samples} seed={args.bootstrap_seed} "
        f"confidence={args.confidence_level:.3f}"
    )

    print("\nPer-pair deltas (Phase3 vs Baseline, percent):")
    header = (
        f"{'Pair':<14} {'Travel%':>10} {'Wait%':>10} {'Throughput%':>12} "
        f"{'FairTotal%':>12} {'BaseRun':<22} {'P3Run':<22}"
    )
    print(header)
    print("-" * len(header))

    for delta in pair_deltas:
        print(
            f"{delta.pair_id:<14} "
            f"{_fmt(delta.travel_time_delta_pct):>10} "
            f"{_fmt(delta.waiting_time_delta_pct):>10} "
            f"{_fmt(delta.throughput_delta_pct):>12} "
            f"{_fmt(delta.fair_total_time_delta_pct):>12} "
            f"{delta.baseline_run_id:<22} "
            f"{delta.phase3_run_id:<22}"
        )

    print("\nAggregate deltas:")
    print(
        f"  Travel time delta mean={_fmt(travel_mean)}% "
        f"CI{int(args.confidence_level * 100)}=[{_fmt(travel_ci[0])}, {_fmt(travel_ci[1])}] "
        f"=> {'PASS' if travel_pass else 'FAIL'}"
    )
    print(
        f"  Waiting time delta mean={_fmt(wait_mean)}% "
        f"CI{int(args.confidence_level * 100)}=[{_fmt(wait_ci[0])}, {_fmt(wait_ci[1])}] "
        f"=> {'PASS' if wait_pass else 'FAIL'}"
    )
    print(
        f"  Throughput delta mean={_fmt(throughput_mean)}% "
        f"CI{int(args.confidence_level * 100)}=[{_fmt(throughput_ci[0])}, {_fmt(throughput_ci[1])}] "
        f"=> {'PASS' if throughput_pass else 'FAIL'}"
    )

    print("\nGate verdict:")
    print(f"  P3.4 TRUE KPI REGRESSION: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 108 + "\n")


def main() -> int:
    args = parse_args()

    baseline_runs = _load_runs(args.baseline_glob, args.source_type)
    phase3_runs = _load_runs(args.phase3_glob, args.source_type)
    pairs = _pair_runs(baseline_runs, phase3_runs, args.pairing)

    pair_deltas = [_build_pair_delta(baseline, phase3, pair_id) for baseline, phase3, pair_id in pairs]

    travel_values = [item.travel_time_delta_pct for item in pair_deltas if math.isfinite(item.travel_time_delta_pct)]
    wait_values = [item.waiting_time_delta_pct for item in pair_deltas if math.isfinite(item.waiting_time_delta_pct)]
    throughput_values = [item.throughput_delta_pct for item in pair_deltas if math.isfinite(item.throughput_delta_pct)]

    travel_mean = _mean(travel_values)
    wait_mean = _mean(wait_values)
    throughput_mean = _mean(throughput_values)

    travel_ci = _bootstrap_mean_ci(
        travel_values,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    wait_ci = _bootstrap_mean_ci(
        wait_values,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    throughput_ci = _bootstrap_mean_ci(
        throughput_values,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )

    travel_pass = (
        math.isfinite(travel_mean)
        and math.isfinite(travel_ci[1])
        and travel_mean <= float(args.max_travel_time_regression_pct)
        and travel_ci[1] <= float(args.max_travel_time_regression_pct)
    )
    wait_pass = (
        math.isfinite(wait_mean)
        and math.isfinite(wait_ci[1])
        and wait_mean <= float(args.max_waiting_time_regression_pct)
        and wait_ci[1] <= float(args.max_waiting_time_regression_pct)
    )
    throughput_pass = (
        math.isfinite(throughput_mean)
        and math.isfinite(throughput_ci[0])
        and throughput_mean >= -float(args.max_throughput_drop_pct)
        and throughput_ci[0] >= -float(args.max_throughput_drop_pct)
    )

    overall_pass = travel_pass and wait_pass and throughput_pass

    _print_report(
        pair_deltas,
        travel_mean=travel_mean,
        travel_ci=travel_ci,
        wait_mean=wait_mean,
        wait_ci=wait_ci,
        throughput_mean=throughput_mean,
        throughput_ci=throughput_ci,
        travel_pass=travel_pass,
        wait_pass=wait_pass,
        throughput_pass=throughput_pass,
        overall_pass=overall_pass,
        args=args,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_type": args.source_type,
        "pairing": args.pairing,
        "pair_count": len(pair_deltas),
        "thresholds": {
            "max_travel_time_regression_pct": float(args.max_travel_time_regression_pct),
            "max_waiting_time_regression_pct": float(args.max_waiting_time_regression_pct),
            "max_throughput_drop_pct": float(args.max_throughput_drop_pct),
        },
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "seed": int(args.bootstrap_seed),
            "confidence_level": float(args.confidence_level),
        },
        "methodology": {
            "travel_delta_formula_pct": "100 * (phase3 - baseline) / baseline",
            "waiting_delta_formula_pct": "100 * (phase3 - baseline) / baseline",
            "throughput_delta_formula_pct": "100 * (phase3 - baseline) / baseline",
            "ci_method": "nonparametric bootstrap percentile CI on paired mean deltas",
            "references": [
                "https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html",
                "https://sumo.dlr.de/docs/Simulation/Output/StatisticOutput.html",
                "https://sumo.dlr.de/docs/Simulation/Output/Summary.html",
            ],
        },
        "aggregate": {
            "travel_time_delta_pct": {
                "mean": travel_mean,
                "ci": list(travel_ci),
                "pass": travel_pass,
            },
            "waiting_time_delta_pct": {
                "mean": wait_mean,
                "ci": list(wait_ci),
                "pass": wait_pass,
            },
            "throughput_delta_pct": {
                "mean": throughput_mean,
                "ci": list(throughput_ci),
                "pass": throughput_pass,
            },
        },
        "overall_gate": {
            "name": "P3.4_true_kpi_regression",
            "status": "PASS" if overall_pass else "FAIL",
        },
        "pairs": [
            {
                "pair_id": item.pair_id,
                "baseline_run_id": item.baseline_run_id,
                "phase3_run_id": item.phase3_run_id,
                "baseline_file": item.baseline_file,
                "phase3_file": item.phase3_file,
                "baseline": asdict(item.baseline),
                "phase3": asdict(item.phase3),
                "travel_time_delta_pct": item.travel_time_delta_pct,
                "waiting_time_delta_pct": item.waiting_time_delta_pct,
                "throughput_delta_pct": item.throughput_delta_pct,
                "fair_total_time_delta_pct": item.fair_total_time_delta_pct,
            }
            for item in pair_deltas
        ],
    }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved KPI gate report to: {output_path}")

    if args.strict and not overall_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
