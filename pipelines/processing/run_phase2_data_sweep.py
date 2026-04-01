from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
DEFAULT_OUTPUT_ROOT = "data/processed/phase2_sweeps"
DEFAULT_REPORT_PATH = "docs/reports/phase2_data_sweep_report.md"
DEFAULT_CONTRACT = "sumo/scenarios/sumo_contract.json"
DEFAULT_HORIZONS = "60,120"
DEFAULT_SCENARIOS = "demo,low,medium,high"
DEFAULT_SEEDS = "11,17,23"
DEFAULT_TRAFFIC_SCALES = "0.8,1.0,1.2"
DEFAULT_TRAFFIC_REDUCTION_PCTS = "20,40"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    scenario: str
    seed: int
    traffic_scale: float
    traffic_reduction_pct: float


@dataclass
class CommandResult:
    command: list[str]
    return_code: int | None
    duration_seconds: float
    skipped: bool
    stdout_tail: list[str]
    stderr_tail: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase-2 data sweeps with SUMO + labeling/split/leakage checks and generate a quality report."
        )
    )
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Project root directory (default: repository root inferred from this script).",
    )
    parser.add_argument(
        "--contract",
        default=DEFAULT_CONTRACT,
        help="Path to SUMO scenario contract JSON (relative to project root unless absolute).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional JSON config for matrix values. Supported keys: "
            "scenarios, seeds, traffic_scales, traffic_reduction_pcts, quality_overrides."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default=DEFAULT_SCENARIOS,
        help="Comma-separated SUMO scenarios.",
    )
    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS,
        help="Comma-separated integer seeds.",
    )
    parser.add_argument(
        "--traffic-scales",
        default=DEFAULT_TRAFFIC_SCALES,
        help="Comma-separated traffic-scale values passed to SUMO.",
    )
    parser.add_argument(
        "--traffic-reduction-pcts",
        default=DEFAULT_TRAFFIC_REDUCTION_PCTS,
        help="Comma-separated traffic-reduction percentages passed to SUMO.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max-steps override passed to SUMO runs.",
    )
    parser.add_argument(
        "--horizons",
        default=DEFAULT_HORIZONS,
        help="Comma-separated label horizons in seconds (default: 60,120).",
    )
    parser.add_argument(
        "--gap-seconds",
        type=int,
        default=5,
        help="Leakage gap for temporal split and validator (default: 5).",
    )
    parser.add_argument(
        "--run-prefix",
        default="phase2sweep",
        help="Prefix for generated runtime log run ids.",
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help="Optional sweep id. Default is UTC timestamped id.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for sweep manifest artifacts (relative to project root unless absolute).",
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help="Markdown report path relative to project root unless absolute.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for subprocess commands (default: current interpreter).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=500,
        help="Minimum labeled rows expected per run for quality gate.",
    )
    parser.add_argument(
        "--min-positive-rate",
        type=float,
        default=0.01,
        help="Minimum acceptable positive rate per horizon.",
    )
    parser.add_argument(
        "--max-positive-rate",
        type=float,
        default=0.99,
        help="Maximum acceptable positive rate per horizon.",
    )
    parser.add_argument(
        "--min-split-positive-rows",
        type=int,
        default=3,
        help="Minimum positive rows per split (train/val/test) and horizon.",
    )
    parser.add_argument(
        "--skip-sumo",
        action="store_true",
        help="Skip SUMO generation and process existing run directories only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute commands; only plan matrix and write report/manifest.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any run fails command execution or quality gates.",
    )
    return parser.parse_args()


def _parse_csv(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one CSV value")
    return values


def _parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for item in _parse_csv(raw):
        values.append(int(item))
    return sorted(set(values))


def _parse_float_csv(raw: str) -> list[float]:
    values: list[float] = []
    for item in _parse_csv(raw):
        values.append(float(item))
    return sorted(set(values))


def _parse_horizons(raw: str) -> list[int]:
    horizons = _parse_int_csv(raw)
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("All horizons must be positive integers")
    return horizons


def _load_optional_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object")
    return payload


def _resolve_from_config(
    config: dict[str, Any],
    key: str,
    fallback: list[Any],
) -> list[Any]:
    if key not in config:
        return fallback
    raw = config[key]
    if not isinstance(raw, list):
        raise ValueError(f"Config field '{key}' must be a list")
    if not raw:
        raise ValueError(f"Config field '{key}' must not be empty")
    return raw


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_sweep_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def _float_token(value: float) -> str:
    raw = f"{value:.3f}".rstrip("0").rstrip(".")
    return raw.replace("-", "m").replace(".", "p")


def _build_run_id(prefix: str, scenario: str, seed: int, scale: float, reduction: float) -> str:
    return (
        f"{prefix}_{scenario}_seed{seed}_"
        f"ts{_float_token(scale)}_tr{_float_token(reduction)}"
    )


def _tail_lines(text: str, max_lines: int = 20) -> list[str]:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    return lines[-max_lines:]


def _run_command(command: list[str], cwd: Path, dry_run: bool) -> CommandResult:
    if dry_run:
        return CommandResult(
            command=command,
            return_code=None,
            duration_seconds=0.0,
            skipped=True,
            stdout_tail=[],
            stderr_tail=[],
        )

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start

    return CommandResult(
        command=command,
        return_code=completed.returncode,
        duration_seconds=round(elapsed, 3),
        skipped=False,
        stdout_tail=_tail_lines(completed.stdout),
        stderr_tail=_tail_lines(completed.stderr),
    )


def _command_success(result: CommandResult) -> bool:
    return result.skipped or result.return_code == 0


def _load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _positive_count(rows: list[dict[str, str]], column_name: str) -> int:
    count = 0
    for row in rows:
        try:
            if int(float(row[column_name])) > 0:
                count += 1
        except Exception:
            continue
    return count


def _compute_processed_quality(processed_csv: Path, horizons: list[int]) -> dict[str, Any]:
    fieldnames, rows = _load_csv_rows(processed_csv)
    total_rows = len(rows)

    per_horizon: dict[str, Any] = {}
    missing_columns: list[str] = []
    for horizon in horizons:
        column_name = f"label_congestion_{horizon}s"
        if column_name not in fieldnames:
            per_horizon[str(horizon)] = {"column": column_name, "missing": True}
            missing_columns.append(column_name)
            continue

        positive_rows = _positive_count(rows, column_name)
        positive_rate = (positive_rows / total_rows) if total_rows > 0 else 0.0
        per_horizon[str(horizon)] = {
            "column": column_name,
            "positive_rows": positive_rows,
            "positive_rate": round(positive_rate, 6),
        }

    unique_ts = len({row.get("timestamp_s", "") for row in rows})
    unique_rsus = len({row.get("rsu_node", "") for row in rows if row.get("rsu_node")})

    return {
        "total_rows": total_rows,
        "unique_timestamps": unique_ts,
        "unique_rsu_nodes": unique_rsus,
        "missing_columns": missing_columns,
        "horizons": per_horizon,
    }


def _compute_split_quality(split_dir: Path, horizons: list[int]) -> dict[str, Any]:
    split_files = {
        "train": split_dir / "train.csv",
        "val": split_dir / "val.csv",
        "test": split_dir / "test.csv",
    }

    split_stats: dict[str, Any] = {}
    for split_name, split_path in split_files.items():
        fieldnames, rows = _load_csv_rows(split_path)
        horizon_stats: dict[str, Any] = {}

        for horizon in horizons:
            column_name = f"label_congestion_{horizon}s"
            if column_name not in fieldnames:
                horizon_stats[str(horizon)] = {"column": column_name, "missing": True}
                continue

            positive_rows = _positive_count(rows, column_name)
            total_rows = len(rows)
            positive_rate = (positive_rows / total_rows) if total_rows > 0 else 0.0
            horizon_stats[str(horizon)] = {
                "column": column_name,
                "positive_rows": positive_rows,
                "positive_rate": round(positive_rate, 6),
            }

        split_stats[split_name] = {
            "rows": len(rows),
            "horizons": horizon_stats,
        }

    leakage_report_path = split_dir / "leakage_report.json"
    leakage_passed = False
    leakage_report = None
    if leakage_report_path.exists():
        with leakage_report_path.open("r", encoding="utf-8") as handle:
            leakage_report = json.load(handle)
            leakage_passed = bool(leakage_report.get("passed", False))

    return {
        "splits": split_stats,
        "leakage_report_path": str(leakage_report_path),
        "leakage_passed": leakage_passed,
        "leakage_report": leakage_report,
    }


def _evaluate_quality(
    *,
    processed_quality: dict[str, Any],
    split_quality: dict[str, Any],
    min_rows: int,
    min_positive_rate: float,
    max_positive_rate: float,
    min_split_positive_rows: int,
    horizons: list[int],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    total_rows = int(processed_quality.get("total_rows", 0))
    checks["min_rows"] = total_rows >= min_rows
    details["min_rows"] = {"observed": total_rows, "required": min_rows}

    missing_columns = list(processed_quality.get("missing_columns", []))
    checks["required_columns_present"] = len(missing_columns) == 0
    details["required_columns_present"] = {"missing": missing_columns}

    rate_ok = True
    horizon_rates: dict[str, Any] = {}
    for horizon in horizons:
        key = str(horizon)
        horizon_stats = processed_quality.get("horizons", {}).get(key, {})
        if horizon_stats.get("missing", False):
            rate_ok = False
            horizon_rates[key] = {"missing": True}
            continue

        rate = float(horizon_stats.get("positive_rate", 0.0))
        in_range = min_positive_rate <= rate <= max_positive_rate
        if not in_range:
            rate_ok = False
        horizon_rates[key] = {
            "positive_rate": rate,
            "min": min_positive_rate,
            "max": max_positive_rate,
            "in_range": in_range,
        }

    checks["horizon_positive_rate_range"] = rate_ok
    details["horizon_positive_rate_range"] = horizon_rates

    split_ok = True
    split_details: dict[str, Any] = {}
    for split_name, split_info in split_quality.get("splits", {}).items():
        split_horizons = split_info.get("horizons", {})
        split_details[split_name] = {}
        for horizon in horizons:
            key = str(horizon)
            horizon_stats = split_horizons.get(key, {})
            if horizon_stats.get("missing", False):
                split_ok = False
                split_details[split_name][key] = {"missing": True}
                continue

            positive_rows = int(horizon_stats.get("positive_rows", 0))
            sufficient = positive_rows >= min_split_positive_rows
            if not sufficient:
                split_ok = False

            split_details[split_name][key] = {
                "positive_rows": positive_rows,
                "required": min_split_positive_rows,
                "sufficient": sufficient,
            }

    checks["split_positive_support"] = split_ok
    details["split_positive_support"] = split_details

    leakage_passed = bool(split_quality.get("leakage_passed", False))
    checks["leakage_passed"] = leakage_passed
    details["leakage_passed"] = {"observed": leakage_passed}

    passed = all(checks.values())
    return {"passed": passed, "checks": checks, "details": details}


def _as_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean for {field_name}")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(float(value))
    raise ValueError(f"Invalid value for {field_name}: {value}")


def _as_optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean for {field_name}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    raise ValueError(f"Invalid value for {field_name}: {value}")


def _resolve_quality_thresholds(
    *,
    scenario: str,
    defaults: dict[str, Any],
    quality_overrides: dict[str, Any],
) -> dict[str, Any]:
    thresholds = dict(defaults)

    def _apply_override(raw_override: Any) -> None:
        if raw_override is None:
            return
        if not isinstance(raw_override, dict):
            raise ValueError("quality_overrides entries must be objects")

        min_rows = _as_optional_int(raw_override.get("min_rows"), "min_rows")
        if min_rows is not None:
            thresholds["min_rows"] = max(0, int(min_rows))

        min_split_positive_rows = _as_optional_int(
            raw_override.get("min_split_positive_rows"),
            "min_split_positive_rows",
        )
        if min_split_positive_rows is not None:
            thresholds["min_split_positive_rows"] = max(0, int(min_split_positive_rows))

        min_positive_rate = _as_optional_float(raw_override.get("min_positive_rate"), "min_positive_rate")
        if min_positive_rate is not None:
            thresholds["min_positive_rate"] = float(min_positive_rate)

        max_positive_rate = _as_optional_float(raw_override.get("max_positive_rate"), "max_positive_rate")
        if max_positive_rate is not None:
            thresholds["max_positive_rate"] = float(max_positive_rate)

    _apply_override(quality_overrides.get("default"))
    _apply_override(quality_overrides.get(scenario))

    if thresholds["min_positive_rate"] > thresholds["max_positive_rate"]:
        raise ValueError(
            f"Invalid quality thresholds for scenario '{scenario}': min_positive_rate > max_positive_rate"
        )
    return thresholds


def _generate_run_specs(
    *,
    run_prefix: str,
    scenarios: list[str],
    seeds: list[int],
    traffic_scales: list[float],
    traffic_reduction_pcts: list[float],
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    seen_ids: set[str] = set()

    for scenario in scenarios:
        for seed in seeds:
            for scale in traffic_scales:
                for reduction in traffic_reduction_pcts:
                    run_id = _build_run_id(run_prefix, scenario, seed, scale, reduction)
                    if run_id in seen_ids:
                        raise ValueError(f"Duplicate run id generated: {run_id}")
                    seen_ids.add(run_id)
                    specs.append(
                        RunSpec(
                            run_id=run_id,
                            scenario=scenario,
                            seed=seed,
                            traffic_scale=scale,
                            traffic_reduction_pct=reduction,
                        )
                    )

    if not specs:
        raise ValueError("No run specs generated")
    return specs


def _build_sumo_command(
    *,
    python_executable: str,
    contract: str,
    spec: RunSpec,
    max_steps: int | None,
    dry_run: bool,
) -> list[str]:
    command = [
        python_executable,
        "sumo/run_sumo_pipeline.py",
        "--contract",
        contract,
        "--scenario",
        spec.scenario,
        "--seed",
        str(spec.seed),
        "--traffic-scale",
        str(spec.traffic_scale),
        "--traffic-reduction-pct",
        str(spec.traffic_reduction_pct),
        "--enable-runtime-logging",
        "--runtime-log-run-id",
        spec.run_id,
    ]
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if dry_run:
        command.append("--dry-run")
    return command


def _build_processing_commands(
    *,
    python_executable: str,
    run_id: str,
    horizons: str,
    gap_seconds: int,
) -> dict[str, list[str]]:
    raw_csv = f"data/raw/{run_id}/rsu_features_1hz.csv"
    processed_csv = f"data/processed/{run_id}/rsu_horizon_labels.csv"
    split_dir = f"data/splits/{run_id}"

    return {
        "label": [
            python_executable,
            "pipelines/processing/horizon_labeler.py",
            "--input-rsu",
            raw_csv,
            "--output",
            processed_csv,
            "--horizons",
            horizons,
        ],
        "split": [
            python_executable,
            "pipelines/processing/temporal_split.py",
            "--input",
            processed_csv,
            "--output-dir",
            split_dir,
            "--gap-seconds",
            str(gap_seconds),
        ],
        "leakage": [
            python_executable,
            "pipelines/processing/leakage_validator.py",
            "--split-dir",
            split_dir,
            "--expected-gap-seconds",
            str(gap_seconds),
        ],
    }


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except Exception:
        return str(path)


def _render_markdown_report(
    *,
    sweep_id: str,
    created_at: str,
    config_summary: dict[str, Any],
    run_records: list[dict[str, Any]],
) -> str:
    total_runs = len(run_records)
    generated_ok = sum(1 for record in run_records if record.get("sumo_ok", False))
    processed_ok = sum(1 for record in run_records if record.get("processing_ok", False))
    quality_evaluated = sum(1 for record in run_records if record.get("quality") is not None)
    quality_ok = sum(1 for record in run_records if record.get("quality_gate_passed") is True)

    lines: list[str] = []
    lines.append("# Phase-2 Data Sweep Report")
    lines.append("")
    lines.append(f"- Sweep id: {sweep_id}")
    lines.append(f"- Created (UTC): {created_at}")
    lines.append("- Execution policy: heavy training/evaluation remains local-only in this repository runtime.")
    lines.append("")
    lines.append("## Matrix")
    lines.append("")
    lines.append(f"- Scenarios: {', '.join(config_summary['scenarios'])}")
    lines.append(f"- Seeds: {', '.join(str(v) for v in config_summary['seeds'])}")
    lines.append(
        "- Traffic scales: "
        + ", ".join(str(v) for v in config_summary["traffic_scales"])
    )
    lines.append(
        "- Traffic reduction pcts: "
        + ", ".join(str(v) for v in config_summary["traffic_reduction_pcts"])
    )
    lines.append(f"- Horizons: {', '.join(str(v) for v in config_summary['horizons'])}")
    lines.append(f"- Gap seconds: {config_summary['gap_seconds']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Runs planned: {total_runs}")
    lines.append(f"- SUMO stage success: {generated_ok}/{total_runs}")
    lines.append(f"- Processing stage success: {processed_ok}/{total_runs}")
    lines.append(f"- Quality gates passed: {quality_ok}/{quality_evaluated} (evaluated runs)")
    lines.append("")
    lines.append("## Per-Run Status")
    lines.append("")
    lines.append("| Run ID | SUMO | Processing | Quality Gate | Notes |")
    lines.append("|---|---:|---:|---:|---|")

    for record in run_records:
        notes = []
        if record.get("missing_raw"):
            notes.append("missing raw rsu_features_1hz.csv")
        if record.get("quality_gate_passed") is False and record.get("quality"):
            failed_checks = [
                name for name, ok in record["quality"]["checks"].items() if not ok
            ]
            if failed_checks:
                notes.append("failed checks: " + ", ".join(failed_checks))

        lines.append(
            "| {run_id} | {sumo} | {processing} | {quality} | {notes} |".format(
                run_id=record["run_id"],
                sumo="PASS" if record.get("sumo_ok", False) else "FAIL",
                processing="PASS" if record.get("processing_ok", False) else "FAIL",
                quality=(
                    "N/A"
                    if record.get("quality") is None
                    else ("PASS" if record.get("quality_gate_passed") is True else "FAIL")
                ),
                notes="; ".join(notes) if notes else "",
            )
        )

    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append(
        "Use run IDs with passing quality gates as training input for Phase-2 baselines in local runtime."
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        raise ValueError(f"Project root does not exist: {project_root}")

    config_path = Path(args.config).resolve() if args.config else None
    config_payload = _load_optional_config(config_path)

    scenarios = [str(item) for item in _resolve_from_config(config_payload, "scenarios", _parse_csv(args.scenarios))]
    seeds = [int(item) for item in _resolve_from_config(config_payload, "seeds", _parse_int_csv(args.seeds))]
    traffic_scales = [
        float(item) for item in _resolve_from_config(config_payload, "traffic_scales", _parse_float_csv(args.traffic_scales))
    ]
    traffic_reduction_pcts = [
        float(item)
        for item in _resolve_from_config(
            config_payload,
            "traffic_reduction_pcts",
            _parse_float_csv(args.traffic_reduction_pcts),
        )
    ]
    quality_overrides = config_payload.get("quality_overrides", {})
    if quality_overrides is None:
        quality_overrides = {}
    if not isinstance(quality_overrides, dict):
        raise ValueError("Config field 'quality_overrides' must be an object")

    horizons = _parse_horizons(args.horizons)
    if args.gap_seconds < 0:
        raise ValueError("gap-seconds must be >= 0")

    sweep_id = args.sweep_id or _default_sweep_id(args.run_prefix)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_dir = output_root / sweep_id
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report_path)
    if not report_path.is_absolute():
        report_path = project_root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    contract_path = Path(args.contract)
    if not contract_path.is_absolute():
        contract_path = project_root / contract_path

    run_specs = _generate_run_specs(
        run_prefix=args.run_prefix,
        scenarios=scenarios,
        seeds=seeds,
        traffic_scales=traffic_scales,
        traffic_reduction_pcts=traffic_reduction_pcts,
    )

    run_records: list[dict[str, Any]] = []
    quality_defaults = {
        "min_rows": int(args.min_rows),
        "min_positive_rate": float(args.min_positive_rate),
        "max_positive_rate": float(args.max_positive_rate),
        "min_split_positive_rows": int(args.min_split_positive_rows),
    }
    for spec in run_specs:
        record: dict[str, Any] = {
            "run_id": spec.run_id,
            "spec": asdict(spec),
            "sumo": None,
            "sumo_ok": False,
            "processing": {},
            "processing_ok": False,
            "missing_raw": False,
            "quality": None,
            "quality_gate_passed": None,
        }

        if args.skip_sumo:
            record["sumo"] = {
                "skipped": True,
                "reason": "--skip-sumo requested",
            }
            record["sumo_ok"] = True
        else:
            sumo_cmd = _build_sumo_command(
                python_executable=args.python_executable,
                contract=str(contract_path),
                spec=spec,
                max_steps=args.max_steps,
                dry_run=args.dry_run,
            )
            sumo_result = _run_command(sumo_cmd, project_root, args.dry_run)
            record["sumo"] = asdict(sumo_result)
            record["sumo_ok"] = _command_success(sumo_result)

        raw_csv = project_root / "data" / "raw" / spec.run_id / "rsu_features_1hz.csv"

        # Dry-run mode validates command assembly only; no processing commands are executed.
        if args.dry_run:
            record["processing"] = {"skipped": True, "reason": "--dry-run requested"}
            record["processing_ok"] = True
            run_records.append(record)
            continue

        if not raw_csv.exists():
            record["missing_raw"] = True
            record["processing"] = {
                "skipped": True,
                "reason": f"missing raw file: {_relative_to_project(raw_csv, project_root)}",
            }
            run_records.append(record)
            continue

        processing_commands = _build_processing_commands(
            python_executable=args.python_executable,
            run_id=spec.run_id,
            horizons=args.horizons,
            gap_seconds=args.gap_seconds,
        )

        step_results: dict[str, Any] = {}
        processing_ok = True
        for step_name, command in processing_commands.items():
            step_result = _run_command(command, project_root, dry_run=False)
            step_results[step_name] = asdict(step_result)
            if not _command_success(step_result):
                processing_ok = False
                break

        record["processing"] = step_results
        record["processing_ok"] = processing_ok

        if processing_ok:
            processed_csv = project_root / "data" / "processed" / spec.run_id / "rsu_horizon_labels.csv"
            split_dir = project_root / "data" / "splits" / spec.run_id
            processed_quality = _compute_processed_quality(processed_csv, horizons)
            split_quality = _compute_split_quality(split_dir, horizons)
            scenario_thresholds = _resolve_quality_thresholds(
                scenario=spec.scenario,
                defaults=quality_defaults,
                quality_overrides=quality_overrides,
            )
            quality = _evaluate_quality(
                processed_quality=processed_quality,
                split_quality=split_quality,
                min_rows=int(scenario_thresholds["min_rows"]),
                min_positive_rate=float(scenario_thresholds["min_positive_rate"]),
                max_positive_rate=float(scenario_thresholds["max_positive_rate"]),
                min_split_positive_rows=int(scenario_thresholds["min_split_positive_rows"]),
                horizons=horizons,
            )
            record["quality"] = {
                "processed": processed_quality,
                "split": split_quality,
                "thresholds": scenario_thresholds,
                "checks": quality["checks"],
                "details": quality["details"],
                "passed": quality["passed"],
            }
            record["quality_gate_passed"] = bool(quality["passed"])

        run_records.append(record)

    created_at = _now_iso()
    config_summary = {
        "scenarios": scenarios,
        "seeds": seeds,
        "traffic_scales": traffic_scales,
        "traffic_reduction_pcts": traffic_reduction_pcts,
        "horizons": horizons,
        "gap_seconds": args.gap_seconds,
        "min_rows": args.min_rows,
        "min_positive_rate": args.min_positive_rate,
        "max_positive_rate": args.max_positive_rate,
        "min_split_positive_rows": args.min_split_positive_rows,
        "quality_overrides": quality_overrides,
        "skip_sumo": args.skip_sumo,
        "dry_run": args.dry_run,
    }

    manifest = {
        "schema_version": "1.0",
        "sweep_id": sweep_id,
        "created_at_utc": created_at,
        "execution_policy": {
            "training_location": "local_only",
            "heavy_training_location": "local_only",
            "kaggle_required": False,
        },
        "project_root": str(project_root),
        "contract": str(contract_path),
        "python_executable": args.python_executable,
        "config": config_summary,
        "runs": run_records,
    }

    manifest_path = output_dir / "phase2_data_sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    report_content = _render_markdown_report(
        sweep_id=sweep_id,
        created_at=created_at,
        config_summary=config_summary,
        run_records=run_records,
    )
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(report_content)

    print(f"[phase2-sweep] manifest: {manifest_path}")
    print(f"[phase2-sweep] report: {report_path}")

    failures = []
    for record in run_records:
        if not record.get("sumo_ok", False):
            failures.append((record["run_id"], "sumo"))
            continue
        if not record.get("processing_ok", False):
            failures.append((record["run_id"], "processing"))
            continue
        if record.get("quality_gate_passed") is False and record.get("quality") is not None:
            failures.append((record["run_id"], "quality"))

    if args.strict and failures:
        details = ", ".join(f"{run_id}:{stage}" for run_id, stage in failures)
        raise SystemExit(f"[phase2-sweep] strict mode failure: {details}")


if __name__ == "__main__":
    main()
