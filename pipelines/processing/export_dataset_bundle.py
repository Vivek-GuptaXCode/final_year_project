from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import tarfile
from typing import Any


DEFAULT_REQUIRED_HORIZONS = "60,120"
DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a phase-1 dataset export bundle with unified manifest and quality report."
        )
    )
    parser.add_argument(
        "--run-ids",
        required=True,
        help="Comma-separated run ids (example: smoke_phase1_logger,smoke_phase1_logger_seed43).",
    )
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Project root directory (default: repository root inferred from this script).",
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data root (default: data/raw).")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Processed data root (default: data/processed).",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Split data root (default: data/splits).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/exports",
        help="Output root for manifest and bundle (default: data/exports).",
    )
    parser.add_argument(
        "--export-id",
        default=None,
        help="Optional export id (default: auto UTC timestamp).",
    )
    parser.add_argument(
        "--bundle-name",
        default="phase1_dataset_bundle.tar.gz",
        help="Archive filename (default: phase1_dataset_bundle.tar.gz).",
    )
    parser.add_argument(
        "--manifest-name",
        default="phase1_dataset_manifest.json",
        help="Unified manifest filename (default: phase1_dataset_manifest.json).",
    )
    parser.add_argument(
        "--report-path",
        default="docs/reports/phase1_data_report.md",
        help="Markdown report path relative to project root.",
    )
    parser.add_argument(
        "--schema-version",
        default="1.1",
        help="Unified manifest schema version (default: 1.1).",
    )
    parser.add_argument(
        "--require-horizons",
        default=DEFAULT_REQUIRED_HORIZONS,
        help="Required horizon labels in seconds (default: 60,120).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required artifact is missing for a selected run.",
    )
    return parser.parse_args()


def _parse_csv_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one value must be provided")
    return values


def _parse_required_horizons(raw: str) -> list[int]:
    horizons: list[int] = []
    for item in _parse_csv_list(raw):
        value = int(item)
        if value <= 0:
            raise ValueError("Horizon values must be positive integers")
        horizons.append(value)
    return sorted(set(horizons))


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_export_id() -> str:
    return datetime.now(timezone.utc).strftime("phase1_export_%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(handle, "sha256").hexdigest()

        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_int(value: str, field_name: str) -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"Invalid integer-like value for {field_name}: {value}") from exc


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV header missing: {path}")
        return sum(1 for _ in reader)


def _read_csv_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV header missing: {path}")
        return list(reader.fieldnames)


def _extract_seed_from_run_id(run_id: str) -> int | None:
    match = re.search(r"seed(\d+)", run_id)
    if not match:
        return None
    return int(match.group(1))


def _resolve_manifest_run_meta(raw_manifest: dict[str, Any], run_id: str) -> tuple[str | None, int | None]:
    run_meta = raw_manifest.get("run", {}) if isinstance(raw_manifest, dict) else {}
    scenario = run_meta.get("scenario") if isinstance(run_meta, dict) else None
    seed = run_meta.get("seed") if isinstance(run_meta, dict) else None

    if isinstance(seed, str):
        try:
            seed = int(seed)
        except Exception:
            seed = None
    if not isinstance(seed, int):
        seed = _extract_seed_from_run_id(run_id)

    if scenario is not None:
        scenario = str(scenario)

    return scenario, seed


def _file_info(path: Path, project_root: Path) -> dict[str, Any]:
    relative = path.relative_to(project_root)
    return {
        "path": str(relative),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _collect_label_stats(processed_csv: Path, required_horizons: list[int]) -> tuple[dict[str, Any], bool]:
    required_columns = [f"label_congestion_{h}s" for h in required_horizons]

    with processed_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV header missing: {processed_csv}")
        fieldnames = list(reader.fieldnames)

        missing_columns = [name for name in required_columns if name not in fieldnames]
        present_columns = [name for name in required_columns if name in fieldnames]

        positives = {name: 0 for name in present_columns}
        total_rows = 0
        for row in reader:
            total_rows += 1
            for name in present_columns:
                positives[name] += 1 if _to_int(row[name], name) > 0 else 0

    horizons: dict[str, Any] = {}
    for column_name in required_columns:
        horizon_token = column_name.replace("label_congestion_", "").replace("s", "")
        if column_name in positives:
            positive_rows = positives[column_name]
            positive_rate = (positive_rows / total_rows) if total_rows > 0 else 0.0
            horizons[horizon_token] = {
                "column": column_name,
                "positive_rows": positive_rows,
                "positive_rate": round(positive_rate, 6),
            }
        else:
            horizons[horizon_token] = {
                "column": column_name,
                "missing": True,
            }

    return (
        {
            "total_rows": total_rows,
            "required_columns": required_columns,
            "missing_columns": missing_columns,
            "horizons": horizons,
        },
        len(missing_columns) == 0,
    )


def _resolve_expected_files(run_id: str, roots: dict[str, Path]) -> dict[str, Path]:
    raw_dir = roots["raw"] / run_id
    processed_dir = roots["processed"] / run_id
    split_dir = roots["splits"] / run_id

    return {
        "raw_rsu_features": raw_dir / "rsu_features_1hz.csv",
        "raw_edge_flow": raw_dir / "edge_flow_1hz.csv",
        "raw_logger_manifest": raw_dir / "logger_manifest.json",
        "processed_labels": processed_dir / "rsu_horizon_labels.csv",
        "split_train": split_dir / "train.csv",
        "split_val": split_dir / "val.csv",
        "split_test": split_dir / "test.csv",
        "split_manifest": split_dir / "split_manifest.json",
        "leakage_report": split_dir / "leakage_report.json",
    }


def _collect_run_record(
    *,
    run_id: str,
    roots: dict[str, Path],
    project_root: Path,
    required_horizons: list[int],
    strict: bool,
) -> tuple[dict[str, Any], list[Path]]:
    expected = _resolve_expected_files(run_id, roots)
    missing_files: list[str] = []
    for key, path in expected.items():
        if not path.exists():
            missing_files.append(key)

    if strict and missing_files:
        raise FileNotFoundError(f"Run {run_id} missing required files: {missing_files}")

    run_files: dict[str, Any] = {}
    bundle_members: list[Path] = []
    for key, path in expected.items():
        if path.exists():
            run_files[key] = _file_info(path, project_root)
            bundle_members.append(path)

    scenario = None
    seed = None
    if expected["raw_logger_manifest"].exists():
        raw_manifest = _load_json(expected["raw_logger_manifest"])
        scenario, seed = _resolve_manifest_run_meta(raw_manifest, run_id)

    label_stats: dict[str, Any] = {
        "total_rows": 0,
        "required_columns": [f"label_congestion_{h}s" for h in required_horizons],
        "missing_columns": [f"label_congestion_{h}s" for h in required_horizons],
        "horizons": {},
    }
    horizons_ok = False
    if expected["processed_labels"].exists():
        label_stats, horizons_ok = _collect_label_stats(expected["processed_labels"], required_horizons)

    split_stats: dict[str, Any] = {
        "rows": {
            "train": None,
            "val": None,
            "test": None,
        },
        "gap_seconds": None,
        "hashes": {},
    }
    if expected["split_manifest"].exists():
        split_manifest = _load_json(expected["split_manifest"])
        split_stats = {
            "rows": split_manifest.get("rows", split_stats["rows"]),
            "gap_seconds": split_manifest.get("gap_seconds"),
            "hashes": split_manifest.get("hashes", {}),
        }
    else:
        for key in ("split_train", "split_val", "split_test"):
            if expected[key].exists():
                row_key = key.replace("split_", "")
                split_stats["rows"][row_key] = _count_csv_rows(expected[key])

    leakage_passed = False
    if expected["leakage_report"].exists():
        leakage_report = _load_json(expected["leakage_report"])
        leakage_passed = bool(leakage_report.get("passed", False))

    required_files_ok = len(missing_files) == 0
    gate_ready = required_files_ok and horizons_ok and leakage_passed

    run_record: dict[str, Any] = {
        "run_id": run_id,
        "scenario": scenario,
        "seed": seed,
        "files": run_files,
        "missing_files": missing_files,
        "quality": {
            "labels": label_stats,
            "splits": split_stats,
            "leakage_passed": leakage_passed,
        },
        "checks": {
            "required_files_present": required_files_ok,
            "required_horizons_present": horizons_ok,
            "leakage_passed": leakage_passed,
            "phase1_gate_ready": gate_ready,
        },
    }
    return run_record, bundle_members


def _tar_member_filter(tar_info: tarfile.TarInfo) -> tarfile.TarInfo:
    # Normalize metadata for stable archive output across machines.
    tar_info.uid = 0
    tar_info.gid = 0
    tar_info.uname = "root"
    tar_info.gname = "root"
    tar_info.mtime = 0
    return tar_info


def _build_bundle(
    *,
    project_root: Path,
    bundle_path: Path,
    data_paths: list[Path],
) -> tuple[list[str], dict[str, Any]]:
    unique_paths = sorted(set(data_paths), key=lambda p: str(p.relative_to(project_root)))
    members: list[str] = []

    with tarfile.open(bundle_path, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in unique_paths:
            arcname = str(path.relative_to(project_root))
            archive.add(path, arcname=arcname, filter=_tar_member_filter)
            members.append(arcname)

    bundle_info = {
        "path": str(bundle_path.relative_to(project_root)),
        "bytes": bundle_path.stat().st_size,
        "sha256": _sha256_file(bundle_path),
        "member_count": len(members),
    }
    return members, bundle_info


def _summarize_runs(run_records: list[dict[str, Any]], required_horizons: list[int]) -> dict[str, Any]:
    gate_ready_runs = sum(1 for run in run_records if run["checks"].get("phase1_gate_ready", False))
    leakage_pass_runs = sum(1 for run in run_records if run["checks"].get("leakage_passed", False))

    total_label_rows = 0
    horizon_positive_totals: dict[str, int] = {str(h): 0 for h in required_horizons}
    horizon_positive_rates: dict[str, float] = {}

    missing_files_total = 0
    for run in run_records:
        labels = run.get("quality", {}).get("labels", {})
        total_rows = int(labels.get("total_rows", 0) or 0)
        total_label_rows += total_rows

        missing_files_total += len(run.get("missing_files", []))

        horizons = labels.get("horizons", {})
        for horizon in required_horizons:
            token = str(horizon)
            horizon_info = horizons.get(token, {})
            positive_rows = int(horizon_info.get("positive_rows", 0) or 0)
            horizon_positive_totals[token] += positive_rows

    for token, positives in horizon_positive_totals.items():
        horizon_positive_rates[token] = round((positives / total_label_rows), 6) if total_label_rows > 0 else 0.0

    return {
        "run_count": len(run_records),
        "gate_ready_runs": gate_ready_runs,
        "leakage_pass_runs": leakage_pass_runs,
        "missing_files_total": missing_files_total,
        "total_labeled_rows": total_label_rows,
        "horizon_positive_rows": horizon_positive_totals,
        "horizon_positive_rates": horizon_positive_rates,
    }


def _build_markdown_report(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 1 Data Report")
    lines.append("")
    lines.append(f"- Generated UTC: {manifest['created_at_utc']}")
    lines.append(f"- Manifest schema version: {manifest['schema_version']}")
    lines.append(f"- Required horizons: {', '.join(str(h) for h in manifest['required_horizons'])}")
    lines.append(f"- Bundle: {manifest['bundle']['path']}")
    lines.append(f"- Bundle SHA256: {manifest['bundle']['sha256']}")
    lines.append("")

    summary = manifest["summary"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Runs selected: {summary['run_count']}")
    lines.append(f"- Runs gate-ready: {summary['gate_ready_runs']}")
    lines.append(f"- Runs leakage PASS: {summary['leakage_pass_runs']}")
    lines.append(f"- Missing artifact entries: {summary['missing_files_total']}")
    lines.append(f"- Total labeled rows: {summary['total_labeled_rows']}")
    lines.append("")

    lines.append("## Horizon Balance")
    lines.append("")
    for horizon in manifest["required_horizons"]:
        token = str(horizon)
        rows = summary["horizon_positive_rows"].get(token, 0)
        rate = summary["horizon_positive_rates"].get(token, 0.0)
        lines.append(f"- Horizon {token}s: positive_rows={rows}, positive_rate={rate}")
    lines.append("")

    lines.append("## Per-Run Status")
    lines.append("")
    lines.append("| run_id | scenario | seed | required_files | horizons | leakage | gate_ready |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for run in manifest["runs"]:
        checks = run.get("checks", {})
        lines.append(
            "| {run_id} | {scenario} | {seed} | {files_ok} | {horizons_ok} | {leakage_ok} | {gate_ok} |".format(
                run_id=run.get("run_id", ""),
                scenario=run.get("scenario", "unknown") or "unknown",
                seed=run.get("seed", "unknown") if run.get("seed", None) is not None else "unknown",
                files_ok="PASS" if checks.get("required_files_present", False) else "FAIL",
                horizons_ok="PASS" if checks.get("required_horizons_present", False) else "FAIL",
                leakage_ok="PASS" if checks.get("leakage_passed", False) else "FAIL",
                gate_ok="PASS" if checks.get("phase1_gate_ready", False) else "FAIL",
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This report is generated from local CSV and JSON artifacts only.")
    lines.append("- File hashes are recorded in the unified manifest for reproducibility.")
    lines.append("- Any missing artifacts are listed under each run in the manifest.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    roots = {
        "raw": (project_root / args.raw_dir).resolve(),
        "processed": (project_root / args.processed_dir).resolve(),
        "splits": (project_root / args.splits_dir).resolve(),
    }

    run_ids = _parse_csv_list(args.run_ids)
    required_horizons = _parse_required_horizons(args.require_horizons)

    output_root = (project_root / args.output_dir).resolve()
    export_id = args.export_id or _default_export_id()
    export_dir = output_root / export_id
    export_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = export_dir / args.manifest_name
    bundle_path = export_dir / args.bundle_name

    report_path = (project_root / args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    bundle_sources: list[Path] = []

    for run_id in run_ids:
        run_record, run_bundle_files = _collect_run_record(
            run_id=run_id,
            roots=roots,
            project_root=project_root,
            required_horizons=required_horizons,
            strict=args.strict,
        )
        run_records.append(run_record)
        bundle_sources.extend(run_bundle_files)

    bundle_members, bundle_info = _build_bundle(
        project_root=project_root,
        bundle_path=bundle_path,
        data_paths=bundle_sources,
    )

    summary = _summarize_runs(run_records, required_horizons)
    manifest = {
        "schema_version": args.schema_version,
        "created_at_utc": _iso_utc_now(),
        "required_horizons": required_horizons,
        "runs": run_records,
        "summary": summary,
        "bundle": {
            **bundle_info,
            "members": bundle_members,
        },
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report_md = _build_markdown_report(manifest)
    report_path.write_text(report_md + "\n", encoding="utf-8")

    print(f"[export] wrote manifest: {manifest_path.relative_to(project_root)}")
    print(f"[export] wrote bundle: {bundle_path.relative_to(project_root)}")
    print(f"[export] wrote report: {report_path.relative_to(project_root)}")
    print(
        "[export] runs={runs} gate_ready={ready} leakage_pass={leakage} total_rows={rows}".format(
            runs=summary["run_count"],
            ready=summary["gate_ready_runs"],
            leakage=summary["leakage_pass_runs"],
            rows=summary["total_labeled_rows"],
        )
    )


if __name__ == "__main__":
    main()
