"""Phase 4 RL ablation runner.

Runs a compact set of research-backed ablations around:
- stronger DQN architecture
- phase-competition observation features
- pressure-informed reward shaping

Each variant delegates training/evaluation to train_phase4.py and aggregates
the resulting KPI files into one study summary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    extra_args: tuple[str, ...]


VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant(
        name="basic_queue_only",
        description="Baseline DQN without pressure reward and without phase-competition features.",
        extra_args=(
            "--force-basic-dqn",
            "--reward-pressure-weight", "0.0",
            "--reward-throughput-weight", "0.4",
            "--reward-waiting-weight", "0.05",
            "--disable-phase-competition-features",
        ),
    ),
    AblationVariant(
        name="improved_queue_only",
        description="Dueling Double DQN + PER, but no pressure reward and no phase-competition features.",
        extra_args=(
            "--use-improved-dqn",
            "--reward-pressure-weight", "0.0",
            "--reward-throughput-weight", "0.4",
            "--reward-waiting-weight", "0.05",
            "--disable-phase-competition-features",
        ),
    ),
    AblationVariant(
        name="improved_queue_phase_comp",
        description="Improved agent with FRAP-style current-vs-next phase queue features, still queue-only reward.",
        extra_args=(
            "--use-improved-dqn",
            "--reward-pressure-weight", "0.0",
            "--reward-throughput-weight", "0.4",
            "--reward-waiting-weight", "0.05",
        ),
    ),
    AblationVariant(
        name="improved_hybrid_pressure",
        description="Improved agent + phase-competition features + pressure-informed hybrid reward.",
        extra_args=(
            "--use-improved-dqn",
            "--reward-pressure-weight", "0.6",
            "--reward-throughput-weight", "0.3",
            "--reward-waiting-weight", "0.15",
        ),
    ),
    AblationVariant(
        name="improved_reference_warm_start",
        description="Improved agent + max-pressure demonstrations + decaying reference guidance during early training.",
        extra_args=(
            "--use-improved-dqn",
            "--reward-pressure-weight", "0.3",
            "--reward-throughput-weight", "0.4",
            "--reward-waiting-weight", "0.1",
            "--reference-policy", "max_pressure",
            "--reference-demo-episodes", "2",
            "--reference-pretrain-updates", "160",
            "--reference-prob-start", "0.6",
            "--reference-prob-end", "0.0",
            "--reference-prob-decay-episodes", "3",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL traffic-signal ablation study")
    parser.add_argument("--scenario", default="city", help="Scenario passed to train_phase4.py")
    parser.add_argument("--episodes", type=int, default=6, help="Episodes per ablation run")
    parser.add_argument("--steps-per-episode", type=int, default=600, help="Steps per episode")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Evaluation episodes per run")
    parser.add_argument("--decision-interval-steps", type=int, default=5, help="Decision interval")
    parser.add_argument("--train-every", type=int, default=4, help="Gradient update cadence in decisions")
    parser.add_argument("--train-updates-per-step", type=int, default=1, help="Gradient steps per update point")
    parser.add_argument("--train-tls-limit", type=int, default=None, help="Optional cap on training junctions")
    parser.add_argument("--seeds", default="42,43", help="Comma-separated random seeds")
    parser.add_argument(
        "--output-dir",
        default="evaluation/rl_ablation_latest",
        help="Directory for ablation result JSON files",
    )
    return parser.parse_args()


def _run_variant(
    variant: AblationVariant,
    *,
    scenario: str,
    episodes: int,
    steps_per_episode: int,
    eval_episodes: int,
    decision_interval_steps: int,
    train_every: int,
    train_updates_per_step: int,
    train_tls_limit: int | None,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    rel_artifact_dir = Path("models/rl/artifacts") / "rl_ablation" / f"{variant.name}_seed{seed}"
    result_rel_path = output_root.relative_to(_REPO_ROOT) / f"{variant.name}_seed{seed}.json"

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "controllers" / "rl" / "train_phase4.py"),
        "--scenario", scenario,
        "--train-all-tls",
        "--episodes", str(episodes),
        "--steps-per-episode", str(steps_per_episode),
        "--eval-episodes", str(eval_episodes),
        "--decision-interval-steps", str(decision_interval_steps),
        "--train-every", str(train_every),
        "--train-updates-per-step", str(train_updates_per_step),
        "--seed", str(seed),
        "--output-dir", str(rel_artifact_dir),
        "--results-path", str(result_rel_path),
    ]
    if train_tls_limit is not None and train_tls_limit > 0:
        cmd.extend(["--train-tls-limit", str(train_tls_limit)])
    cmd.extend(list(variant.extra_args))

    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)
    wall_time_s = time.perf_counter() - t0

    result_path = output_root / f"{variant.name}_seed{seed}.json"
    data = json.loads(result_path.read_text())
    data["ablation"] = {
        "name": variant.name,
        "description": variant.description,
        "seed": seed,
        "wall_time_s": round(wall_time_s, 2),
    }
    return data


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["ablation"]["name"], []).append(result)

    summary: dict[str, Any] = {}
    baseline_halting = None

    for variant_name, runs in grouped.items():
        dqn_halting = [float(run["single_agent_eval"]["dqn"]["mean_halting"]) for run in runs]
        fixed_halting = [float(run["single_agent_eval"]["fixed_time"]["mean_halting"]) for run in runs]
        simple_halting = [float(run["single_agent_eval"]["simple_actuated"]["mean_halting"]) for run in runs]
        max_pressure_halting = [float(run["single_agent_eval"]["max_pressure"]["mean_halting"]) for run in runs]
        marl_reward = [float(run["marl_eval"]["mean_reward_across_eps"]) for run in runs]

        entry = {
            "runs": len(runs),
            "description": runs[0]["ablation"]["description"],
            "dqn_mean_halting": round(float(np.mean(dqn_halting)), 4),
            "dqn_halting_std": round(float(np.std(dqn_halting)), 4),
            "fixed_time_mean_halting": round(float(np.mean(fixed_halting)), 4),
            "simple_actuated_mean_halting": round(float(np.mean(simple_halting)), 4),
            "max_pressure_mean_halting": round(float(np.mean(max_pressure_halting)), 4),
            "marl_mean_reward": round(float(np.mean(marl_reward)), 4),
            "p42_pass_rate": round(
                float(
                    np.mean(
                        [
                            1.0
                            if str(run["gates"]["P4.2"]).startswith("PASS")
                            else 0.0
                            for run in runs
                        ]
                    )
                ),
                3,
            ),
        }
        summary[variant_name] = entry
        if variant_name == "basic_queue_only":
            baseline_halting = entry["dqn_mean_halting"]

    if baseline_halting and baseline_halting > 0:
        for entry in summary.values():
            improvement = (baseline_halting - entry["dqn_mean_halting"]) / baseline_halting * 100.0
            entry["halting_improvement_vs_basic_pct"] = round(float(improvement), 2)

    ranking = sorted(
        (
            {"variant": name, **entry}
            for name, entry in summary.items()
        ),
        key=lambda item: item["dqn_mean_halting"],
    )

    return {
        "variants": summary,
        "ranking_by_dqn_halting": ranking,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]

    output_root = _REPO_ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[RL-ABL] Running {len(VARIANTS)} variants × {len(seeds)} seeds "
        f"on scenario={args.scenario}"
    )

    all_results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        print(f"[RL-ABL] Variant: {variant.name}")
        for seed in seeds:
            print(f"[RL-ABL]   seed={seed}")
            result = _run_variant(
                variant,
                scenario=args.scenario,
                episodes=args.episodes,
                steps_per_episode=args.steps_per_episode,
                eval_episodes=args.eval_episodes,
                decision_interval_steps=args.decision_interval_steps,
                train_every=args.train_every,
                train_updates_per_step=args.train_updates_per_step,
                train_tls_limit=args.train_tls_limit,
                seed=seed,
                output_root=output_root,
            )
            all_results.append(result)

    summary = {
        "meta": {
            "scenario": args.scenario,
            "episodes": args.episodes,
            "steps_per_episode": args.steps_per_episode,
            "eval_episodes": args.eval_episodes,
            "decision_interval_steps": args.decision_interval_steps,
            "train_every": args.train_every,
            "train_updates_per_step": args.train_updates_per_step,
            "seeds": seeds,
        },
        "results": _aggregate(all_results),
        "raw_runs": all_results,
    }

    summary_path = output_root / "rl_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[RL-ABL] Summary → {summary_path}")

    ranking = summary["results"]["ranking_by_dqn_halting"]
    for item in ranking:
        print(
            f"[RL-ABL] {item['variant']:26s} "
            f"dqn_halting={item['dqn_mean_halting']:.4f} "
            f"fixed={item['fixed_time_mean_halting']:.4f} "
            f"max_pressure={item['max_pressure_mean_halting']:.4f}"
        )


if __name__ == "__main__":
    main()
