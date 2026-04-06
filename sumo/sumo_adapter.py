from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import sys
from typing import Callable


@dataclass(frozen=True)
class SumoScenarioConfig:
    scenario: str
    sumocfg_path: Path
    step_length_seconds: float
    default_max_steps: int
    stop_when_no_vehicles: bool
    sumo_binary: str
    sumo_gui_binary: str
    prefer_libsumo: bool
    gui_settings_path: Path | None
    gui_use_osg_view: bool


def _load_contract(contract_path: Path) -> dict:
    with contract_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_scenario_config(contract_path: str | Path, scenario_name: str) -> SumoScenarioConfig:
    contract_path = Path(contract_path)
    contract = _load_contract(contract_path)

    scenarios = contract.get("scenarios", {})
    if scenario_name not in scenarios:
        available = ", ".join(sorted(scenarios.keys()))
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")

    execution = contract.get("execution", {})
    runner = contract.get("runner", {})
    scenario = scenarios[scenario_name]

    sumocfg_path = Path(scenario["sumocfg"])
    if not sumocfg_path.is_absolute():
        sumocfg_path = contract_path.parent.parent.parent / sumocfg_path

    return SumoScenarioConfig(
        scenario=scenario_name,
        sumocfg_path=sumocfg_path,
        step_length_seconds=float(execution.get("step_length_seconds", 1.0)),
        default_max_steps=int(execution.get("default_max_steps", 3600)),
        stop_when_no_vehicles=bool(execution.get("stop_when_no_vehicles", True)),
        sumo_binary=runner.get("sumo_binary", "sumo"),
        sumo_gui_binary=runner.get("sumo_gui_binary", "sumo-gui"),
        prefer_libsumo=bool(runner.get("prefer_libsumo", True)),
        gui_settings_path=(
            contract_path.parent.parent.parent / scenario["gui_settings"]
            if scenario.get("gui_settings")
            else None
        ),
        gui_use_osg_view=bool(scenario.get("gui_use_osg_view", False)),
    )


def build_sumo_command(
    config: SumoScenarioConfig,
    seed: int,
    use_gui: bool,
    force_3d: bool = False,
    additional_files: list[Path] | None = None,
    route_files: list[Path] | None = None,
    scale: float = 1.0,
    junction_taz: bool = False,
    statistics_output_path: Path | None = None,
    summary_output_path: Path | None = None,
    tripinfo_output_path: Path | None = None,
    tripinfo_write_unfinished: bool = False,
) -> list[str]:
    binary = config.sumo_gui_binary if use_gui else config.sumo_binary
    command = [
        binary,
        "-c",
        str(config.sumocfg_path),
        "--seed",
        str(seed),
        "--step-length",
        str(config.step_length_seconds),
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]

    if use_gui and config.gui_settings_path is not None:
        command += ["--gui-settings-file", str(config.gui_settings_path)]

    if route_files:
        command += ["--route-files", ",".join(str(p) for p in route_files)]

    if additional_files:
        command += ["--additional-files", ",".join(str(p) for p in additional_files)]

    if scale != 1.0:
        command += ["--scale", str(scale)]

    if junction_taz:
        command += ["--junction-taz", "true"]

    if statistics_output_path is not None:
        command += ["--statistic-output", str(statistics_output_path)]

    if summary_output_path is not None:
        command += ["--summary-output", str(summary_output_path)]

    if tripinfo_output_path is not None:
        command += ["--tripinfo-output", str(tripinfo_output_path)]
        if tripinfo_write_unfinished:
            command += ["--tripinfo-output.write-unfinished", "true"]

    if use_gui and (config.gui_use_osg_view or force_3d):
        command += ["--osg-view", "true"]

    return command


def _import_traci(prefer_libsumo: bool):
    if prefer_libsumo:
        try:
            import libsumo as traci  # type: ignore

            return traci
        except Exception:
            pass

    try:
        import traci  # type: ignore

        return traci
    except Exception:
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            tools_dir = Path(sumo_home) / "tools"
            if str(tools_dir) not in sys.path:
                sys.path.append(str(tools_dir))
            import traci  # type: ignore

            return traci
        raise


class SumoAdapter:
    """Thin SUMO TraCI/libsumo wrapper with a deterministic step-loop interface."""

    def __init__(self, traci_module):
        self._traci = traci_module
        self._running = False

    @classmethod
    def create(cls, prefer_libsumo: bool = True) -> "SumoAdapter":
        traci_module = _import_traci(prefer_libsumo=prefer_libsumo)
        return cls(traci_module)

    def start(self, command: list[str]) -> None:
        self._traci.start(command)
        self._running = True

    def run_step_loop(
        self,
        *,
        max_steps: int,
        stop_when_no_vehicles: bool,
        on_step: Callable[[int, float, object], None] | None = None,
    ) -> int:
        if not self._running:
            raise RuntimeError("SUMO adapter is not running. Call start() first.")

        steps = 0
        while steps < max_steps:
            if stop_when_no_vehicles and steps > 0:
                if self._traci.simulation.getMinExpectedNumber() <= 0:
                    break

            self._traci.simulationStep()
            sim_time = float(self._traci.simulation.getTime())

            if on_step is not None:
                on_step(steps, sim_time, self._traci)

            steps += 1

        return steps

    def set_view_boundary(
        self,
        *,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        view_id: str = "View #0",
    ) -> None:
        if not self._running:
            return

        gui = getattr(self._traci, "gui", None)
        if gui is None:
            return

        try:
            gui.setBoundary(view_id, xmin, ymin, xmax, ymax)
        except Exception:
            # Some backends or builds may not expose GUI controls.
            return

    def close(self, wait: bool = True) -> None:
        if self._running:
            try:
                self._traci.close(wait)
            except TypeError:
                # libsumo.close() does not accept a wait argument.
                self._traci.close()
            self._running = False
