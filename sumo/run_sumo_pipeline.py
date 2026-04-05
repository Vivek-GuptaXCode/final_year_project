from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from urllib import error, request
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

try:
    from sumo.sumo_adapter import (
        SumoAdapter,
        build_sumo_command,
        load_scenario_config,
    )
except ModuleNotFoundError:
    from sumo_adapter import (  # type: ignore
        SumoAdapter,
        build_sumo_command,
        load_scenario_config,
    )

try:
    from pipelines.logging.runtime_logger import SumoSimulationDataLogger
except ModuleNotFoundError:
    # Support direct script execution: `python3 sumo/run_sumo_pipeline.py`
    # where sys.path starts at `sumo/` and does not include project root.
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.append(str(_PROJECT_ROOT))
    try:
        from pipelines.logging.runtime_logger import SumoSimulationDataLogger
    except ModuleNotFoundError:
        SumoSimulationDataLogger = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUMO scaffold loop for data pipeline integration.")
    parser.add_argument(
        "--contract",
        default="sumo/scenarios/sumo_contract.json",
        help="Path to SUMO scenario contract JSON.",
    )
    parser.add_argument(
        "--scenario",
        choices=["low", "medium", "high", "demo", "city", "kolkata"],
        default="demo",
        help="Scenario name from contract (default: demo -> real-city 3D hackathon flow).",
    )
    parser.add_argument("--seed", type=int, default=11, help="SUMO random seed.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override contract max steps.")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui binary instead of sumo.")
    parser.add_argument(
        "--three-d",
        action="store_true",
        help="Enable OpenSceneGraph renderer (requires SUMO build with --osg-view support).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command/config only; do not import traci/libsumo.",
    )
    parser.add_argument(
        "--rsu-range-m",
        type=float,
        default=120.0,
        help="RSU range radius in meters used for GUI range overlays.",
    )
    parser.add_argument(
        "--rsu-min-inc-lanes",
        type=int,
        default=4,
        help="Place RSU only on junctions with at least this many incoming lanes.",
    )
    parser.add_argument(
        "--rsu-max-count",
        type=int,
        default=40,
        help="Maximum number of RSU circles to draw.",
    )
    parser.add_argument(
        "--rsu-min-spacing-m",
        type=float,
        default=None,
        help="Minimum center-to-center spacing between RSUs (default: 1.8 * rsu-range-m).",
    )
    parser.add_argument(
        "--rsu-whitelist",
        type=str,
        default=None,
        help="Comma-separated list of RSU aliases to keep (e.g., 'A,B,K,M,R'). Only these RSUs will be active.",
    )
    parser.add_argument(
        "--traffic-scale",
        type=float,
        default=1.0,
        help="Global demand multiplier via SUMO --scale (use >1.0 for jam-level traffic).",
    )
    parser.add_argument(
        "--traffic-reduction-pct",
        type=float,
        default=0.0,
        help="Optional traffic reduction percentage applied to traffic-scale (default: 0, opt-in).",
    )
    parser.add_argument(
        "--controlled-count",
        type=int,
        default=0,
        help="Number of AI-controlled test vehicles generated as a dedicated flow.",
    )
    parser.add_argument(
        "--controlled-source",
        default=None,
        help="Source location for controlled vehicles (junction id or edge id).",
    )
    parser.add_argument(
        "--controlled-destination",
        default=None,
        help="Destination location for controlled vehicles (junction id or edge id).",
    )
    parser.add_argument(
        "--controlled-via-rsus",
        default="",
        help="Comma-separated intermediate RSU locations (junction ids or edge ids).",
    )
    parser.add_argument(
        "--controlled-begin",
        type=float,
        default=90.0,
        help="Begin time for controlled vehicle flow.",
    )
    parser.add_argument(
        "--controlled-end",
        type=float,
        default=900.0,
        help="End time for controlled vehicle flow.",
    )
    parser.add_argument(
        "--emergency-count",
        type=int,
        default=0,
        help="Base emergency vehicle count; effective generated count is tripled (x3).",
    )
    parser.add_argument(
        "--emergency-source",
        default=None,
        help="Source location for emergency vehicles (junction id or edge id).",
    )
    parser.add_argument(
        "--emergency-destination",
        default=None,
        help="Destination location for emergency vehicles (junction id or edge id).",
    )
    parser.add_argument(
        "--emergency-via-rsus",
        default="",
        help="Comma-separated intermediate RSU locations for emergency vehicles.",
    )
    parser.add_argument(
        "--emergency-begin",
        type=float,
        default=120.0,
        help="Begin time for emergency vehicle flow.",
    )
    parser.add_argument(
        "--emergency-end",
        type=float,
        default=1800.0,
        help="End time for emergency vehicle flow.",
    )
    parser.add_argument(
        "--suggest-near-junction",
        default=None,
        help="Print nearby valid drivable junction IDs around the given junction and exit.",
    )
    parser.add_argument(
        "--suggest-purpose",
        choices=["source", "destination", "checkpoint", "any"],
        default="any",
        help="Filter suggested junctions for source/destination/checkpoint suitability.",
    )
    parser.add_argument(
        "--suggest-count",
        type=int,
        default=8,
        help="Number of nearest suggested junctions to print.",
    )
    parser.add_argument(
        "--list-rsus",
        action="store_true",
        help="Print RSU aliases (A, B, ... AA) mapped to junction IDs and exit.",
    )
    parser.add_argument(
        "--auto-fallback-junctions",
        action="store_true",
        help=(
            "Auto-replace invalid controlled junction source/destination/checkpoints with nearest valid "
            "passenger-drivable junctions (junction-mode only)."
        ),
    )
    parser.add_argument(
        "--enable-hybrid-uplink-stub",
        action="store_true",
        help="Send periodic RSU batch payload stubs to server /route during SUMO step loop.",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:5000",
        help="Base server URL for hybrid uplink stub (default: http://localhost:5000).",
    )
    parser.add_argument(
        "--hybrid-batch-seconds",
        type=float,
        default=5.0,
        help="Batch period for hybrid uplink stub payloads in simulation seconds.",
    )
    parser.add_argument(
        "--route-timeout-seconds",
        type=float,
        default=1.5,
        help="HTTP timeout for server /route call in hybrid uplink stub.",
    )
    parser.add_argument(
        "--reroute-highlight-seconds",
        type=float,
        default=8.0,
        help="Duration to keep GUI highlight on vehicles rerouted from server policy.",
    )
    parser.add_argument(
        "--enable-emergency-priority",
        action="store_true",
        help="Enable emergency-vehicle priority: optimal reroute + corridor preemption.",
    )
    parser.add_argument(
        "--emergency-corridor-lookahead-edges",
        type=int,
        default=6,
        help="Number of upcoming edges treated as emergency corridor.",
    )
    parser.add_argument(
        "--emergency-hold-seconds",
        type=float,
        default=8.0,
        help="Duration to hold non-emergency traffic stopped on emergency corridor edges.",
    )
    parser.add_argument(
        "--enable-runtime-logging",
        action="store_true",
        help="Enable Phase-1 1 Hz logging to data/raw/<run_id>/ (RSU + edge + manifest).",
    )
    parser.add_argument(
        "--runtime-log-root",
        default="data/raw",
        help="Output root for runtime logs (default: data/raw).",
    )
    parser.add_argument(
        "--runtime-log-run-id",
        default=None,
        help="Optional explicit run id for runtime logs (default: auto timestamp_scenario_seed).",
    )
    parser.add_argument(
        "--statistics-output",
        default=None,
        help="Optional SUMO statistics XML output path (--statistic-output).",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional SUMO summary XML output path (--summary-output).",
    )
    parser.add_argument(
        "--tripinfo-output",
        default=None,
        help="Optional SUMO tripinfo XML output path (--tripinfo-output).",
    )
    parser.add_argument(
        "--tripinfo-write-unfinished",
        action="store_true",
        help="Include vehicles that have not arrived by simulation end in tripinfo output.",
    )
    parser.add_argument(
        "--kpi-output-dir",
        default=None,
        help=(
            "Optional output directory for auto-named KPI XML files "
            "(statistics/summary/tripinfo)."
        ),
    )
    parser.add_argument(
        "--kpi-output-prefix",
        default=None,
        help="Filename prefix used with --kpi-output-dir (default: auto timestamp_scenario_seed).",
    )
    # ── Phase 4: RL adaptive signal control ──────────────────────────────
    parser.add_argument(
        "--enable-rl-signal-control",
        action="store_true",
        help=(
            "Enable Phase-4 RL adaptive traffic signal control. "
            "Uses pre-trained DQN weights if --rl-model-dir is set, "
            "otherwise falls back to SimpleActuated policy."
        ),
    )
    parser.add_argument(
        "--rl-model-dir",
        default=None,
        help="Path to DQN weights directory (models/rl/artifacts by default).",
    )
    parser.add_argument(
        "--rl-tls-ids",
        default=None,
        help="Comma-separated TLS junction IDs to control (auto-discovers all if omitted).",
    )
    parser.add_argument(
        "--rl-min-green-seconds",
        type=float,
        default=15.0,
        help="Minimum green duration enforced by safety guardrail (default: 15 s).",
    )
    parser.add_argument(
        "--rl-yellow-duration-seconds",
        type=float,
        default=3.0,
        help="Yellow transition window inserted between green phases (default: 3 s).",
    )
    parser.add_argument(
        "--rl-max-switches-per-window",
        type=int,
        default=4,
        help="Max phase switches allowed per 60-s rolling window (anti-oscillation).",
    )
    return parser.parse_args()


def _post_json(url: str, payload: dict, timeout_seconds: float) -> dict | None:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def _update_edge_weights_from_congestion(traci_module, *, conservative: bool = True) -> int:
    """Update SUMO edge travel times based on real-time congestion.
    
    This implements dynamic edge weight updates (key technique from PressLight/MA2C).
    Vehicles rerouting will use these updated weights for path finding.
    
    Args:
        traci_module: The SUMO TraCI module
        conservative: If True, use gentler penalties to avoid route oscillation
        
    Returns number of edges updated.
    """
    updated = 0
    try:
        edge_ids = traci_module.edge.getIDList()
        for edge_id in edge_ids:
            if edge_id.startswith(":"):  # Skip internal edges
                continue
            try:
                # Get current travel time (based on actual vehicle speeds)
                current_tt = traci_module.edge.getTraveltime(edge_id)
                # Get number of halting vehicles (queue length proxy)
                halting = traci_module.edge.getLastStepHaltingNumber(edge_id)
                # Get mean speed
                mean_speed = traci_module.edge.getLastStepMeanSpeed(edge_id)
                
                # Conservative mode: Only penalize severely congested edges
                # This avoids route oscillation where all vehicles switch together
                if conservative:
                    # Much higher thresholds to avoid false positives
                    if halting > 8 or mean_speed < 1.0:
                        # Gentler penalty to avoid over-steering
                        congestion_factor = 1.0 + (halting * 0.05) + (max(0, 3.0 - mean_speed) * 0.1)
                        adjusted_tt = current_tt * min(congestion_factor, 1.8)  # Cap at 1.8x
                        traci_module.edge.adaptTraveltime(edge_id, adjusted_tt)
                        updated += 1
                else:
                    # Original aggressive mode
                    if halting > 3 or mean_speed < 2.0:
                        congestion_factor = 1.0 + (halting * 0.15) + (max(0, 5.0 - mean_speed) * 0.2)
                        adjusted_tt = current_tt * min(congestion_factor, 3.0)
                        traci_module.edge.adaptTraveltime(edge_id, adjusted_tt)
                        updated += 1
            except Exception:
                continue
    except Exception:
        pass
    return updated


def _filter_vehicles_for_rerouting(
    traci_module,
    vehicle_ids: list[str],
    *,
    min_remaining_distance: float = 200.0,
    min_remaining_edges: int = 3,
) -> list[str]:
    """Filter vehicles that would benefit from rerouting.
    
    Avoids rerouting vehicles that are:
    - Too close to destination (would add overhead without benefit)
    - Already on optimal path (short remaining route)
    - Currently waiting/stopped (may cause issues)
    
    This is a key optimization from traffic engineering literature.
    """
    eligible = []
    for vid in vehicle_ids:
        try:
            # Skip vehicles near destination
            route = traci_module.vehicle.getRoute(vid)
            route_idx = traci_module.vehicle.getRouteIndex(vid)
            remaining_edges = len(route) - route_idx - 1
            
            if remaining_edges < min_remaining_edges:
                continue
            
            # Estimate remaining distance
            remaining_dist = 0.0
            current_edge = traci_module.vehicle.getRoadID(vid)
            if current_edge and not current_edge.startswith(":"):
                pos_on_edge = traci_module.vehicle.getLanePosition(vid)
                edge_length = traci_module.lane.getLength(traci_module.vehicle.getLaneID(vid))
                remaining_dist = edge_length - pos_on_edge
            
            # Add remaining edges
            for edge in route[route_idx + 1:]:
                if not edge.startswith(":"):
                    try:
                        remaining_dist += traci_module.lane.getLength(edge + "_0")
                    except Exception:
                        remaining_dist += 100.0  # Estimate
            
            if remaining_dist < min_remaining_distance:
                continue
            
            # Vehicle is eligible for rerouting
            eligible.append(vid)
        except Exception:
            continue
    
    return eligible


def _prioritize_vehicles_by_delay(
    traci_module,
    vehicle_ids: list[str],
    target_count: int,
) -> list[str]:
    """Prioritize vehicles with highest accumulated delay for rerouting.
    
    Vehicles stuck in traffic benefit most from rerouting.
    This improves overall efficiency vs random selection.
    """
    if len(vehicle_ids) <= target_count:
        return vehicle_ids
    
    vehicle_delays = []
    for vid in vehicle_ids:
        try:
            waiting_time = traci_module.vehicle.getAccumulatedWaitingTime(vid)
            vehicle_delays.append((vid, waiting_time))
        except Exception:
            vehicle_delays.append((vid, 0.0))
    
    # Sort by waiting time (highest first) and return top target_count
    vehicle_delays.sort(key=lambda x: -x[1])
    return [vid for vid, _ in vehicle_delays[:target_count]]


def _is_reroute_safe_now(traci_module, vehicle_id: str) -> bool:
    """Avoid applying route changes while a vehicle is on internal junction edges."""
    try:
        road_id = str(traci_module.vehicle.getRoadID(vehicle_id))
    except Exception:
        return False
    if not road_id:
        return False
    return not road_id.startswith(":")


def _reroute_with_dijkstra_fallback(traci_module, vehicle_id: str) -> bool:
    """Fallback route recomputation using findRoute (Dijkstra by default in SUMO)."""
    try:
        current_edge = str(traci_module.vehicle.getRoadID(vehicle_id))
        if not current_edge or current_edge.startswith(":"):
            return False

        current_route = list(traci_module.vehicle.getRoute(vehicle_id))
        if not current_route:
            return False
        destination_edge = str(current_route[-1])
        if not destination_edge:
            return False

        stage = traci_module.simulation.findRoute(current_edge, destination_edge)
        new_edges = list(getattr(stage, "edges", []))
        if not new_edges:
            return False
        if new_edges[0] != current_edge:
            return False

        traci_module.vehicle.setRoute(vehicle_id, new_edges)
        return True
    except Exception:
        return False


def _build_rsu_knn_edges(
    rsu_alias_table: list[tuple[str, str, float, float]],
    k: int = 3,
) -> list[tuple[str, str]]:
    """Connect each RSU to its K nearest neighbours (undirected, no duplicates)."""
    nodes = [(jid, x, y) for _alias, jid, x, y in rsu_alias_table]
    if len(nodes) < 2:
        return []
    edges: set[tuple[str, str]] = set()
    for i, (jid_a, xa, ya) in enumerate(nodes):
        distances = sorted(
            (math.hypot(xa - xb, ya - yb), jid_b)
            for j, (jid_b, xb, yb) in enumerate(nodes)
            if i != j
        )
        for _dist, jid_b in distances[:k]:
            edge = (min(jid_a, jid_b), max(jid_a, jid_b))
            edges.add(edge)
    return list(edges)


def _try_register_rsu_graph(
    register_url: str,
    rsu_alias_table: list[tuple[str, str, float, float]],
    k_neighbors: int = 3,
    timeout: float = 2.0,
) -> bool:
    """POST RSU graph topology to server /graph/register. Returns True on success."""
    if not rsu_alias_table:
        return False
    nodes = [jid for _alias, jid, _x, _y in rsu_alias_table]
    edges = _build_rsu_knn_edges(rsu_alias_table, k=k_neighbors)
    payload = {"nodes": nodes, "edges": [[u, v] for u, v in edges]}
    result = _post_json(register_url, payload, timeout_seconds=timeout)
    if result is not None and result.get("status") == "ok":
        print(
            "[SUMO][GNN] RSU graph registered: nodes={n} edges={e}".format(
                n=result.get("node_count", len(nodes)),
                e=result.get("edge_count", len(edges)),
            )
        )
        return True
    print("[SUMO][GNN] RSU graph registration failed — server may not be up yet or missing /graph/register")
    return False


def _is_emergency_vehicle(traci_module, vehicle_id: str) -> bool:
    try:
        vclass = str(traci_module.vehicle.getVehicleClass(vehicle_id)).lower()
        if vclass == "emergency":
            return True
    except Exception:
        pass

    try:
        type_id = str(traci_module.vehicle.getTypeID(vehicle_id)).lower()
    except Exception:
        type_id = ""

    emergency_tokens = ("emergency", "ambulance", "fire", "police")
    return any(token in type_id for token in emergency_tokens)


def _apply_emergency_priority_policy(
    traci_module,
    *,
    sim_time: float,
    vehicle_ids: list[str],
    held_until: dict[str, float],
    lookahead_edges: int,
    hold_seconds: float,
) -> dict[str, int]:
    emergency_ids = [vid for vid in vehicle_ids if _is_emergency_vehicle(traci_module, vid)]
    corridor_edges: set[str] = set()
    emergency_reroutes = 0

    for evid in emergency_ids:
        if not _is_reroute_safe_now(traci_module, evid):
            continue

        try:
            current_edge = str(traci_module.vehicle.getRoadID(evid))
            current_route = list(traci_module.vehicle.getRoute(evid))
            if not current_route or not current_edge or current_edge.startswith(":"):
                continue
            destination_edge = str(current_route[-1])

            stage = traci_module.simulation.findRoute(current_edge, destination_edge)
            optimal_edges = list(getattr(stage, "edges", []))
            if optimal_edges and optimal_edges[0] == current_edge:
                traci_module.vehicle.setRoute(evid, optimal_edges)
                active_route = optimal_edges
                emergency_reroutes += 1
            else:
                active_route = current_route

            try:
                idx = active_route.index(current_edge)
            except Exception:
                idx = max(0, int(traci_module.vehicle.getRouteIndex(evid)))

            for edge_id in active_route[idx : idx + max(1, lookahead_edges)]:
                if edge_id and not str(edge_id).startswith(":"):
                    corridor_edges.add(str(edge_id))
        except Exception:
            continue

    preempted = 0
    if corridor_edges:
        for vid in vehicle_ids:
            if vid in emergency_ids:
                continue
            try:
                road_id = str(traci_module.vehicle.getRoadID(vid))
            except Exception:
                continue
            if road_id in corridor_edges:
                try:
                    traci_module.vehicle.setSpeed(vid, 0.0)
                    held_until[vid] = sim_time + max(0.1, hold_seconds)
                    preempted += 1
                except Exception:
                    continue

    released = 0
    for vid, until in list(held_until.items()):
        if vid not in vehicle_ids or sim_time >= until:
            try:
                if vid in vehicle_ids:
                    traci_module.vehicle.setSpeed(vid, -1)
                    released += 1
            except Exception:
                pass
            held_until.pop(vid, None)

    return {
        "emergency_count": len(emergency_ids),
        "emergency_reroutes": emergency_reroutes,
        "corridor_preempted": preempted,
        "released": released,
    }


def _apply_server_reroute_policy(
    traci_module,
    vehicle_ids: list[str],
    route_response: dict,
    *,
    sim_time: float | None = None,
    reroute_cooldown_until: dict[str, float] | None = None,
    reroute_cooldown_seconds: float = 25.0,
) -> dict[str, Any]:
    """Apply live rerouting decisions from server policy fields.

    This is the runtime bridge that turns cloud policy output into TraCI route updates.
    """
    rec = route_response.get("recommended_action") or {}
    vehicle_id_set = set(vehicle_ids)
    emergency_action = route_response.get("emergency_action") or {}
    emergency_active = bool(emergency_action.get("active", False))
    emergency_vehicle_ids = {
        str(vid)
        for vid in (emergency_action.get("vehicle_ids") or [])
        if str(vid) in vehicle_id_set
    }

    if not bool(rec.get("reroute_enabled", False)) and not emergency_active:
        return {"count": 0, "vehicle_ids": []}

    try:
        confidence = float(route_response.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    try:
        min_confidence = float(rec.get("min_confidence", 0.5))
    except Exception:
        min_confidence = 0.5

    if not emergency_active:
        try:
            conf_floor = float(os.getenv("HYBRID_REROUTE_MIN_CONF_FLOOR", "0.58"))
        except Exception:
            conf_floor = 0.58
        conf_floor = max(0.0, min(1.0, conf_floor))
        min_confidence = max(min_confidence, conf_floor)

    if confidence < min_confidence and not emergency_active:
        return {"count": 0, "vehicle_ids": []}

    if not vehicle_ids:
        return {"count": 0, "vehicle_ids": []}

    try:
        reroute_fraction = float(rec.get("reroute_fraction", 0.0))
    except Exception:
        reroute_fraction = 0.0
    reroute_fraction = max(0.0, min(1.0, reroute_fraction))
    if not emergency_active:
        try:
            fraction_cap = float(os.getenv("HYBRID_REROUTE_FRACTION_CAP", "0.12"))
        except Exception:
            fraction_cap = 0.12
        fraction_cap = max(0.0, min(1.0, fraction_cap))
        reroute_fraction = min(reroute_fraction, fraction_cap)

    reroute_mode = str(rec.get("reroute_mode", "travel_time"))
    fallback_algorithm = str(rec.get("fallback_algorithm", "")).lower()
    routing_engine = route_response.get("routing_engine") or {}
    if not fallback_algorithm:
        fallback_algorithm = str(routing_engine.get("fallback", "")).lower()

    directives_raw = route_response.get("route_directives")
    planned_reroutes: list[tuple[str, str]] = []
    if isinstance(directives_raw, list):
        seen_ids: set[str] = set()
        for row in directives_raw:
            if not isinstance(row, dict):
                continue
            vid = str(row.get("vehicle_id", "")).strip()
            if not vid or vid in seen_ids or vid not in vehicle_id_set:
                continue
            mode = str(row.get("mode", reroute_mode))
            planned_reroutes.append((vid, mode))
            seen_ids.add(vid)

    if emergency_active and emergency_vehicle_ids:
        # Emergency flow should clear the path for emergency vehicles, not reroute all traffic.
        planned_reroutes = [
            (vid, mode)
            for vid, mode in planned_reroutes
            if vid in emergency_vehicle_ids
        ]
        if not planned_reroutes:
            planned_reroutes = [(vid, "dijkstra") for vid in vehicle_ids if vid in emergency_vehicle_ids]
    else:
        if reroute_fraction <= 0.0:
            return {"count": 0, "vehicle_ids": []}
        
        # IMPROVEMENT: Filter vehicles that would benefit from rerouting
        # Skip vehicles near destination or with short remaining routes
        eligible_vehicles = _filter_vehicles_for_rerouting(
            traci_module, 
            vehicle_ids,
            min_remaining_distance=150.0,
            min_remaining_edges=2,
        )
        
        if not eligible_vehicles:
            return {"count": 0, "vehicle_ids": []}
        
        # Calculate target count from eligible vehicles
        target_count = max(1, int(len(eligible_vehicles) * reroute_fraction))
        
        # IMPROVEMENT: Prioritize vehicles with highest delay
        # Vehicles stuck in traffic benefit most from rerouting
        if planned_reroutes:
            # Use server directives but filter to eligible
            planned_reroutes = [
                (vid, mode) for vid, mode in planned_reroutes 
                if vid in eligible_vehicles
            ][:target_count]
        else:
            # Prioritize by accumulated delay
            priority_vehicles = _prioritize_vehicles_by_delay(
                traci_module, eligible_vehicles, target_count
            )
            planned_reroutes = [(vid, reroute_mode) for vid in priority_vehicles]

    applied = 0
    rerouted_ids: list[str] = []
    for vid, mode in planned_reroutes:
        if (
            reroute_cooldown_until is not None
            and sim_time is not None
            and float(reroute_cooldown_until.get(vid, -1.0)) > sim_time
        ):
            continue

        if not _is_reroute_safe_now(traci_module, vid):
            continue

        try:
            if mode in {"gnn_effort", "effort"}:
                traci_module.vehicle.rerouteEffort(vid)
            elif mode == "dijkstra":
                if not _reroute_with_dijkstra_fallback(traci_module, vid):
                    continue
            else:
                traci_module.vehicle.rerouteTraveltime(vid)
            applied += 1
            rerouted_ids.append(vid)
            if reroute_cooldown_until is not None and sim_time is not None:
                reroute_cooldown_until[vid] = sim_time + max(1.0, reroute_cooldown_seconds)
        except Exception:
            if fallback_algorithm == "dijkstra":
                if _reroute_with_dijkstra_fallback(traci_module, vid):
                    applied += 1
                    rerouted_ids.append(vid)
                    if reroute_cooldown_until is not None and sim_time is not None:
                        reroute_cooldown_until[vid] = sim_time + max(1.0, reroute_cooldown_seconds)
            continue

    return {"count": applied, "vehicle_ids": rerouted_ids}


def _resolve_net_file_from_sumocfg(sumocfg_path: Path) -> Path | None:
    try:
        root = ET.parse(sumocfg_path).getroot()
    except Exception:
        return None

    net_node = root.find("./input/net-file")
    if net_node is None:
        return None

    value = net_node.attrib.get("value")
    if not value:
        return None

    net_path = Path(value)
    if not net_path.is_absolute():
        net_path = sumocfg_path.parent / net_path
    return net_path.resolve()


def _parse_world_bounds_from_net(net_file: Path) -> tuple[float, float, float, float] | None:
    try:
        root = ET.parse(net_file).getroot()
    except Exception:
        return None

    location = root.find("location")
    if location is None:
        return None

    conv_boundary = location.attrib.get("convBoundary")
    if not conv_boundary:
        return None

    try:
        min_x, min_y, max_x, max_y = [float(v) for v in conv_boundary.split(",")]
    except Exception:
        return None

    # SUMO GUI camera controls expect network (converted) coordinates.
    return (min_x, min_y, max_x, max_y)


def _resolve_additional_files_from_sumocfg(sumocfg_path: Path) -> list[Path]:
    try:
        root = ET.parse(sumocfg_path).getroot()
    except Exception:
        return []

    node = root.find("./input/additional-files")
    if node is None:
        return []

    raw_value = node.attrib.get("value", "")
    if not raw_value.strip():
        return []

    resolved: list[Path] = []
    for piece in raw_value.split(","):
        part = piece.strip()
        if not part:
            continue
        p = Path(part)
        if not p.is_absolute():
            p = (sumocfg_path.parent / p).resolve()
        resolved.append(p)
    return resolved


def _resolve_route_files_from_sumocfg(sumocfg_path: Path) -> list[Path]:
    try:
        root = ET.parse(sumocfg_path).getroot()
    except Exception:
        return []

    node = root.find("./input/route-files")
    if node is None:
        return []

    raw_value = node.attrib.get("value", "")
    if not raw_value.strip():
        return []

    resolved: list[Path] = []
    for piece in raw_value.split(","):
        part = piece.strip()
        if not part:
            continue
        p = Path(part)
        if not p.is_absolute():
            p = (sumocfg_path.parent / p).resolve()
        resolved.append(p)
    return resolved


def _resolve_net_ids(net_file: Path) -> tuple[set[str], set[str]]:
    root = ET.parse(net_file).getroot()

    junction_ids: set[str] = set()
    for junction in root.findall("junction"):
        jid = junction.attrib.get("id")
        if not jid:
            continue
        jtype = junction.attrib.get("type", "")
        if jtype == "internal":
            continue
        junction_ids.add(jid)

    edge_ids: set[str] = set()
    for edge in root.findall("edge"):
        eid = edge.attrib.get("id")
        if not eid or eid.startswith(":"):
            continue
        if edge.attrib.get("function", "") == "internal":
            continue
        edge_ids.add(eid)

    return junction_ids, edge_ids


def _lane_allows_passenger(lane_node: ET.Element) -> bool:
    allow = lane_node.attrib.get("allow", "").strip()
    disallow = lane_node.attrib.get("disallow", "").strip()

    if allow:
        allowed = set(allow.split())
        return "passenger" in allowed or "all" in allowed

    if disallow:
        disallowed = set(disallow.split())
        return "passenger" not in disallowed and "all" not in disallowed

    # SUMO default lane permissions allow passenger unless restricted.
    return True


def _resolve_passenger_junction_connectivity(net_file: Path) -> tuple[dict[str, int], dict[str, int]]:
    root = ET.parse(net_file).getroot()

    incoming_counts: dict[str, int] = {}
    outgoing_counts: dict[str, int] = {}

    for edge in root.findall("edge"):
        edge_id = edge.attrib.get("id", "")
        if not edge_id or edge_id.startswith(":"):
            continue
        if edge.attrib.get("function", "") == "internal":
            continue

        # Only consider edges that have at least one lane usable by passenger vehicles.
        if not any(_lane_allows_passenger(lane) for lane in edge.findall("lane")):
            continue

        from_junction = edge.attrib.get("from")
        to_junction = edge.attrib.get("to")
        if from_junction:
            outgoing_counts[from_junction] = outgoing_counts.get(from_junction, 0) + 1
        if to_junction:
            incoming_counts[to_junction] = incoming_counts.get(to_junction, 0) + 1

    return incoming_counts, outgoing_counts


def _resolve_junction_positions(net_file: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(net_file).getroot()
    positions: dict[str, tuple[float, float]] = {}

    for junction in root.findall("junction"):
        jid = junction.attrib.get("id")
        if not jid:
            continue
        try:
            x = float(junction.attrib.get("x", ""))
            y = float(junction.attrib.get("y", ""))
        except Exception:
            continue
        positions[jid] = (x, y)

    return positions


def _suggest_nearest_junctions(
    *,
    target_junction: str,
    purpose: str,
    count: int,
    positions: dict[str, tuple[float, float]],
    incoming_counts: dict[str, int],
    outgoing_counts: dict[str, int],
) -> list[tuple[float, str, int, int]]:
    if target_junction not in positions:
        return []

    tx, ty = positions[target_junction]
    candidates: list[tuple[float, str, int, int]] = []

    for jid, (x, y) in positions.items():
        if jid == target_junction:
            continue

        incoming = incoming_counts.get(jid, 0)
        outgoing = outgoing_counts.get(jid, 0)

        if purpose == "source" and outgoing <= 0:
            continue
        if purpose == "destination" and incoming <= 0:
            continue
        if purpose == "checkpoint" and (incoming <= 0 or outgoing <= 0):
            continue
        if purpose == "any" and (incoming <= 0 and outgoing <= 0):
            continue

        dist = math.hypot(x - tx, y - ty)
        candidates.append((dist, jid, incoming, outgoing))

    candidates.sort(key=lambda item: item[0])
    return candidates[: max(1, count)]


def _auto_fix_controlled_junctions(
    *,
    net_file: Path,
    source: str,
    destination: str,
    via_list: list[str],
) -> tuple[str, str, list[str], list[tuple[str, str, str]]]:
    """Auto-fix junction IDs to nearest drivable alternatives.

    Returns (source, destination, via_list, replacements) where replacements contain
    tuples of (role, old_id, new_id).
    """
    junction_ids, _edge_ids = _resolve_net_ids(net_file)
    all_as_junctions = source in junction_ids and destination in junction_ids and all(
        via in junction_ids for via in via_list
    )
    if not all_as_junctions:
        # Fallback applies only to junction-mode input.
        return source, destination, via_list, []

    incoming_counts, outgoing_counts = _resolve_passenger_junction_connectivity(net_file)
    positions = _resolve_junction_positions(net_file)

    replacements: list[tuple[str, str, str]] = []

    fixed_source = source
    if outgoing_counts.get(fixed_source, 0) <= 0:
        candidates = _suggest_nearest_junctions(
            target_junction=fixed_source,
            purpose="source",
            count=1,
            positions=positions,
            incoming_counts=incoming_counts,
            outgoing_counts=outgoing_counts,
        )
        if not candidates:
            raise ValueError(
                f"Controlled source junction '{source}' is invalid and no nearby valid source fallback was found."
            )
        fixed_source = candidates[0][1]
        replacements.append(("source", source, fixed_source))

    fixed_destination = destination
    if incoming_counts.get(fixed_destination, 0) <= 0:
        candidates = _suggest_nearest_junctions(
            target_junction=fixed_destination,
            purpose="destination",
            count=1,
            positions=positions,
            incoming_counts=incoming_counts,
            outgoing_counts=outgoing_counts,
        )
        if not candidates:
            raise ValueError(
                f"Controlled destination junction '{destination}' is invalid and no nearby valid destination fallback was found."
            )
        fixed_destination = candidates[0][1]
        replacements.append(("destination", destination, fixed_destination))

    fixed_via: list[str] = []
    used_ids = {fixed_source, fixed_destination}
    for via in via_list:
        current_via = via
        if incoming_counts.get(current_via, 0) <= 0 or outgoing_counts.get(current_via, 0) <= 0:
            candidates = _suggest_nearest_junctions(
                target_junction=current_via,
                purpose="checkpoint",
                count=12,
                positions=positions,
                incoming_counts=incoming_counts,
                outgoing_counts=outgoing_counts,
            )
            replacement = None
            for _dist, jid, _incoming, _outgoing in candidates:
                if jid not in used_ids:
                    replacement = jid
                    break
            if replacement is None:
                raise ValueError(
                    f"Controlled checkpoint junction '{via}' is invalid and no nearby valid checkpoint fallback was found."
                )
            current_via = replacement
            replacements.append(("checkpoint", via, current_via))

        if current_via not in used_ids:
            fixed_via.append(current_via)
            used_ids.add(current_via)

    return fixed_source, fixed_destination, fixed_via, replacements


def _parse_csv_values(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _build_runtime_run_id(*, scenario: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{scenario}_seed{seed}"


def _resolve_project_path(path_value: str | Path, *, project_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    return path


def _to_bijective_base26_label(index_1_based: int) -> str:
    if index_1_based <= 0:
        raise ValueError("index must be >= 1")

    n = index_1_based
    out: list[str] = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out.append(chr(ord("A") + rem))
    out.reverse()
    return "".join(out)


def _build_rsu_alias_table(
    *,
    net_file: Path,
    min_incoming_lanes: int,
    max_count: int,
    min_spacing_m: float,
) -> list[tuple[str, str, float, float]]:
    try:
        root = ET.parse(net_file).getroot()
    except Exception:
        return []

    selected, _candidate_count = _select_rsu_junctions(
        root,
        min_incoming_lanes=min_incoming_lanes,
        max_count=max_count,
        min_spacing_m=min_spacing_m,
    )

    table: list[tuple[str, str, float, float]] = []
    for idx, (jid, x, y) in enumerate(selected, start=1):
        alias = _to_bijective_base26_label(idx)
        table.append((alias, jid, x, y))
    return table


def _resolve_rsu_identifier(token: str, alias_to_junction: dict[str, str]) -> str:
    normalized = token.strip()
    if not normalized:
        return normalized

    upper = normalized.upper()
    if upper in alias_to_junction:
        return alias_to_junction[upper]

    if upper.startswith("RSU_") and upper[4:] in alias_to_junction:
        return alias_to_junction[upper[4:]]
    if upper.startswith("RSU-") and upper[4:] in alias_to_junction:
        return alias_to_junction[upper[4:]]
    if upper.startswith("RSU") and upper[3:] in alias_to_junction:
        return alias_to_junction[upper[3:]]

    return normalized


def _resolve_rsu_route_inputs(
    *,
    source: str,
    destination: str,
    via_list: list[str],
    alias_to_junction: dict[str, str],
) -> tuple[str, str, list[str], list[tuple[str, str, str]]]:
    replacements: list[tuple[str, str, str]] = []

    resolved_source = _resolve_rsu_identifier(source, alias_to_junction)
    if resolved_source != source:
        replacements.append(("source", source, resolved_source))

    resolved_destination = _resolve_rsu_identifier(destination, alias_to_junction)
    if resolved_destination != destination:
        replacements.append(("destination", destination, resolved_destination))

    resolved_via: list[str] = []
    for via in via_list:
        resolved = _resolve_rsu_identifier(via, alias_to_junction)
        if resolved != via:
            replacements.append(("checkpoint", via, resolved))
        resolved_via.append(resolved)

    return resolved_source, resolved_destination, resolved_via, replacements


def _resolve_route_mode_and_attrs(
    *,
    net_file: Path,
    source: str,
    destination: str,
    via_list: list[str],
) -> tuple[str, str, str, str]:
    junction_ids, edge_ids = _resolve_net_ids(net_file)

    all_as_junctions = source in junction_ids and destination in junction_ids and all(
        via in junction_ids for via in via_list
    )
    all_as_edges = source in edge_ids and destination in edge_ids and all(via in edge_ids for via in via_list)

    if all_as_junctions:
        incoming_counts, outgoing_counts = _resolve_passenger_junction_connectivity(net_file)

        src_outgoing = outgoing_counts.get(source, 0)
        dst_incoming = incoming_counts.get(destination, 0)
        if src_outgoing <= 0:
            raise ValueError(
                f"Source junction '{source}' has no passenger-drivable outgoing edges. "
                "Pick a source junction connected to a drivable road."
            )
        if dst_incoming <= 0:
            raise ValueError(
                f"Destination junction '{destination}' has no passenger-drivable incoming edges. "
                "Pick a destination junction reachable via drivable roads."
            )

        for via in via_list:
            via_incoming = incoming_counts.get(via, 0)
            via_outgoing = outgoing_counts.get(via, 0)
            if via_incoming <= 0 or via_outgoing <= 0:
                raise ValueError(
                    f"Checkpoint junction '{via}' is not usable as an intermediate passenger waypoint "
                    "(needs both incoming and outgoing passenger-drivable edges)."
                )

        route_mode = "junction"
        src_attr = 'fromJunction="{}"'.format(escape(source))
        dst_attr = 'toJunction="{}"'.format(escape(destination))
        via_attr = ""
        if via_list:
            via_attr = ' viaJunctions="{}"'.format(escape(" ".join(via_list)))
        return route_mode, src_attr, dst_attr, via_attr

    if all_as_edges:
        route_mode = "edge"
        src_attr = 'from="{}"'.format(escape(source))
        dst_attr = 'to="{}"'.format(escape(destination))
        via_attr = ""
        if via_list:
            via_attr = ' via="{}"'.format(escape(" ".join(via_list)))
        return route_mode, src_attr, dst_attr, via_attr

    raise ValueError(
        "Route IDs must consistently be valid junction IDs or valid edge IDs. "
        "Use either (source, destination, via) all as junctions or all as edges."
    )


def _generate_guided_flow_route_file(
    *,
    net_file: Path,
    scenario_name: str,
    route_file_suffix: str,
    flow_id: str,
    vehicle_type_id: str,
    vehicle_class: str,
    vehicle_color: str,
    max_speed: float,
    vehicle_count: int,
    source: str,
    destination: str,
    via_list: list[str],
    begin_time: float,
    end_time: float,
) -> tuple[Path, str]:
    if vehicle_count <= 0:
        raise ValueError("vehicle count must be positive")
    if begin_time < 0:
        raise ValueError("begin time must be >= 0")
    if end_time <= begin_time:
        raise ValueError("end time must be greater than begin time")

    route_mode, src_attr, dst_attr, via_attr = _resolve_route_mode_and_attrs(
        net_file=net_file,
        source=source,
        destination=destination,
        via_list=via_list,
    )

    route_file = net_file.parent.parent / "scenarios" / f"{scenario_name}_{route_file_suffix}.rou.xml"
    route_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "<routes>",
        (
            f'    <vType id="{escape(vehicle_type_id)}" vClass="{escape(vehicle_class)}" '
            f'color="{escape(vehicle_color)}" maxSpeed="{max_speed:.2f}" accel="2.6" decel="4.5" sigma="0.2"/>'
        ),
        (
            "    "
            f'<flow id="{escape(flow_id)}" type="{escape(vehicle_type_id)}" begin="{begin_time:.2f}" end="{end_time:.2f}" '
            f'number="{vehicle_count}" departLane="best" departSpeed="max" departPos="base" '
            f'{src_attr} {dst_attr}{via_attr}/>'
        ),
        "</routes>",
    ]
    route_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return route_file, route_mode


def _generate_controlled_group_route_file(
    *,
    net_file: Path,
    scenario_name: str,
    vehicle_count: int,
    source: str,
    destination: str,
    via_list: list[str],
    begin_time: float,
    end_time: float,
) -> tuple[Path, str]:
    return _generate_guided_flow_route_file(
        net_file=net_file,
        scenario_name=scenario_name,
        route_file_suffix="controlled_group",
        flow_id="controlled_group_flow",
        vehicle_type_id="controlled_ai_vehicle",
        vehicle_class="passenger",
        vehicle_color="0,51,153",
        max_speed=16.67,
        vehicle_count=vehicle_count,
        source=source,
        destination=destination,
        via_list=via_list,
        begin_time=begin_time,
        end_time=end_time,
    )


def _generate_emergency_group_route_file(
    *,
    net_file: Path,
    scenario_name: str,
    vehicle_count: int,
    source: str,
    destination: str,
    via_list: list[str],
    begin_time: float,
    end_time: float,
) -> tuple[Path, str]:
    return _generate_guided_flow_route_file(
        net_file=net_file,
        scenario_name=scenario_name,
        route_file_suffix="emergency_group",
        flow_id="emergency_group_flow",
        vehicle_type_id="emergency_priority_vehicle",
        vehicle_class="emergency",
        vehicle_color="255,255,0",
        max_speed=22.22,
        vehicle_count=vehicle_count,
        source=source,
        destination=destination,
        via_list=via_list,
        begin_time=begin_time,
        end_time=end_time,
    )


def _highlight_vehicle_circle(
    traci_module,
    vehicle_id: str,
    *,
    set_vehicle_color: bool,
    vehicle_color: tuple[int, int, int, int],
    highlight_color: tuple[int, int, int, int],
    radius_m: float,
    alpha_max: int = -1,
    duration_s: float = -1.0,
    highlight_type: int = 0,
) -> None:
    # Keep the vehicle body color stable and high-contrast.
    if set_vehicle_color:
        try:
            traci_module.vehicle.setColor(vehicle_id, vehicle_color)
        except Exception:
            pass

    try:
        # Persistent highlighting avoids fade-reset flicker when called each step.
        if alpha_max > 0 and duration_s > 0:
            traci_module.vehicle.highlight(
                vehicle_id,
                highlight_color,
                radius_m,
                alpha_max,
                duration_s,
                highlight_type,
            )
        else:
            traci_module.vehicle.highlight(vehicle_id, highlight_color, radius_m)
    except Exception:
        # If highlighting is unavailable on this build, keep body-color fallback only.
        pass


def _apply_visual_vehicle_markers(traci_module, vehicle_ids: list[str]) -> dict[str, int]:
    marked_controlled = 0
    marked_emergency = 0

    for vid in vehicle_ids:
        try:
            type_id = str(traci_module.vehicle.getTypeID(vid))
        except Exception:
            continue

        if type_id == "controlled_ai_vehicle":
            _highlight_vehicle_circle(
                traci_module,
                vid,
                set_vehicle_color=True,
                vehicle_color=(0, 51, 153, 255),
                highlight_color=(64, 224, 255, 170),
                radius_m=5.2,
            )
            marked_controlled += 1
            continue

        if type_id == "emergency_priority_vehicle" or _is_emergency_vehicle(traci_module, vid):
            _highlight_vehicle_circle(
                traci_module,
                vid,
                set_vehicle_color=True,
                vehicle_color=(255, 255, 0, 255),
                highlight_color=(255, 69, 0, 190),
                radius_m=6.6,
            )
            marked_emergency += 1

    return {
        "controlled_marked": marked_controlled,
        "emergency_marked": marked_emergency,
    }


def _apply_active_reroute_highlights(
    traci_module,
    *,
    sim_time: float,
    active_vehicle_ids: set[str],
    reroute_highlight_until: dict[str, float],
) -> int:
    highlighted = 0
    for vid, until in list(reroute_highlight_until.items()):
        if sim_time >= until or vid not in active_vehicle_ids:
            reroute_highlight_until.pop(vid, None)
            continue

        _highlight_vehicle_circle(
            traci_module,
            vid,
            set_vehicle_color=False,
            vehicle_color=(0, 0, 0, 0),
            highlight_color=(255, 20, 147, 200),
            radius_m=4.7,
            highlight_type=2,
        )
        highlighted += 1
    return highlighted


def _build_circle_shape_points(*, x: float, y: float, radius_m: float, points: int = 24) -> str:
    coords: list[str] = []
    for i in range(points):
        theta = 2.0 * math.pi * (i / points)
        cx = x + radius_m * math.cos(theta)
        cy = y + radius_m * math.sin(theta)
        coords.append(f"{cx:.2f},{cy:.2f}")
    return " ".join(coords)


def _distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def _parse_shape_points(shape: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for raw in shape.split():
        if "," not in raw:
            continue
        sx, sy = raw.split(",", 1)
        try:
            points.append((float(sx), float(sy)))
        except Exception:
            continue
    return points


def _normalize_vector(dx: float, dy: float) -> tuple[float, float] | None:
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    return (dx / length, dy / length)


def _collect_connected_lane_samples_and_normals(
    root,
    *,
    junction_id: str,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    samples: list[tuple[float, float]] = []
    normals: list[tuple[float, float]] = []

    for edge in root.findall("edge"):
        edge_id = edge.attrib.get("id", "")
        if not edge_id or edge_id.startswith(":"):
            continue
        if edge.attrib.get("function", "") == "internal":
            continue

        from_junction = edge.attrib.get("from")
        to_junction = edge.attrib.get("to")
        if from_junction != junction_id and to_junction != junction_id:
            continue

        for lane in edge.findall("lane"):
            points = _parse_shape_points(lane.attrib.get("shape", ""))
            if len(points) < 2:
                continue

            if from_junction == junction_id:
                near = points[0]
                away = points[1]
                samples.extend(points[: min(4, len(points))])
            else:
                near = points[-1]
                away = points[-2]
                samples.extend(points[max(0, len(points) - 4) :])

            direction = _normalize_vector(away[0] - near[0], away[1] - near[1])
            if direction is not None:
                nx, ny = -direction[1], direction[0]
                normals.append((nx, ny))
                normals.append((-nx, -ny))

            # One lane is enough to infer side-of-road orientation for this edge.
            break

    return samples, normals


def _select_rsu_label_position(
    root,
    *,
    junction_id: str,
    x: float,
    y: float,
    alias_index: int,
) -> tuple[float, float, tuple[float, float]]:
    lane_samples, normal_candidates = _collect_connected_lane_samples_and_normals(root, junction_id=junction_id)

    if not normal_candidates:
        # Deterministic fallback orientation based on alias index.
        angle = math.radians((alias_index * 37) % 360)
        normal_candidates = [
            (math.cos(angle), math.sin(angle)),
            (-math.cos(angle), -math.sin(angle)),
        ]

    offsets = (18.0, 24.0, 30.0)
    best_score = -1.0
    best_x = x
    best_y = y - offsets[0]
    best_dir = normal_candidates[0]

    for nx, ny in normal_candidates:
        for offset in offsets:
            cx = x + nx * offset
            cy = y + ny * offset

            if lane_samples:
                clearance = min(_distance_xy((cx, cy), sample) for sample in lane_samples)
            else:
                clearance = offset

            # Prefer points away from lane centerlines and slightly away from junction center.
            score = clearance + 0.05 * offset
            if score > best_score:
                best_score = score
                best_x = cx
                best_y = cy
                best_dir = (nx, ny)

    return best_x, best_y, best_dir


def _select_rsu_junctions(
    root,
    *,
    min_incoming_lanes: int,
    max_count: int,
    min_spacing_m: float,
) -> tuple[list[tuple[str, float, float]], int]:
    candidates: list[tuple[str, float, float, int, bool]] = []

    for junction in root.findall("junction"):
        jid = junction.attrib.get("id")
        jtype = junction.attrib.get("type", "")
        x = junction.attrib.get("x")
        y = junction.attrib.get("y")
        if not jid or not x or not y:
            continue

        # Ignore helper and terminal nodes.
        if jtype in {"internal", "dead_end"}:
            continue

        inc_lanes = junction.attrib.get("incLanes", "").split()
        inc_count = sum(1 for lane_id in inc_lanes if lane_id and not lane_id.startswith(":"))
        if inc_count < min_incoming_lanes:
            continue

        try:
            xv = float(x)
            yv = float(y)
        except Exception:
            continue

        is_signalized = jtype in {"traffic_light", "traffic_light_unregulated", "traffic_light_right_on_red"}
        candidates.append((jid, xv, yv, inc_count, is_signalized))

    # Prioritize signalized and higher-lane junctions.
    candidates.sort(key=lambda item: (item[4], item[3]), reverse=True)

    selected: list[tuple[str, float, float]] = []
    for jid, xv, yv, _inc_count, _is_signalized in candidates:
        pos = (xv, yv)
        if any(_distance_xy(pos, (sx, sy)) < min_spacing_m for _sid, sx, sy in selected):
            continue
        selected.append((jid, xv, yv))
        if len(selected) >= max_count:
            break

    return selected, len(candidates)


def _generate_rsu_poi_add_file(
    net_file: Path,
    scenario_name: str,
    rsu_range_m: float,
    min_incoming_lanes: int,
    max_count: int,
    min_spacing_m: float,
    rsu_whitelist: set[str] | None = None,
) -> tuple[Path | None, int, int]:
    try:
        root = ET.parse(net_file).getroot()
    except Exception:
        return None, 0, 0

    rsu_nodes, candidate_count = _select_rsu_junctions(
        root,
        min_incoming_lanes=min_incoming_lanes,
        max_count=max_count,
        min_spacing_m=min_spacing_m,
    )

    if not rsu_nodes:
        return None, 0, candidate_count

    output_path = net_file.parent.parent / "scenarios" / f"{scenario_name}_rsu_pois.add.xml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "<additional>",
    ]
    placed_labels: list[tuple[float, float]] = []
    
    # Build full node list with aliases first (to maintain consistent alias assignment)
    nodes_with_alias: list[tuple[str, str, float, float]] = []
    for idx, (jid, x, y) in enumerate(rsu_nodes, start=1):
        alias = _to_bijective_base26_label(idx)
        nodes_with_alias.append((alias, jid, x, y))
    
    # Filter by whitelist if provided
    if rsu_whitelist:
        nodes_with_alias = [(alias, jid, x, y) for alias, jid, x, y in nodes_with_alias if alias in rsu_whitelist]
    
    for alias, jid, x, y in nodes_with_alias:
        label_text = f"RSU_{alias}"

        range_shape = _build_circle_shape_points(x=x, y=y, radius_m=rsu_range_m)
        # Find original index for label positioning
        original_idx = next((i+1 for i, (j, _, _) in enumerate(rsu_nodes) if j == jid), 1)
        label_x, label_y, label_dir = _select_rsu_label_position(
            root,
            junction_id=jid,
            x=x,
            y=y,
            alias_index=original_idx,
        )

        # Keep text labels spread out and away from dense center areas.
        for _ in range(4):
            if not any(_distance_xy((label_x, label_y), pos) < 16.0 for pos in placed_labels):
                break
            label_x += label_dir[0] * 7.0
            label_y += label_dir[1] * 7.0
        placed_labels.append((label_x, label_y))

        # Transparent RSU range with red circumference only.
        lines.append(
            f'    <poly id="rsu_range_{escape(jid)}" type="rsu_range" color="255,0,0,255" layer="12" lineWidth="2" fill="false" shape="{range_shape}"/>'
        )
        lines.append(
            f'    <poi id="rsu_label_anchor_{escape(alias)}" type="rsu_anchor" color="26,140,26,220" layer="13" x="{label_x:.2f}" y="{label_y:.2f}" width="2.4"/>'
        )
        lines.append(
            f'    <poi id="rsu_label_text_{escape(alias)}" type="{escape(label_text)}" color="0,0,0,0" layer="14" x="{label_x:.2f}" y="{label_y:.2f}">'
        )
        lines.append(f'        <param key="name" value="{escape(label_text)}"/>')
        lines.append("    </poi>")

    lines.append("</additional>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path, len(nodes_with_alias), candidate_count


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    project_root = Path(__file__).resolve().parent.parent

    config = load_scenario_config(contract_path, args.scenario)
    additional_files = _resolve_additional_files_from_sumocfg(config.sumocfg_path)
    route_files = _resolve_route_files_from_sumocfg(config.sumocfg_path)
    net_file = _resolve_net_file_from_sumocfg(config.sumocfg_path)
    use_junction_taz = False

    rsu_alias_table: list[tuple[str, str, float, float]] = []
    rsu_alias_map: dict[str, str] = {}
    if net_file is not None:
        rsu_spacing_for_alias = args.rsu_min_spacing_m
        if rsu_spacing_for_alias is None:
            rsu_spacing_for_alias = max(80.0, args.rsu_range_m * 1.8)

        rsu_alias_table = _build_rsu_alias_table(
            net_file=net_file,
            min_incoming_lanes=max(1, args.rsu_min_inc_lanes),
            max_count=max(1, args.rsu_max_count),
            min_spacing_m=max(1.0, rsu_spacing_for_alias),
        )
        
        # Apply RSU whitelist filter if specified
        if args.rsu_whitelist:
            whitelist_set = set()
            for token in args.rsu_whitelist.split(","):
                alias = token.strip().upper()
                # Handle RSU_X or just X format
                if alias.startswith("RSU_"):
                    alias = alias[4:]
                elif alias.startswith("RSU"):
                    alias = alias[3:]
                whitelist_set.add(alias)
            
            original_count = len(rsu_alias_table)
            rsu_alias_table = [
                (alias, jid, x, y) for alias, jid, x, y in rsu_alias_table
                if alias in whitelist_set
            ]
            if rsu_alias_table:
                print(f"[SUMO][RSU] Whitelist applied: {len(rsu_alias_table)}/{original_count} RSUs retained")
                print(f"[SUMO][RSU] Active RSUs: {', '.join('RSU_' + alias for alias, _, _, _ in rsu_alias_table)}")
            else:
                print(f"[SUMO][RSU] Warning: Whitelist filtered out all RSUs! Check aliases.")
        
        rsu_alias_map = {alias: jid for alias, jid, _x, _y in rsu_alias_table}

    if args.list_rsus:
        if net_file is None:
            raise ValueError("--list-rsus requires a valid net-file in the scenario config")

        print(f"[SUMO] RSU aliases for scenario '{config.scenario}':")
        if not rsu_alias_table:
            print("[SUMO] No RSUs selected with current filters.")
        else:
            for alias, jid, x, y in rsu_alias_table:
                print(f"  - RSU_{alias}: junction={jid} x={x:.2f} y={y:.2f}")
        return

    if args.suggest_near_junction is not None:
        if net_file is None:
            raise ValueError("--suggest-near-junction requires a valid net-file in the scenario config")

        incoming_counts, outgoing_counts = _resolve_passenger_junction_connectivity(net_file)
        positions = _resolve_junction_positions(net_file)
        target = args.suggest_near_junction
        if target not in positions:
            raise ValueError(f"Junction '{target}' not found in network {net_file}")

        suggestions = _suggest_nearest_junctions(
            target_junction=target,
            purpose=args.suggest_purpose,
            count=args.suggest_count,
            positions=positions,
            incoming_counts=incoming_counts,
            outgoing_counts=outgoing_counts,
        )

        print(f"[SUMO] Suggestions near junction {target} (purpose={args.suggest_purpose}):")
        if not suggestions:
            print("[SUMO] No suitable nearby junctions found.")
        else:
            for dist, jid, incoming, outgoing in suggestions:
                print(
                    f"  - {jid}  distance={dist:.2f}m  incoming(passenger)={incoming}  outgoing(passenger)={outgoing}"
                )
        return

    traffic_scale = args.traffic_scale
    if traffic_scale <= 0:
        raise ValueError("--traffic-scale must be > 0")

    reduction_pct = args.traffic_reduction_pct
    if reduction_pct < 0 or reduction_pct >= 100:
        raise ValueError("--traffic-reduction-pct must be in [0, 100)")
    effective_scale = traffic_scale * (1.0 - (reduction_pct / 100.0))
    if effective_scale <= 0:
        raise ValueError("effective traffic scale must stay > 0")
    traffic_scale = effective_scale
    print(f"[SUMO] Traffic scale after {reduction_pct:.1f}% reduction: {traffic_scale:.4f}")

    if args.controlled_count < 0:
        raise ValueError("--controlled-count must be >= 0")
    if args.controlled_count > 0:
        if net_file is None:
            raise ValueError("controlled flow generation requires a valid net-file in the scenario config")
        if not args.controlled_source or not args.controlled_destination:
            raise ValueError(
                "--controlled-source and --controlled-destination are required when --controlled-count > 0"
            )

        controlled_via = _parse_csv_values(args.controlled_via_rsus)
        controlled_source = args.controlled_source
        controlled_destination = args.controlled_destination

        (
            controlled_source,
            controlled_destination,
            controlled_via,
            controlled_alias_replacements,
        ) = _resolve_rsu_route_inputs(
            source=controlled_source,
            destination=controlled_destination,
            via_list=controlled_via,
            alias_to_junction=rsu_alias_map,
        )
        for role, old_id, new_id in controlled_alias_replacements:
            print(f"[SUMO] Controlled RSU alias ({role}): {old_id} -> {new_id}")

        if args.auto_fallback_junctions:
            (
                controlled_source,
                controlled_destination,
                controlled_via,
                replacements,
            ) = _auto_fix_controlled_junctions(
                net_file=net_file,
                source=controlled_source,
                destination=controlled_destination,
                via_list=controlled_via,
            )
            for role, old_id, new_id in replacements:
                print(f"[SUMO] Auto-fallback ({role}): {old_id} -> {new_id}")

        controlled_file, route_mode = _generate_controlled_group_route_file(
            net_file=net_file,
            scenario_name=config.scenario,
            vehicle_count=args.controlled_count,
            source=controlled_source,
            destination=controlled_destination,
            via_list=controlled_via,
            begin_time=args.controlled_begin,
            end_time=args.controlled_end,
        )
        route_files.append(controlled_file)
        if route_mode == "junction":
            use_junction_taz = True

        print(
            f"[SUMO] Controlled cohort: {args.controlled_count} vehicles, mode={route_mode}, "
            f"source={controlled_source}, destination={controlled_destination}, "
            f"via={controlled_via if controlled_via else '[]'}"
        )

    if args.emergency_count < 0:
        raise ValueError("--emergency-count must be >= 0")
    if args.emergency_count > 0:
        emergency_count_multiplier = 3
        effective_emergency_count = args.emergency_count * emergency_count_multiplier

        if net_file is None:
            raise ValueError("emergency flow generation requires a valid net-file in the scenario config")
        if not args.emergency_source or not args.emergency_destination:
            raise ValueError(
                "--emergency-source and --emergency-destination are required when --emergency-count > 0"
            )

        emergency_via = _parse_csv_values(args.emergency_via_rsus)
        emergency_source = args.emergency_source
        emergency_destination = args.emergency_destination

        (
            emergency_source,
            emergency_destination,
            emergency_via,
            emergency_alias_replacements,
        ) = _resolve_rsu_route_inputs(
            source=emergency_source,
            destination=emergency_destination,
            via_list=emergency_via,
            alias_to_junction=rsu_alias_map,
        )
        for role, old_id, new_id in emergency_alias_replacements:
            print(f"[SUMO] Emergency RSU alias ({role}): {old_id} -> {new_id}")

        if args.auto_fallback_junctions:
            (
                emergency_source,
                emergency_destination,
                emergency_via,
                replacements,
            ) = _auto_fix_controlled_junctions(
                net_file=net_file,
                source=emergency_source,
                destination=emergency_destination,
                via_list=emergency_via,
            )
            for role, old_id, new_id in replacements:
                print(f"[SUMO] Emergency auto-fallback ({role}): {old_id} -> {new_id}")

        emergency_file, emergency_mode = _generate_emergency_group_route_file(
            net_file=net_file,
            scenario_name=config.scenario,
            vehicle_count=effective_emergency_count,
            source=emergency_source,
            destination=emergency_destination,
            via_list=emergency_via,
            begin_time=args.emergency_begin,
            end_time=args.emergency_end,
        )
        route_files.append(emergency_file)
        if emergency_mode == "junction":
            use_junction_taz = True

        print(
            f"[SUMO] Emergency cohort: {effective_emergency_count} vehicles (base={args.emergency_count}, x3), mode={emergency_mode}, "
            f"source={emergency_source}, destination={emergency_destination}, "
            f"via={emergency_via if emergency_via else '[]'}"
        )

    if args.gui and net_file is not None:
        rsu_spacing = args.rsu_min_spacing_m
        if rsu_spacing is None:
            rsu_spacing = max(80.0, args.rsu_range_m * 1.8)

        # Parse whitelist for POI generation
        poi_whitelist: set[str] | None = None
        if args.rsu_whitelist:
            poi_whitelist = set()
            for token in args.rsu_whitelist.split(","):
                alias = token.strip().upper()
                if alias.startswith("RSU_"):
                    alias = alias[4:]
                elif alias.startswith("RSU"):
                    alias = alias[3:]
                poi_whitelist.add(alias)

        poi_file, selected_count, candidate_count = _generate_rsu_poi_add_file(
            net_file,
            config.scenario,
            rsu_range_m=max(5.0, args.rsu_range_m),
            min_incoming_lanes=max(1, args.rsu_min_inc_lanes),
            max_count=max(1, args.rsu_max_count),
            min_spacing_m=max(1.0, rsu_spacing),
            rsu_whitelist=poi_whitelist,
        )
        if poi_file is not None:
            additional_files.append(poi_file)
            whitelist_note = f" (whitelist: {len(poi_whitelist)} RSUs)" if poi_whitelist else ""
            print(
                f"[SUMO] RSU overlays: selected {selected_count} intersections "
                f"out of {candidate_count} candidates{whitelist_note} (min-inc-lanes={args.rsu_min_inc_lanes}, "
                f"min-spacing={rsu_spacing:.1f}m, max-count={args.rsu_max_count})."
            )
            if config.gui_use_osg_view or args.three_d:
                print(
                    "[SUMO] Note: OSG 3D mode may hide POI/poly overlays on some builds. "
                    "Use 2D GUI (omit --three-d) for guaranteed RSU-range visibility."
                )

    statistics_output_path = (
        _resolve_project_path(args.statistics_output, project_root=project_root)
        if args.statistics_output
        else None
    )
    summary_output_path = (
        _resolve_project_path(args.summary_output, project_root=project_root)
        if args.summary_output
        else None
    )
    tripinfo_output_path = (
        _resolve_project_path(args.tripinfo_output, project_root=project_root)
        if args.tripinfo_output
        else None
    )

    if args.kpi_output_dir:
        kpi_output_dir = _resolve_project_path(args.kpi_output_dir, project_root=project_root)
        kpi_output_prefix = args.kpi_output_prefix or _build_runtime_run_id(
            scenario=config.scenario,
            seed=args.seed,
        )

        if statistics_output_path is None:
            statistics_output_path = kpi_output_dir / f"{kpi_output_prefix}_statistics.xml"
        if summary_output_path is None:
            summary_output_path = kpi_output_dir / f"{kpi_output_prefix}_summary.xml"
        if tripinfo_output_path is None:
            tripinfo_output_path = kpi_output_dir / f"{kpi_output_prefix}_tripinfo.xml"

    kpi_output_paths = {
        "statistics_output": statistics_output_path,
        "summary_output": summary_output_path,
        "tripinfo_output": tripinfo_output_path,
    }
    if any(path is not None for path in kpi_output_paths.values()):
        print(
            "[SUMO] KPI outputs enabled: "
            f"statistics={statistics_output_path if statistics_output_path is not None else 'disabled'}, "
            f"summary={summary_output_path if summary_output_path is not None else 'disabled'}, "
            f"tripinfo={tripinfo_output_path if tripinfo_output_path is not None else 'disabled'}"
        )

    command = build_sumo_command(
        config,
        seed=args.seed,
        use_gui=args.gui,
        force_3d=args.three_d,
        additional_files=additional_files,
        route_files=route_files,
        scale=traffic_scale,
        junction_taz=use_junction_taz,
        statistics_output_path=statistics_output_path,
        summary_output_path=summary_output_path,
        tripinfo_output_path=tripinfo_output_path,
        tripinfo_write_unfinished=bool(args.tripinfo_write_unfinished),
    )
    max_steps = args.max_steps if args.max_steps is not None else config.default_max_steps

    print("[SUMO] Scenario:", config.scenario)
    print("[SUMO] Config:", config.sumocfg_path)
    print("[SUMO] Command:", command)
    print("[SUMO] Max steps:", max_steps)

    if args.dry_run:
        print("[SUMO] Dry-run complete. No TraCI/libsumo session started.")
        return

    for output_path in kpi_output_paths.values():
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

    # libsumo is an in-process backend and does not provide the GUI window.
    # Force traci backend whenever GUI mode is requested.
    prefer_libsumo = config.prefer_libsumo and not args.gui
    adapter = SumoAdapter.create(prefer_libsumo=prefer_libsumo)
    adapter.start(command)

    if args.gui:
        if net_file is not None:
            bounds = _parse_world_bounds_from_net(net_file)
            if bounds is not None:
                adapter.set_view_boundary(
                    xmin=bounds[0],
                    ymin=bounds[1],
                    xmax=bounds[2],
                    ymax=bounds[3],
                )

    runtime_logger = None
    runtime_logger_failed = False
    if args.enable_runtime_logging:
        if SumoSimulationDataLogger is None:
            print("[SUMO][LOGGER] runtime logger unavailable: could not import pipelines.logging.runtime_logger")
        elif net_file is None:
            print("[SUMO][LOGGER] runtime logging skipped: scenario has no resolvable net-file")
        else:
            runtime_log_root = _resolve_project_path(args.runtime_log_root, project_root=project_root)

            runtime_run_id = args.runtime_log_run_id or _build_runtime_run_id(
                scenario=config.scenario,
                seed=args.seed,
            )
            runtime_run_dir = runtime_log_root / runtime_run_id

            rsu_alias_table_for_logging = list(rsu_alias_table)
            if not rsu_alias_table_for_logging:
                rsu_spacing_for_logging = args.rsu_min_spacing_m
                if rsu_spacing_for_logging is None:
                    rsu_spacing_for_logging = max(80.0, args.rsu_range_m * 1.8)
                rsu_alias_table_for_logging = _build_rsu_alias_table(
                    net_file=net_file,
                    min_incoming_lanes=1,
                    max_count=max(1, args.rsu_max_count),
                    min_spacing_m=max(1.0, rsu_spacing_for_logging),
                )

            run_metadata = {
                "run_id": runtime_run_id,
                "scenario": config.scenario,
                "seed": args.seed,
                "contract": str(contract_path),
                "sumocfg": str(config.sumocfg_path),
                "max_steps": int(max_steps),
                "step_length_seconds": float(config.step_length_seconds),
                "traffic_scale": float(traffic_scale),
                "sumo_command": command,
            }
            kpi_outputs_metadata = {
                key: str(path)
                for key, path in kpi_output_paths.items()
                if path is not None
            }
            if kpi_outputs_metadata:
                run_metadata["kpi_outputs"] = kpi_outputs_metadata

            try:
                runtime_logger = SumoSimulationDataLogger(
                    run_dir=runtime_run_dir,
                    run_metadata=run_metadata,
                    net_file=net_file,
                    rsu_alias_table=rsu_alias_table_for_logging,
                    rsu_range_m=max(5.0, args.rsu_range_m),
                )
                print(
                    "[SUMO][LOGGER] Enabled 1 Hz logging: run_id={run_id} rsu_count={rsu_count} edge_count={edge_count} dir={dir}".format(
                        run_id=runtime_run_id,
                        rsu_count=len(rsu_alias_table_for_logging),
                        edge_count=len(getattr(runtime_logger, "_edge_ids", [])),
                        dir=runtime_run_dir,
                    )
                )
            except Exception as exc:
                runtime_logger = None
                print(f"[SUMO][LOGGER] Failed to initialize runtime logger: {exc}")

    # ── Phase 4: RL signal controller (lazy import so it's optional) ─────
    rl_signal_controller = None
    if getattr(args, "enable_rl_signal_control", False):
        try:
            from controllers.rl.inference_hook import RLSignalController
            rl_signal_controller = RLSignalController.from_args(args, None)  # traci not yet live
            print("[SUMO][RL] RL signal controller initialised (will activate on first step)")
        except Exception as _rl_exc:
            print(f"[SUMO][RL] Could not load RL controller: {_rl_exc}")

    try:
        last_hybrid_push_sim_time = -1e9
        try:
            reroute_cooldown_seconds = float(os.getenv("HYBRID_REROUTE_COOLDOWN_SECONDS", "20.0"))
        except Exception:
            reroute_cooldown_seconds = 20.0
        reroute_cooldown_seconds = max(1.0, reroute_cooldown_seconds)
        held_until: dict[str, float] = {}
        reroute_highlight_until: dict[str, float] = {}
        reroute_cooldown_until: dict[str, float] = {}
        enable_vehicle_markers = args.controlled_count > 0 or args.emergency_count > 0
        rsu_graph_registered: list[bool] = [False]  # mutable flag for closure

        def _on_step(step_idx: int, sim_time: float, traci_module) -> None:
            nonlocal last_hybrid_push_sim_time, runtime_logger_failed
            try:
                vehicle_ids = list(traci_module.vehicle.getIDList())
            except Exception:
                return
            active_vehicle_ids = set(vehicle_ids)

            if runtime_logger is not None and not runtime_logger_failed:
                try:
                    runtime_logger.maybe_log(
                        sim_time_seconds=sim_time,
                        frame_idx=step_idx,
                        traci_module=traci_module,
                        vehicle_ids=vehicle_ids,
                    )
                except Exception as exc:
                    runtime_logger_failed = True
                    print(f"[SUMO][LOGGER] Disabled after runtime error: {exc}")

            if enable_vehicle_markers:
                marker_stats = _apply_visual_vehicle_markers(traci_module, vehicle_ids)
                if step_idx % 50 == 0 and (
                    marker_stats["controlled_marked"] > 0 or marker_stats["emergency_marked"] > 0
                ):
                    print(
                        "[SUMO][MARKERS] deep_blue_controlled={c} bright_yellow_emergency={e}".format(
                            c=marker_stats["controlled_marked"],
                            e=marker_stats["emergency_marked"],
                        )
                    )

            reroute_marker_count = _apply_active_reroute_highlights(
                traci_module,
                sim_time=sim_time,
                active_vehicle_ids=active_vehicle_ids,
                reroute_highlight_until=reroute_highlight_until,
            )
            for vid, until in list(reroute_cooldown_until.items()):
                if vid not in active_vehicle_ids or sim_time >= until:
                    reroute_cooldown_until.pop(vid, None)
            if step_idx % 50 == 0 and reroute_marker_count > 0:
                print(f"[SUMO][MARKERS] reroute_highlights={reroute_marker_count}")

            if args.enable_emergency_priority:
                emergency_stats = _apply_emergency_priority_policy(
                    traci_module,
                    sim_time=sim_time,
                    vehicle_ids=vehicle_ids,
                    held_until=held_until,
                    lookahead_edges=max(1, args.emergency_corridor_lookahead_edges),
                    hold_seconds=max(0.1, args.emergency_hold_seconds),
                )
                if emergency_stats["emergency_count"] > 0:
                    print(
                        "[SUMO][EMERGENCY] active={a} rerouted={r} preempted={p} released={rel}".format(
                            a=emergency_stats["emergency_count"],
                            r=emergency_stats["emergency_reroutes"],
                            p=emergency_stats["corridor_preempted"],
                            rel=emergency_stats["released"],
                        )
                    )

            # ── Phase 4: RL adaptive signal control ──────────────────────────
            if rl_signal_controller is not None:
                try:
                    rl_signal_controller.step(sim_time, traci_module)
                except Exception as _rl_step_exc:
                    if step_idx % 200 == 0:
                        print(f"[SUMO][RL] step error (step={step_idx}): {_rl_step_exc}")

            if not args.enable_hybrid_uplink_stub:
                return

            if sim_time - last_hybrid_push_sim_time < max(0.1, args.hybrid_batch_seconds):
                return

            # ── Edge weight updates DISABLED ───────────────────────────────────
            # Testing showed that dynamic edge weight updates cause route oscillation
            # where vehicles all switch to alternate routes simultaneously.
            # TODO: Re-enable with per-vehicle randomization to prevent herding.
            # if step_idx % 10 == 0:
            #     edges_updated = _update_edge_weights_from_congestion(traci_module, conservative=True)
            #     if edges_updated > 10 and step_idx % 100 == 0:
            #         print(f"[SUMO][HYBRID] Updated {edges_updated} edge weights")

            # ── One-time RSU graph registration so GNN has real topology ──────
            if not rsu_graph_registered[0] and rsu_alias_table:
                register_url = args.server_url.rstrip("/") + "/graph/register"
                rsu_graph_registered[0] = _try_register_rsu_graph(
                    register_url,
                    rsu_alias_table,
                    k_neighbors=3,
                    timeout=max(0.5, args.route_timeout_seconds),
                )

            # ── Per-RSU vehicle segmentation ──────────────────────────────────
            # Collect vehicle positions to assign each vehicle to its nearest RSU.
            vehicle_positions: dict[str, tuple[float, float]] = {}
            for vid in vehicle_ids:
                try:
                    vehicle_positions[vid] = traci_module.vehicle.getPosition(vid)
                except Exception:
                    pass

            rsu_range_m = max(5.0, args.rsu_range_m)
            # Map junction_id → vehicles within coverage range
            rsu_vehicle_map: dict[str, list[str]] = {}
            if rsu_alias_table:
                for vid, (vx, vy) in vehicle_positions.items():
                    best_jid: str | None = None
                    best_dist = float("inf")
                    for _alias, jid, rx, ry in rsu_alias_table:
                        d = math.hypot(vx - rx, vy - ry)
                        if d < best_dist:
                            best_dist = d
                            best_jid = jid
                    if best_jid is not None and best_dist <= rsu_range_m:
                        rsu_vehicle_map.setdefault(best_jid, []).append(vid)

            # Select dominant RSU (most vehicles within coverage) for this batch
            if rsu_vehicle_map:
                dominant_jid = max(rsu_vehicle_map, key=lambda j: len(rsu_vehicle_map[j]))
                local_vehicles = rsu_vehicle_map[dominant_jid]
            elif rsu_alias_table:
                # No vehicles within coverage radius; use first RSU, all vehicles
                dominant_jid = rsu_alias_table[0][1]
                local_vehicles = vehicle_ids
            else:
                dominant_jid = "global_stub"
                local_vehicles = vehicle_ids

            # Compute avg speed from vehicles near the dominant RSU
            speeds: list[float] = []
            for vid in local_vehicles:
                try:
                    speeds.append(float(traci_module.vehicle.getSpeed(vid)))
                except Exception:
                    continue
            avg_speed_mps = (sum(speeds) / len(speeds)) if speeds else 0.0

            emergency_vehicle_ids = [
                vid for vid in vehicle_ids if _is_emergency_vehicle(traci_module, vid)
            ]
            uplink_payload = {
                "rsu_id": dominant_jid,
                "timestamp": sim_time,
                "vehicle_count": len(local_vehicles),
                "avg_speed_mps": avg_speed_mps,
                "vehicle_ids": local_vehicles,
                "emergency_vehicle_ids": emergency_vehicle_ids,
                "features": {
                    "scenario": config.scenario,
                    "seed": args.seed,
                    "traffic_scale": traffic_scale,
                    "step": step_idx,
                    "dominant_rsu_vehicle_count": len(local_vehicles),
                    "total_vehicle_count": len(vehicle_ids),
                },
            }

            route_url = args.server_url.rstrip("/") + "/route"
            route_response = _post_json(route_url, uplink_payload, timeout_seconds=max(0.1, args.route_timeout_seconds))
            if route_response is not None:
                reroute_result = _apply_server_reroute_policy(
                    traci_module,
                    local_vehicles,
                    route_response,
                    sim_time=sim_time,
                    reroute_cooldown_until=reroute_cooldown_until,
                    reroute_cooldown_seconds=reroute_cooldown_seconds,
                )
                reroutes_applied = int(reroute_result.get("count", 0))
                if args.reroute_highlight_seconds > 0:
                    hold_until = sim_time + max(0.1, args.reroute_highlight_seconds)
                    for vid in reroute_result.get("vehicle_ids", []):
                        reroute_highlight_until[str(vid)] = hold_until
                print(
                    "[SUMO][HYBRID] /route rsu={rsu} p={p:.2f} u={u:.2f} c={c:.2f} "
                    "risk={risk} strategy={strat} reroutes={r}".format(
                        rsu=route_response.get("rsu_id", "?"),
                        p=float(route_response.get("p_congestion", 0.0)),
                        u=float(route_response.get("uncertainty", 1.0)),
                        c=float(route_response.get("confidence", 0.0)),
                        risk=route_response.get("risk_level", "unknown"),
                        strat=(
                            route_response.get("gnn_routing", {}).get("strategy")
                            or route_response.get("phase3", {}).get("strategy")
                            or route_response.get("forecast_source", "?")
                        ),
                        r=reroutes_applied,
                    )
                )

            last_hybrid_push_sim_time = sim_time

        executed_steps = adapter.run_step_loop(
            max_steps=max_steps,
            stop_when_no_vehicles=config.stop_when_no_vehicles,
            on_step=_on_step
            if (
                args.enable_hybrid_uplink_stub
                or args.enable_emergency_priority
                or args.controlled_count > 0
                or args.emergency_count > 0
                or runtime_logger is not None
                or rl_signal_controller is not None
            )
            else None,
        )
        print(f"[SUMO] Executed steps: {executed_steps}")
        if rl_signal_controller is not None:
            rl_summary = rl_signal_controller.summary()
            print(
                "[SUMO][RL] summary: junctions={j} steps={s} signal_switches={sw}".format(
                    j=rl_summary.get("junctions_controlled", 0),
                    s=rl_summary.get("total_steps", 0),
                    sw=rl_summary.get("signal_switches", 0),
                )
            )
    finally:
        if runtime_logger is not None:
            try:
                runtime_logger.close()
            except Exception:
                pass
        adapter.close(wait=True)


if __name__ == "__main__":
    main()
