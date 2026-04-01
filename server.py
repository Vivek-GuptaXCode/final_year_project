"""
V2X Central Server
------------------
Runs on laptop (or Raspberry Pi later).

RSU graph:
  - Nodes  = RSU junction IDs (e.g. "C", "E", "M1" ...)
  - Edges  = road-level connectivity between RSUs

Events (SocketIO):
  Client → Server : rsu_register      { "nodes": [...], "edges": [[u,v], ...] }
  Client → Server : congestion_alert  { "from_rsu": "C", "score": 0.75, "metrics": {...} }
  Server → All    : congestion_broadcast  (same payload, minus sender)

HTTP:
  GET /graph   → JSON snapshot of the RSU graph
  GET /status  → JSON list of known congestion events
"""

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import networkx as nx
from datetime import datetime
import threading
import os
from pathlib import Path

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "v2x-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─── RSU Graph (server-side) ──────────────────────────────────────────────────
rsu_graph = nx.Graph()

# sid → rsu_node mapping for connected simulator clients
connected_clients = {}   # sid → "simulator" (one simulator client for now)

# Congestion event log
congestion_log = []      # list of dicts
log_lock = threading.Lock()

_forecast_engine = None
_forecast_engine_error = None
_route_audit_logger = None
_route_audit_logger_error = None
_gnn_reroute_engine = None
_gnn_reroute_engine_error = None

# ─── Helpers ──────────────────────────────────────────────────────────────────
def ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def _is_truthy_env(name: str) -> bool:
    value = str(os.getenv(name, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _is_forecast_artifact_enabled() -> bool:
    return _is_truthy_env("HYBRID_ENABLE_FORECAST_MODEL")


def _is_phase3_routing_enabled() -> bool:
    return _is_truthy_env("HYBRID_ENABLE_PHASE3_ROUTING")


def _is_gnn_routing_enabled() -> bool:
    return _is_truthy_env("HYBRID_ENABLE_GNN_ROUTING")


def _load_forecast_engine():
    global _forecast_engine, _forecast_engine_error
    if _forecast_engine is not None or _forecast_engine_error is not None:
        return _forecast_engine

    artifact_path = os.getenv(
        "HYBRID_FORECAST_ARTIFACT",
        "models/forecast/artifacts/latest/forecast_artifact.json",
    )
    try:
        from models.forecast.inference import ForecastInferenceEngine

        _forecast_engine = ForecastInferenceEngine.from_artifact_path(artifact_path)
        log(f"[FORECAST] Loaded inference artifact: {artifact_path}")
    except Exception as exc:
        _forecast_engine_error = str(exc)
        log(f"[FORECAST] Artifact unavailable, fallback to deterministic stub: {exc}")
    return _forecast_engine


def _load_gnn_reroute_engine():
    global _gnn_reroute_engine, _gnn_reroute_engine_error
    if _gnn_reroute_engine is not None or _gnn_reroute_engine_error is not None:
        return _gnn_reroute_engine

    try:
        from routing.gnn_reroute_engine import GNNRerouteConfig, GNNRerouteEngine

        _gnn_reroute_engine = GNNRerouteEngine(config=GNNRerouteConfig.from_env())
        log("[GNN] Graph reroute engine enabled: routing.gnn_reroute_engine")
    except Exception as exc:
        _gnn_reroute_engine_error = str(exc)
        log(f"[GNN] Graph reroute engine unavailable, fallback to current policy: {exc}")
    return _gnn_reroute_engine


def _load_route_audit_logger():
    global _route_audit_logger, _route_audit_logger_error
    if _route_audit_logger is not None or _route_audit_logger_error is not None:
        return _route_audit_logger

    output_path = Path(
        os.getenv(
            "HYBRID_ROUTE_AUDIT_PATH",
            "data/raw/route_audit/route_decisions.jsonl",
        )
    )
    try:
        from routing.route_audit_logger import RouteAuditLogger

        _route_audit_logger = RouteAuditLogger(output_path)
        log(f"[PHASE3] Route audit logger enabled: {output_path}")
    except Exception as exc:
        _route_audit_logger_error = str(exc)
        log(f"[PHASE3] Route audit logger unavailable: {exc}")
    return _route_audit_logger


def _validate_optional_forecast_payload(payload_value):
    if payload_value is None:
        return {}, []

    if not isinstance(payload_value, dict):
        return {}, ["forecast must be an object when provided"]

    errors = []
    normalized = {}
    for key in ("p_congestion", "confidence", "uncertainty"):
        if key not in payload_value:
            continue
        raw_value = payload_value.get(key)
        try:
            parsed_value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"forecast.{key} must be numeric")
            continue
        if parsed_value < 0.0 or parsed_value > 1.0:
            errors.append(f"forecast.{key} must be in [0, 1]")
            continue
        normalized[key] = parsed_value

    if "model" in payload_value and payload_value.get("model") is not None:
        normalized["model"] = str(payload_value.get("model"))

    return normalized, errors


# ─── HTTP endpoints ────────────────────────────────────────────────────────────
@app.route("/graph")
def graph_endpoint():
    """Return the current RSU graph as JSON."""
    data = {
        "nodes": list(rsu_graph.nodes()),
        "edges": [{"from": u, "to": v} for u, v in rsu_graph.edges()],
    }
    return jsonify(data)


@app.route("/graph/register", methods=["POST"])
def graph_register_endpoint():
    """Register RSU topology via HTTP (mirrors the rsu_register SocketIO event).

    Allows external clients (e.g. the SUMO pipeline) to seed the RSU graph used
    by the GNN rerouting engine without needing a SocketIO connection.

    Payload: { "nodes": ["jid1", "jid2", ...], "edges": [["jid1","jid2"], ...] }
    """
    if not request.is_json:
        return jsonify({"status": "error", "message": "Expected JSON payload"}), 400
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"status": "error", "message": "Malformed payload"}), 400

    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])

    if not isinstance(nodes, list):
        return jsonify({"status": "error", "message": "nodes must be a list"}), 400
    if not isinstance(edges, list):
        return jsonify({"status": "error", "message": "edges must be a list"}), 400

    for raw_node in nodes:
        node_id = str(raw_node).strip()
        if node_id:
            rsu_graph.add_node(node_id)

    for raw_edge in edges:
        if isinstance(raw_edge, (list, tuple)) and len(raw_edge) >= 2:
            u = str(raw_edge[0]).strip()
            v = str(raw_edge[1]).strip()
            if u and v and u != v:
                rsu_graph.add_edge(u, v)

    log(
        f"[GRAPH] RSU graph updated via HTTP: nodes={rsu_graph.number_of_nodes()} "
        f"edges={rsu_graph.number_of_edges()}"
    )
    return jsonify({
        "status": "ok",
        "node_count": rsu_graph.number_of_nodes(),
        "edge_count": rsu_graph.number_of_edges(),
    })


@app.route("/status")
def status_endpoint():
    """Return the last 50 congestion events."""
    with log_lock:
        return jsonify(congestion_log[-50:])


@app.route("/route", methods=["POST"])
def route_endpoint():
    """Return a routing decision payload compatible with the hybrid workflow contract.

    The endpoint supports deterministic fallback plus optional GNN/Phase-3 upgrades.
    """
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Expected JSON payload (Content-Type: application/json)",
        }), 400

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({
            "status": "error",
            "message": "Malformed payload: expected a JSON object",
        }), 400

    validation_errors = []

    rsu_id = str(payload.get("rsu_id", "global"))

    sim_time = 0.0
    if "timestamp" in payload:
        try:
            sim_time = float(payload.get("timestamp"))
        except (TypeError, ValueError):
            validation_errors.append("timestamp must be numeric")

    raw_vehicle_ids = payload.get("vehicle_ids", [])
    if raw_vehicle_ids is None:
        raw_vehicle_ids = []
    if not isinstance(raw_vehicle_ids, list):
        validation_errors.append("vehicle_ids must be a list")
        vehicle_ids = []
    else:
        vehicle_ids = [str(vid) for vid in raw_vehicle_ids]

    raw_emergency_ids = payload.get("emergency_vehicle_ids", [])
    if raw_emergency_ids is None:
        raw_emergency_ids = []
    if not isinstance(raw_emergency_ids, list):
        validation_errors.append("emergency_vehicle_ids must be a list")
        emergency_vehicle_ids = []
    else:
        emergency_vehicle_ids = [str(vid) for vid in raw_emergency_ids]

    vehicle_count = len(vehicle_ids)
    if "vehicle_count" in payload:
        try:
            vehicle_count = int(payload.get("vehicle_count"))
            if vehicle_count < 0:
                validation_errors.append("vehicle_count must be >= 0")
        except (TypeError, ValueError):
            validation_errors.append("vehicle_count must be an integer")

    avg_speed_mps = 0.0
    if "avg_speed_mps" in payload:
        try:
            avg_speed_mps = float(payload.get("avg_speed_mps"))
            if avg_speed_mps < 0:
                validation_errors.append("avg_speed_mps must be >= 0")
        except (TypeError, ValueError):
            validation_errors.append("avg_speed_mps must be numeric")

    request_forecast, forecast_errors = _validate_optional_forecast_payload(payload.get("forecast"))
    validation_errors.extend(forecast_errors)

    if validation_errors:
        return jsonify({
            "status": "error",
            "message": "Malformed payload",
            "details": validation_errors,
        }), 400

    # Default deterministic surrogate preserves prior behavior.
    count_score = min(1.0, max(0.0, vehicle_count / 50.0))
    speed_score = 1.0 - min(1.0, max(0.0, avg_speed_mps / 15.0))
    p_congestion = max(0.0, min(1.0, 0.6 * count_score + 0.4 * speed_score))
    confidence = max(0.5, min(0.9, 0.9 - abs(p_congestion - 0.5)))
    model_label = "gnn_surrogate_v1"
    forecast_source = "deterministic_stub"
    gnn_decision = None
    prioritized_vehicle_ids = list(vehicle_ids)

    # Priority 1: explicit forecast fields in request payload (fully additive).
    if request_forecast:
        p_congestion = float(request_forecast.get("p_congestion", p_congestion))
        confidence = float(request_forecast.get("confidence", confidence))
        if "uncertainty" in request_forecast and "confidence" not in request_forecast:
            confidence = 1.0 - float(request_forecast.get("uncertainty"))
        confidence = max(0.0, min(1.0, confidence))
        p_congestion = max(0.0, min(1.0, p_congestion))
        model_label = str(request_forecast.get("model", "payload_forecast_v1"))
        forecast_source = "request_payload"

    # Priority 2: graph message-passing reroute inference behind feature flag.
    elif _is_gnn_routing_enabled():
        gnn_engine = _load_gnn_reroute_engine()
        if gnn_engine is not None:
            try:
                gnn_decision = gnn_engine.predict(
                    rsu_graph=rsu_graph,
                    rsu_id=rsu_id,
                    sim_timestamp=sim_time,
                    vehicle_ids=vehicle_ids,
                    emergency_vehicle_ids=emergency_vehicle_ids,
                    vehicle_count=vehicle_count,
                    avg_speed_mps=avg_speed_mps,
                )
                p_congestion = max(0.0, min(1.0, float(gnn_decision.get("p_congestion", p_congestion))))
                confidence = max(0.0, min(1.0, float(gnn_decision.get("confidence", confidence))))
                model_label = str(gnn_decision.get("model", "gnn_reroute_v1"))
                forecast_source = str(gnn_decision.get("source", "graph_message_passing"))

                vehicle_priority_order = gnn_decision.get("vehicle_priority_order", [])
                if isinstance(vehicle_priority_order, list):
                    seen_priorities = set()
                    ordered: list[str] = []
                    vehicle_id_set = set(vehicle_ids)
                    for raw_vid in vehicle_priority_order:
                        vid = str(raw_vid)
                        if not vid or vid in seen_priorities or vid not in vehicle_id_set:
                            continue
                        ordered.append(vid)
                        seen_priorities.add(vid)
                    if ordered:
                        ordered.extend([vid for vid in vehicle_ids if vid not in seen_priorities])
                        prioritized_vehicle_ids = ordered
            except Exception as exc:
                gnn_decision = None
                log(f"[GNN] Inference failed, fallback to deterministic stub: {exc}")

    # Priority 3: local artifact model behind feature flag.
    elif _is_forecast_artifact_enabled():
        engine = _load_forecast_engine()
        if engine is not None:
            try:
                forecast = engine.predict_from_route_payload(payload)
                p_congestion = max(0.0, min(1.0, float(forecast.get("p_congestion", p_congestion))))
                confidence = max(0.0, min(1.0, float(forecast.get("confidence", confidence))))
                model_label = str(forecast.get("model", "phase2_forecast_artifact_v1"))
                forecast_source = str(forecast.get("source", "forecast_artifact"))
            except Exception as exc:
                log(f"[FORECAST] Inference failed, fallback to deterministic stub: {exc}")

    uncertainty = max(0.0, min(1.0, 1.0 - confidence))
    if isinstance(gnn_decision, dict) and str(gnn_decision.get("risk_level", "")).lower() in {
        "low",
        "medium",
        "high",
    }:
        risk_level = str(gnn_decision.get("risk_level", "low")).lower()
    elif p_congestion >= 0.70:
        risk_level = "high"
    elif p_congestion >= 0.45:
        risk_level = "medium"
    else:
        risk_level = "low"

    emergency_active = len(emergency_vehicle_ids) > 0
    routing_engine = {
        "primary": "gnn_surrogate",
        "fallback": "dijkstra",
    }
    recommended_action = {
        "reroute_bias": "avoid_hotspots" if risk_level != "low" else "normal",
        "signal_priority": "inbound_relief" if risk_level == "high" else "balanced",
        "reroute_enabled": emergency_active or (risk_level != "low"),
        "reroute_mode": "dijkstra" if emergency_active else "gnn_effort",
        "reroute_fraction": 1.0 if emergency_active else (0.35 if risk_level == "high" else (0.20 if risk_level == "medium" else 0.0)),
        "min_confidence": 0.0 if emergency_active else 0.50,
        "fallback_algorithm": "dijkstra",
    }
    route_directives = []

    if isinstance(gnn_decision, dict):
        routing_engine["primary"] = str(gnn_decision.get("model", "gnn_reroute_v1"))

        gnn_recommended_action = gnn_decision.get("recommended_action")
        if isinstance(gnn_recommended_action, dict):
            recommended_action = {
                **recommended_action,
                **gnn_recommended_action,
            }

        gnn_route_directives = gnn_decision.get("route_directives")
        if isinstance(gnn_route_directives, list):
            route_directives = gnn_route_directives

    response = {
        "status": "ok",
        "rsu_id": rsu_id,
        "model": model_label,
        "forecast_source": forecast_source,
        "routing_engine": routing_engine,
        "p_congestion": p_congestion,
        "uncertainty": uncertainty,
        "confidence": confidence,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "emergency_action": {
            "active": emergency_active,
            "vehicle_ids": emergency_vehicle_ids,
            "strategy": "optimal_route_plus_corridor_preemption" if emergency_active else "none",
            "traffic_control": "stop_non_emergency_on_corridor" if emergency_active else "normal_hybrid_control",
        },
        "sim_timestamp": sim_time,
        "server_timestamp": ts(),
    }

    if route_directives:
        response["route_directives"] = route_directives

    if isinstance(gnn_decision, dict):
        response["gnn_routing"] = {
            "enabled": True,
            "strategy": str(gnn_decision.get("strategy", "gnn_primary")),
            "diagnostics": gnn_decision.get("diagnostics", {}),
        }

    if _is_phase3_routing_enabled():
        try:
            from routing.phase3_risk_router import Phase3RoutingConfig, build_phase3_decision

            phase3_decision = build_phase3_decision(
                rsu_id=rsu_id,
                sim_timestamp=sim_time,
                vehicle_ids=prioritized_vehicle_ids,
                emergency_vehicle_ids=emergency_vehicle_ids,
                vehicle_count=vehicle_count,
                avg_speed_mps=avg_speed_mps,
                p_congestion=p_congestion,
                confidence=confidence,
                uncertainty=uncertainty,
                config=Phase3RoutingConfig.from_env(),
            )

            response["routing_engine"] = phase3_decision.get("routing_engine", response["routing_engine"])
            response["risk_level"] = phase3_decision.get("risk_level", response["risk_level"])
            response["recommended_action"] = phase3_decision.get(
                "recommended_action", response["recommended_action"]
            )

            if "route_directives" in phase3_decision:
                response["route_directives"] = phase3_decision.get("route_directives", [])

            phase3_payload = phase3_decision.get("phase3", {})
            if isinstance(phase3_payload, dict):
                if isinstance(gnn_decision, dict):
                    phase3_payload.setdefault(
                        "gnn_context",
                        {
                            "strategy": str(gnn_decision.get("strategy", "gnn_primary")),
                            "model": str(gnn_decision.get("model", "gnn_reroute_v1")),
                            "source": str(gnn_decision.get("source", "graph_message_passing")),
                        },
                    )
                response["phase3"] = phase3_payload

            audit_logger = _load_route_audit_logger()
            if audit_logger is not None:
                audit_id = audit_logger.log(
                    {
                        "rsu_id": rsu_id,
                        "sim_timestamp": sim_time,
                        "vehicle_count": vehicle_count,
                        "avg_speed_mps": avg_speed_mps,
                        "emergency_vehicle_count": len(emergency_vehicle_ids),
                        "forecast": {
                            "model": model_label,
                            "source": forecast_source,
                            "p_congestion": p_congestion,
                            "confidence": confidence,
                            "uncertainty": uncertainty,
                        },
                        "routing_engine": response.get("routing_engine", {}),
                        "risk_level": response.get("risk_level", "unknown"),
                        "recommended_action": response.get("recommended_action", {}),
                        "route_directives": response.get("route_directives", []),
                        "phase3": response.get("phase3", {}),
                    }
                )
                response["route_audit_id"] = audit_id
                phase3 = response.get("phase3")
                if isinstance(phase3, dict):
                    phase3["audit_id"] = audit_id
        except Exception as exc:
            log(f"[PHASE3] Routing decision failed, fallback to legacy policy: {exc}")

    return jsonify(response)


# ─── SocketIO events ───────────────────────────────────────────────────────────
@socketio.on("connect")
def handle_connect():
    log(f"[CONNECT] Client connected  sid={request.sid}")


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    connected_clients.pop(sid, None)
    log(f"[DISCONNECT] Client disconnected  sid={sid}")


@socketio.on("rsu_register")
def handle_register(data):
    """
    Payload:
        { "nodes": ["C","E","F","G","H","M1","M2","J"],
          "edges": [["C","E"], ["E","G"], ...] }
    """
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # Build / update the RSU graph
    for node in nodes:
        rsu_graph.add_node(node)
    for u, v in edges:
        rsu_graph.add_edge(u, v)

    connected_clients[request.sid] = "simulator"

    log(f"[REGISTER] RSU graph updated  nodes={nodes}  edges={len(edges)}")
    emit("register_ack", {"status": "ok", "rsu_count": len(nodes)})


@socketio.on("junction_congestion")
def handle_junction_congestion(data):
    """
    Payload:
        { "from_rsu": "C",
          "vehicle_count": 5,
          "avg_wait": 350 }
    """
    from_rsu = data.get("from_rsu", "?")
    count    = data.get("vehicle_count", 0)
    avg_wait = data.get("avg_wait", 0)

    log(
        f"[LONG WAIT] Junction '{from_rsu}' reports {count} vehicles "
        f"with avg wait {avg_wait} frames"
    )
    # Log the event
    event = {
        "type": "congestion",
        "from_rsu": from_rsu,
        "vehicle_count": count,
        "avg_wait": avg_wait,
        "timestamp": ts()
    }
    with log_lock:
        congestion_log.append(event)
    # Broadcast to all
    broadcast_payload = {
        "from_rsu":      from_rsu,
        "vehicle_count": count,
        "avg_wait":      avg_wait,
        "timestamp":     ts(),
    }
    emit("junction_broadcast", broadcast_payload, broadcast=True)


@socketio.on("junction_clear")
def handle_junction_clear(data):
    """
    Payload: { "from_rsu": "C" }
    """
    from_rsu = data.get("from_rsu", "?")
    log(f"[CLEAR] Junction '{from_rsu}' traffic resumed")
    # Log the event
    event = {
        "type": "clear",
        "from_rsu": from_rsu,
        "timestamp": ts()
    }
    with log_lock:
        congestion_log.append(event)
    broadcast_payload = {
        "from_rsu":  from_rsu,
        "timestamp": ts(),
    }
    emit("junction_clear_broadcast", broadcast_payload, broadcast=True)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  V2X Central Server  |  Flask-SocketIO")
    print("  Listening on  http://0.0.0.0:5000")
    print("  Endpoints:  GET /graph   GET /status")
    print("=" * 60)
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True,
        use_reloader=False,
        log_output=True,
    )
