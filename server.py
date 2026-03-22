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

from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
import networkx as nx
from datetime import datetime
import threading

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

# ─── Helpers ──────────────────────────────────────────────────────────────────
def ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


# ─── HTTP endpoints ────────────────────────────────────────────────────────────
@app.route("/graph")
def graph_endpoint():
    """Return the current RSU graph as JSON."""
    data = {
        "nodes": list(rsu_graph.nodes()),
        "edges": [{"from": u, "to": v} for u, v in rsu_graph.edges()],
    }
    return jsonify(data)


@app.route("/status")
def status_endpoint():
    """Return the last 50 congestion events."""
    with log_lock:
        return jsonify(congestion_log[-50:])


# ─── SocketIO events ───────────────────────────────────────────────────────────
@socketio.on("connect")
def handle_connect():
    from flask import request
    log(f"[CONNECT] Client connected  sid={request.sid}")


@socketio.on("disconnect")
def handle_disconnect():
    from flask import request
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
    from flask import request
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
    socketio.run(app, host="0.0.0.0", port=5000, debug=False,allow_unsafe_werkzeug=True)
