# """
# RSU Network Client
# ------------------
# Singleton that the simulator uses to talk to the V2X central server.

# Usage (from simulator.py):
#     from rsu_client import rsu_network_client as rnc
#     rnc.connect("http://localhost:5000")
#     rnc.register_rsus(rsu_nodes, rsu_graph_edges)
#     ...
#     rnc.send_congestion_alert("C", 0.78, metrics_dict)
# """

# import threading
# import socketio          # pip install python-socketio[client]
# from datetime import datetime
# import time


# def _ts():
#     return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# class RSUNetworkClient:
#     def __init__(self):
#         self.sio = socketio.Client(reconnection=True, logger=False, engineio_logger=False)
#         self._connected = False
#         self._thread = None

#         # Tracking global congestion for routing (node -> expiry_timestamp)
#         self.global_congested_nodes = {}
#         self.recently_cleared_nodes = set() 

#         self._register_handlers()

#     # ── Internal SocketIO handlers ─────────────────────────────────────────────
#     def _register_handlers(self):
#         @self.sio.on("connect")
#         def _on_connect():
#             self._connected = True
#             print(f"[{_ts()}][RSU CLIENT] ✓ Connected to V2X server", flush=True)

#         @self.sio.on("disconnect")
#         def _on_disconnect():
#             self._connected = False
#             print(f"[{_ts()}][RSU CLIENT] ✗ Disconnected from V2X server", flush=True)

#         @self.sio.on("register_ack")
#         def _on_ack(data):
#             print(
#                 f"[{_ts()}][RSU CLIENT] ✓ RSUs registered  "
#                 f"count={data.get('rsu_count','?')}", flush=True
#             )

#         @self.sio.on("junction_broadcast")
#         def _on_junction_broadcast(data):
#             rsu_node = data.get("from_rsu", "?")
#             count = data.get("vehicle_count", 0)
#             avg_wait = data.get("avg_wait", 0)
#             timestamp = data.get("timestamp", _ts())

#             print(
#                 f"[{timestamp}][RSU CLIENT] !!! JUNCTION CONGESTION "
#                 f"RSU='{rsu_node}'  vehicles={count}  avg_wait={avg_wait} frames",
#                 flush=True
#             )

#             if rsu_node:
#                 self.global_congested_nodes[rsu_node] = time.time() + 15.0

#         @self.sio.on("junction_clear_broadcast")
#         def _on_junction_clear_broadcast(data):
#             rsu_node = data.get("from_rsu", "?")
#             timestamp = data.get("timestamp", _ts())

#             print(
#                 f"[{timestamp}][RSU CLIENT] ✓ CONGESTION CLEARED "
#                 f"RSU='{rsu_node}'",
#                 flush=True
#             )

#             # Immediately remove from global congestion for routing
#             if rsu_node in self.global_congested_nodes:
#                 del self.global_congested_nodes[rsu_node]
#                 self.recently_cleared_nodes.add(rsu_node)
    
#     def consume_cleared_nodes(self) -> set:
#         """Return and clear the set of nodes that were recently explicitly cleared."""
#         cleared = set(self.recently_cleared_nodes)
#         self.recently_cleared_nodes.clear()
#         return cleared
#     # ── Public API ─────────────────────────────────────────────────────────────
#     def connect(self, server_url: str = "http://192.168.31.48:5000"):
#         """Connect to the V2X server in a background daemon thread."""
#         def _run():
#             try:
#                 self.sio.connect(server_url, wait_timeout=10)
#                 self.sio.wait()
#             except Exception as e:
#                 print(f"[{_ts()}][RSU CLIENT] Connection error: {e}", flush=True)

#         self._thread = threading.Thread(target=_run, daemon=True, name="RSUNetClient")
#         self._thread.start()

#     def register_rsus(self, rsu_nodes: list, rsu_graph_edges: list):
#         """
#         Send RSU topology to server.
#         rsu_nodes       : list of node IDs, e.g. ["C", "E", "M1", ...]
#         rsu_graph_edges : list of [u, v] pairs
#         """
#         if not self._connected:
#             print(f"[{_ts()}][RSU CLIENT] ⚠  Not connected — skipping registration", flush=True)
#             return
#         self.sio.emit("rsu_register", {"nodes": rsu_nodes, "edges": rsu_graph_edges})
    
#     def send_junction_congestion_alert(self, rsu_node: str, vehicle_count: int, avg_wait: float):
#         """Send an aggregated junction congestion alert to the server."""
#         if not self._connected:
#             return
#         payload = {
#             "from_rsu":      rsu_node,
#             "vehicle_count": vehicle_count,
#             "avg_wait":      avg_wait,
#         }
#         self.sio.emit("junction_congestion", payload)

#     def send_junction_clear_alert(self, rsu_node: str):
#         """Send a clearance signal to the server."""
#         if not self._connected:
#             return
#         self.sio.emit("junction_clear", {"from_rsu": rsu_node})

#     def get_active_congested_nodes(self) -> list:
#         """Return a list of nodes that are currently congested (not expired)."""
#         now = time.time()
#         active = []
#         # Filter and cleanup expired entries
#         for node in list(self.global_congested_nodes.keys()):
#             if now < self.global_congested_nodes[node]:
#                 active.append(node)
#             else:
#                 del self.global_congested_nodes[node]
#         return active

#     @property
#     def is_connected(self) -> bool:
#         return self._connected


# # ── Module-level singleton ─────────────────────────────────────────────────────
# rsu_network_client = RSUNetworkClient()


"""
RSU Network Client
------------------
Singleton that the simulator uses to talk to the V2X central server.

Usage (from simulator.py):
    from rsu_client import rsu_network_client as rnc
    rnc.connect("http://localhost:5000")
    rnc.register_rsus(rsu_nodes, rsu_graph_edges)
    ...
    rnc.send_congestion_alert("C", 0.78, metrics_dict)
"""

import threading
import socketio
from datetime import datetime
import time
import os
from dotenv import load_dotenv
load_dotenv()
SERVER_URL=os.getenv("SERVER_URL")

def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class RSUNetworkClient:
    def __init__(self):
        self.sio = socketio.Client(reconnection=True, logger=False, engineio_logger=False)
        self._connected = False
        self._thread = None

        # Tracking global congestion for routing (node -> expiry_timestamp)
        self.global_congested_nodes = {}
        self.recently_cleared_nodes = set()  # for immediate visual cleanup
        self.pending_registration = None

        self._register_handlers()

    # ── Internal SocketIO handlers ─────────────────────────────────────────────
    def _register_handlers(self):
        @self.sio.on("connect")
        def _on_connect():
            self._connected = True
            print(f"[{_ts()}][RSU CLIENT] ✓ Connected to V2X server", flush=True)
            if self.pending_registration:
                print(f"[{_ts()}][RSU CLIENT] [SEND] Sending buffered registration for {len(self.pending_registration.get('nodes', []))} nodes", flush=True)
                self.sio.emit("rsu_register", self.pending_registration)
                self.pending_registration = None

        @self.sio.on("disconnect")
        def _on_disconnect():
            self._connected = False
            print(f"[{_ts()}][RSU CLIENT] ✗ Disconnected from V2X server", flush=True)

        @self.sio.on("register_ack")
        def _on_ack(data):
            print(
                f"[{_ts()}][RSU CLIENT] ✓ RSUs registered  "
                f"count={data.get('rsu_count','?')}", flush=True
            )

        @self.sio.on("junction_broadcast")
        def _on_junction_broadcast(data):
            rsu_node = data.get("from_rsu", "?")
            count = data.get("vehicle_count", 0)
            avg_wait = data.get("avg_wait", 0)
            timestamp = data.get("timestamp", _ts())

            print(
                f"[{timestamp}][RSU CLIENT] !!! JUNCTION CONGESTION "
                f"RSU='{rsu_node}'  vehicles={count}  avg_wait={avg_wait} frames",
                flush=True
            )

            if rsu_node:
                self.global_congested_nodes[rsu_node] = time.time() + 15.0

        @self.sio.on("junction_clear_broadcast")
        def _on_junction_clear_broadcast(data):
            rsu_node = data.get("from_rsu", "?")
            timestamp = data.get("timestamp", _ts())

            print(
                f"[{timestamp}][RSU CLIENT] ✓ CONGESTION CLEARED "
                f"RSU='{rsu_node}'",
                flush=True
            )

            # Immediately remove from global congestion for routing
            if rsu_node in self.global_congested_nodes:
                del self.global_congested_nodes[rsu_node]
                self.recently_cleared_nodes.add(rsu_node)

    def consume_cleared_nodes(self) -> set:
        """Return and clear the set of nodes that were recently explicitly cleared."""
        cleared = set(self.recently_cleared_nodes)
        self.recently_cleared_nodes.clear()
        return cleared

    # ── Public API ─────────────────────────────────────────────────────────────
    def connect(self, server_url):
        """Connect to the V2X server in a background daemon thread."""
        def _run():
            # try:
            #     self.sio.connect(server_url, wait_timeout=10)
            #     self.sio.wait()
            # except Exception as e:
            #     print(f"[{_ts()}][RSU CLIENT] Connection error: {e}", flush=True)
            try:
                print(f"[{_ts()}][RSU CLIENT] [INFO] Attempting to connect to {server_url}...", flush=True)
                self.sio.connect(server_url, wait_timeout=20)
                self.sio.wait()
            except Exception as e:
                print(f"[{_ts()}][RSU CLIENT] [ERROR] Connection failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
        self._thread = threading.Thread(target=_run, daemon=True, name="RSUNetClient")
        self._thread.start()

    def register_rsus(self, rsu_nodes: list, rsu_graph_edges: list):
        """
        Send RSU topology to server.
        rsu_nodes       : list of node IDs, e.g. ["C", "E", "M1", ...]
        rsu_graph_edges : list of [u, v] pairs
        """
        payload = {"nodes": rsu_nodes, "edges": rsu_graph_edges}

        if not self._connected:
            print(f"[{_ts()}][RSU CLIENT] [INFO] Not connected yet -- buffering registration", flush=True)
            self.pending_registration = payload
            return

    def send_junction_congestion_alert(self, rsu_node: str, vehicle_count: int, avg_wait: float):
        """Send an aggregated junction congestion alert to the server."""
        if not self._connected:
            return
        payload = {
            "from_rsu":      rsu_node,
            "vehicle_count": vehicle_count,
            "avg_wait":      avg_wait,
        }
        self.sio.emit("junction_congestion", payload)

    def send_junction_clear_alert(self, rsu_node: str):
        """Send a clearance signal to the server."""
        if not self._connected:
            return
        self.sio.emit("junction_clear", {"from_rsu": rsu_node})

    def get_active_congested_nodes(self) -> list:
        """Return a list of nodes that are currently congested (not expired)."""
        now = time.time()
        active = []
        # Filter and cleanup expired entries
        for node in list(self.global_congested_nodes.keys()):
            if now < self.global_congested_nodes[node]:
                active.append(node)
            else:
                del self.global_congested_nodes[node]
        return active

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── Module-level singleton ─────────────────────────────────────────────────────
rsu_network_client = RSUNetworkClient()
