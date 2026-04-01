from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile


SCHEMA_VERSION = "1.0"


RSU_FIELDNAMES = [
    "timestamp_s",
    "frame_idx",
    "rsu_node",
    "connected_vehicle_count",
    "registered_telemetry_count",
    "packets_received",
    "bytes_received",
    "avg_latency_s",
    "congested_local",
    "congested_global",
]


EDGE_FIELDNAMES = [
    "timestamp_s",
    "frame_idx",
    "edge_u",
    "edge_v",
    "length",
    "capacity",
    "traffic",
    "traffic_ratio",
]


class SimulationDataLogger:
    """Log RSU and edge snapshots at 1 Hz into run-scoped CSV files."""

    def __init__(self, run_dir: str | Path, run_metadata: dict):
        self.run_dir = Path(run_dir)
        self.raw_dir = self.run_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.rsu_path = self.raw_dir / "rsu_features_1hz.csv"
        self.edge_path = self.raw_dir / "edge_flow_1hz.csv"
        self.manifest_path = self.raw_dir / "logger_manifest.json"

        self._next_sample_second = 0

        self._rsu_file = self.rsu_path.open("w", newline="", encoding="utf-8")
        self._edge_file = self.edge_path.open("w", newline="", encoding="utf-8")

        self._rsu_writer = csv.DictWriter(self._rsu_file, fieldnames=RSU_FIELDNAMES)
        self._edge_writer = csv.DictWriter(self._edge_file, fieldnames=EDGE_FIELDNAMES)
        self._rsu_writer.writeheader()
        self._edge_writer.writeheader()

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run": run_metadata,
            "files": {
                "rsu_features": {
                    "path": str(self.rsu_path),
                    "fieldnames": RSU_FIELDNAMES,
                    "interval_hz": 1,
                },
                "edge_flow": {
                    "path": str(self.edge_path),
                    "fieldnames": EDGE_FIELDNAMES,
                    "interval_hz": 1,
                },
            },
        }
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def maybe_log(
        self,
        *,
        sim_time_seconds: float,
        frame_idx: int,
        rsus,
        vehicles,
        network,
        local_congested_nodes,
        global_congested_nodes,
    ) -> None:
        current_second = int(sim_time_seconds)
        if current_second < self._next_sample_second:
            return

        self._next_sample_second = current_second + 1
        self._write_rsu_rows(
            timestamp_s=current_second,
            frame_idx=frame_idx,
            rsus=rsus,
            vehicles=vehicles,
            local_congested_nodes=local_congested_nodes,
            global_congested_nodes=global_congested_nodes,
        )
        self._write_edge_rows(
            timestamp_s=current_second,
            frame_idx=frame_idx,
            network=network,
        )

        self._rsu_file.flush()
        self._edge_file.flush()

    def _write_rsu_rows(
        self,
        *,
        timestamp_s: int,
        frame_idx: int,
        rsus,
        vehicles,
        local_congested_nodes,
        global_congested_nodes,
    ) -> None:
        for rsu in rsus:
            connected_vehicle_count = sum(
                1
                for v in vehicles
                if getattr(getattr(v, "obu", None), "connected_rsu", None) is rsu
            )

            if rsu.latencies:
                avg_latency = sum(rsu.latencies) / len(rsu.latencies)
            else:
                avg_latency = 0.0

            row = {
                "timestamp_s": timestamp_s,
                "frame_idx": frame_idx,
                "rsu_node": rsu.node,
                "connected_vehicle_count": connected_vehicle_count,
                "registered_telemetry_count": len(rsu.registered_vehicles),
                "packets_received": rsu.packets_received,
                "bytes_received": rsu.bytes_received,
                "avg_latency_s": round(avg_latency, 6),
                "congested_local": int(rsu.node in local_congested_nodes),
                "congested_global": int(rsu.node in global_congested_nodes),
            }
            self._rsu_writer.writerow(row)

    def _write_edge_rows(self, *, timestamp_s: int, frame_idx: int, network) -> None:
        for u, v, data in network.graph.edges(data=True):
            capacity = float(data.get("capacity", 0.0))
            traffic = float(data.get("traffic", 0.0))
            ratio = traffic / capacity if capacity > 0 else 0.0

            row = {
                "timestamp_s": timestamp_s,
                "frame_idx": frame_idx,
                "edge_u": u,
                "edge_v": v,
                "length": float(data.get("length", 0.0)),
                "capacity": capacity,
                "traffic": traffic,
                "traffic_ratio": round(ratio, 6),
            }
            self._edge_writer.writerow(row)

    def close(self) -> None:
        self._rsu_file.close()
        self._edge_file.close()


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically to reduce risk of partial manifest files."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class SumoSimulationDataLogger:
    """Log 1 Hz RSU and edge snapshots directly from SUMO TraCI/libsumo state."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        run_metadata: dict,
        net_file: str | Path,
        rsu_alias_table: list[tuple[str, str, float, float]],
        rsu_range_m: float,
        schema_version: str = SCHEMA_VERSION,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.rsu_path = self.run_dir / "rsu_features_1hz.csv"
        self.edge_path = self.run_dir / "edge_flow_1hz.csv"
        self.manifest_path = self.run_dir / "logger_manifest.json"

        self._rsu_range_m = max(1.0, float(rsu_range_m))
        self._rsu_range_sq = self._rsu_range_m * self._rsu_range_m
        self._next_sample_second = 0

        self._rsu_points: list[tuple[str, str, float, float]] = []
        for alias, junction_id, x, y in rsu_alias_table:
            self._rsu_points.append((f"RSU_{alias}", junction_id, float(x), float(y)))

        self._packets_received: dict[str, int] = {rsu_node: 0 for rsu_node, *_ in self._rsu_points}
        self._bytes_received: dict[str, int] = {rsu_node: 0 for rsu_node, *_ in self._rsu_points}

        self._edge_meta = self._load_edge_meta(Path(net_file))
        self._edge_ids = sorted(self._edge_meta.keys())

        self._rsu_file = self.rsu_path.open("w", newline="", encoding="utf-8")
        self._edge_file = self.edge_path.open("w", newline="", encoding="utf-8")
        self._rsu_writer = csv.DictWriter(self._rsu_file, fieldnames=RSU_FIELDNAMES)
        self._edge_writer = csv.DictWriter(self._edge_file, fieldnames=EDGE_FIELDNAMES)
        self._rsu_writer.writeheader()
        self._edge_writer.writeheader()

        manifest = {
            "schema_version": schema_version,
            "run": run_metadata,
            "logger": {
                "type": "sumo_runtime_logger",
                "interval_hz": 1,
                "rsu_range_m": self._rsu_range_m,
                "rsu_count": len(self._rsu_points),
                "edge_count": len(self._edge_ids),
            },
            "files": {
                "rsu_features": {
                    "path": str(self.rsu_path),
                    "fieldnames": RSU_FIELDNAMES,
                    "interval_hz": 1,
                },
                "edge_flow": {
                    "path": str(self.edge_path),
                    "fieldnames": EDGE_FIELDNAMES,
                    "interval_hz": 1,
                },
            },
            "rsus": [
                {
                    "rsu_node": rsu_node,
                    "junction_id": junction_id,
                    "x": x,
                    "y": y,
                }
                for rsu_node, junction_id, x, y in self._rsu_points
            ],
        }
        _atomic_write_json(self.manifest_path, manifest)

    @staticmethod
    def _load_edge_meta(net_file: Path) -> dict[str, dict[str, float | str]]:
        edge_meta: dict[str, dict[str, float | str]] = {}
        try:
            import xml.etree.ElementTree as ET

            net_root = ET.parse(net_file).getroot()
        except Exception:
            return edge_meta

        for edge in net_root.findall("edge"):
            edge_id = edge.attrib.get("id")
            if not edge_id or edge_id.startswith(":"):
                continue

            from_junction = edge.attrib.get("from", "")
            to_junction = edge.attrib.get("to", "")

            lane_lengths: list[float] = []
            for lane in edge.findall("lane"):
                try:
                    lane_lengths.append(float(lane.attrib.get("length", "0")))
                except Exception:
                    continue

            lane_count = max(1, len(lane_lengths))
            avg_length = (sum(lane_lengths) / len(lane_lengths)) if lane_lengths else 0.0

            # Approximate standing capacity using ~7.5m per vehicle equivalent.
            capacity = max(1.0, lane_count * max(1.0, avg_length) / 7.5)

            edge_meta[edge_id] = {
                "from_junction": from_junction,
                "to_junction": to_junction,
                "length": float(avg_length),
                "capacity": float(capacity),
            }

        return edge_meta

    def _collect_vehicle_samples(
        self,
        *,
        traci_module,
        vehicle_ids: list[str] | None,
    ) -> list[tuple[float, float, float]]:
        ids = vehicle_ids
        if ids is None:
            try:
                ids = list(traci_module.vehicle.getIDList())
            except Exception:
                return []

        samples: list[tuple[float, float, float]] = []
        for vehicle_id in ids:
            try:
                x, y = traci_module.vehicle.getPosition(vehicle_id)
                speed = float(traci_module.vehicle.getSpeed(vehicle_id))
            except Exception:
                continue
            samples.append((float(x), float(y), speed))
        return samples

    def _collect_edge_snapshot(self, *, traci_module) -> tuple[list[dict[str, float | str]], bool]:
        rows: list[dict[str, float | str]] = []
        global_congested = False

        for edge_id in self._edge_ids:
            meta = self._edge_meta.get(edge_id, {})
            capacity = float(meta.get("capacity", 0.0))
            try:
                traffic = float(traci_module.edge.getLastStepVehicleNumber(edge_id))
            except Exception:
                traffic = 0.0

            ratio = (traffic / capacity) if capacity > 0 else 0.0
            if ratio >= 0.8:
                global_congested = True

            rows.append(
                {
                    "edge_id": edge_id,
                    "edge_u": str(meta.get("from_junction", "")),
                    "edge_v": str(meta.get("to_junction", "")),
                    "length": float(meta.get("length", 0.0)),
                    "capacity": capacity,
                    "traffic": traffic,
                    "traffic_ratio": ratio,
                }
            )

        return rows, global_congested

    def maybe_log(
        self,
        *,
        sim_time_seconds: float,
        frame_idx: int,
        traci_module,
        vehicle_ids: list[str] | None = None,
    ) -> None:
        current_second = int(sim_time_seconds)
        if current_second < self._next_sample_second:
            return

        self._next_sample_second = current_second + 1

        vehicle_samples = self._collect_vehicle_samples(traci_module=traci_module, vehicle_ids=vehicle_ids)
        edge_rows, global_congested = self._collect_edge_snapshot(traci_module=traci_module)

        self._write_rsu_rows(
            timestamp_s=current_second,
            frame_idx=frame_idx,
            vehicle_samples=vehicle_samples,
            global_congested=global_congested,
        )
        self._write_edge_rows(
            timestamp_s=current_second,
            frame_idx=frame_idx,
            edge_rows=edge_rows,
        )

        self._rsu_file.flush()
        self._edge_file.flush()

    def _write_rsu_rows(
        self,
        *,
        timestamp_s: int,
        frame_idx: int,
        vehicle_samples: list[tuple[float, float, float]],
        global_congested: bool,
    ) -> None:
        for rsu_node, _junction_id, rx, ry in self._rsu_points:
            connected_count = 0
            speed_sum = 0.0
            for x, y, speed in vehicle_samples:
                dx = x - rx
                dy = y - ry
                if (dx * dx + dy * dy) <= self._rsu_range_sq:
                    connected_count += 1
                    speed_sum += speed

            avg_speed = (speed_sum / connected_count) if connected_count > 0 else 0.0
            congested_local = int(connected_count >= 5 and avg_speed <= 5.0)

            self._packets_received[rsu_node] = self._packets_received.get(rsu_node, 0) + connected_count
            self._bytes_received[rsu_node] = self._bytes_received.get(rsu_node, 0) + (connected_count * 128)

            if connected_count <= 0:
                avg_latency_s = 0.0
            else:
                avg_latency_s = min(1.0, 0.02 + (0.002 * connected_count) + max(0.0, (5.0 - avg_speed) * 0.003))

            row = {
                "timestamp_s": timestamp_s,
                "frame_idx": frame_idx,
                "rsu_node": rsu_node,
                "connected_vehicle_count": connected_count,
                "registered_telemetry_count": connected_count,
                "packets_received": self._packets_received.get(rsu_node, 0),
                "bytes_received": self._bytes_received.get(rsu_node, 0),
                "avg_latency_s": round(avg_latency_s, 6),
                "congested_local": congested_local,
                "congested_global": int(global_congested),
            }
            self._rsu_writer.writerow(row)

    def _write_edge_rows(
        self,
        *,
        timestamp_s: int,
        frame_idx: int,
        edge_rows: list[dict[str, float | str]],
    ) -> None:
        for edge_row in edge_rows:
            row = {
                "timestamp_s": timestamp_s,
                "frame_idx": frame_idx,
                "edge_u": edge_row["edge_u"],
                "edge_v": edge_row["edge_v"],
                "length": float(edge_row["length"]),
                "capacity": float(edge_row["capacity"]),
                "traffic": float(edge_row["traffic"]),
                "traffic_ratio": round(float(edge_row["traffic_ratio"]), 6),
            }
            self._edge_writer.writerow(row)

    def close(self) -> None:
        self._rsu_file.close()
        self._edge_file.close()
