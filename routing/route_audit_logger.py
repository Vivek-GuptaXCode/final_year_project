from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid


class RouteAuditLogger:
    """Append-only JSONL route audit logger for explainability."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: dict) -> str:
        event_id = str(uuid.uuid4())
        payload = {
            "event_id": event_id,
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        line = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        with self._lock:
            with self._output_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return event_id
