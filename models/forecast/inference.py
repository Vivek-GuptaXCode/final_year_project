from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .common import clamp01
from .feature_builder import FEATURE_NAMES, FeatureState, build_inference_features_from_route_payload

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBClassifier = None  # type: ignore

try:
    import torch  # type: ignore
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None  # type: ignore


def _resolve_model_path(artifact_path: Path, model_file: str) -> Path:
    """Find model file next to artifact or in latest/."""
    p = artifact_path.parent / model_file
    if p.exists():
        return p
    p2 = artifact_path.parent / "latest" / model_file
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Model file not found: {model_file} (tried {artifact_path.parent})")


def _load_model(artifact_path: Path, model_cfg: dict[str, Any]) -> Any:
    """Load model object based on kind and file extension."""
    model_file = model_cfg.get("model_file")
    model_name = str(model_cfg.get("name", "persistence_v1"))
    model_kind = str(model_cfg.get("kind", "rule"))

    if not model_file or model_file == "none":
        return None

    model_path = _resolve_model_path(artifact_path, str(model_file))

    # XGBoost native JSON format
    if model_kind == "xgboost" or model_name in ("xgboost_binary_classifier_v1", "xgboost_v2"):
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost model requested but xgboost is not installed")
        m = XGBClassifier()
        m.load_model(str(model_path))
        return m

    # PyTorch MLP
    if model_kind == "pytorch":
        if not HAS_TORCH:
            raise RuntimeError("pytorch model requested but torch is not installed")
        from .train_phase2_improved import TabularMLP  # lazy import to avoid circular deps
        n_features = len(model_cfg.get("feature_names_ref", [])) or 31
        hidden = tuple(model_cfg.get("hidden_dims") or [256, 128, 64])
        model = TabularMLP(n_features, hidden)
        state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    # Everything else: pickle (LightGBM, HistGB, Ensemble, etc.)
    with model_path.open("rb") as fh:
        return pickle.load(fh)


def _load_scaler(artifact_path: Path, model_cfg: dict[str, Any]) -> Any:
    scaler_file = model_cfg.get("scaler_file")
    if not scaler_file:
        return None
    p = artifact_path.parent / str(scaler_file)
    if not p.exists():
        p2 = artifact_path.parent / "latest" / str(scaler_file)
        if p2.exists():
            p = p2
        else:
            return None
    with p.open("rb") as fh:
        return pickle.load(fh)


class ForecastInferenceEngine:
    """Loads a Phase 2 artifact (v1 or v2) and serves local forecast probabilities."""

    def __init__(
        self,
        artifact_path: Path,
        artifact: dict[str, Any],
        model: Any | None,
        scaler: Any | None = None,
    ) -> None:
        self.artifact_path = artifact_path
        self.artifact = artifact
        self.model = model
        self.scaler = scaler
        self.model_name = str((artifact.get("model") or {}).get("name", "persistence_v1"))
        self.model_kind = str((artifact.get("model") or {}).get("kind", "rule"))

        fc = artifact.get("feature_contract") or {}
        self.feature_names: list[str] = list(fc.get("feature_names", FEATURE_NAMES))
        self._feature_version: str = str(fc.get("version", "v1"))
        self._feature_source: str = str(fc.get("source", "models.forecast.feature_builder"))

        # Per-RSU stateful windows — type depends on feature version
        if self._feature_version == "v2":
            from .feature_builder_v2 import FeatureStateV2
            self.state_by_rsu: dict[str, Any] = {}
            self._FeatureStateClass = FeatureStateV2
        else:
            self.state_by_rsu = {}
            self._FeatureStateClass = FeatureState

    @classmethod
    def from_artifact_path(cls, artifact_path: str | Path) -> "ForecastInferenceEngine":
        path = Path(artifact_path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        model_cfg = payload.get("model") or {}
        # Attach feature_names to model_cfg for pytorch hidden_dims lookup
        fc = payload.get("feature_contract") or {}
        model_cfg["feature_names_ref"] = fc.get("feature_names", [])

        model = _load_model(path, model_cfg)
        scaler = _load_scaler(path, model_cfg)
        return cls(path, payload, model, scaler)

    # ------------------------------------------------------------------ #
    # Feature building                                                     #
    # ------------------------------------------------------------------ #

    def _build_features(self, payload: dict[str, Any]) -> np.ndarray:
        if self._feature_version == "v2":
            from .feature_builder_v2 import build_inference_features_from_route_payload_v2
            return build_inference_features_from_route_payload_v2(payload, self.state_by_rsu)
        return build_inference_features_from_route_payload(payload, self.state_by_rsu)

    # ------------------------------------------------------------------ #
    # Probability prediction                                               #
    # ------------------------------------------------------------------ #

    def _persistence_probability(self, x: np.ndarray) -> float:
        try:
            idx_count = self.feature_names.index("connected_vehicle_count")
            idx_local = self.feature_names.index("congested_local")
        except ValueError:
            return 0.3
        count_term = clamp01(float(x[idx_count]) / 25.0)
        local_term = clamp01(float(x[idx_local]))
        return clamp01(0.10 + 0.55 * local_term + 0.35 * count_term)

    def _predict_probability(self, x: np.ndarray) -> float:
        if self.model_name == "persistence_v1" or self.model is None:
            return self._persistence_probability(x)

        # PyTorch MLP
        if self.model_kind == "pytorch":
            if not HAS_TORCH:
                return self._persistence_probability(x)
            xf = x.astype(np.float32).reshape(1, -1)
            if self.scaler is not None:
                xf = self.scaler.transform(xf).astype(np.float32)
            with torch.no_grad():
                t = torch.tensor(xf)
                p = float(self.model(t).item())
            return clamp01(p)

        data = x.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return clamp01(float(self.model.predict_proba(data)[0, 1]))
        if hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(data)[0])
            return clamp01(1.0 / (1.0 + np.exp(-score)))
        return self._persistence_probability(x)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def predict_from_route_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        x = self._build_features(payload)
        p = self._predict_probability(x)
        confidence = clamp01(max(p, 1.0 - p))
        uncertainty = clamp01(1.0 - confidence)
        return {
            "p_congestion": p,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "model": self.model_name,
            "source": "forecast_artifact",
        }
