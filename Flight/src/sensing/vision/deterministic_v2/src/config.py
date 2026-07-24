"""Configuration loading for the projection runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESET_PATH = PROJECTION_ROOT / "assets" / "pipeline_presets.json"


@dataclass(frozen=True)
class ProjectionConfig:
    payload: dict[str, Any]
    root: Path = PROJECTION_ROOT

    @classmethod
    def from_path(cls, path: Path | str | None = None) -> "ProjectionConfig":
        preset_path = Path(path) if path is not None else DEFAULT_PRESET_PATH
        preset_path = preset_path.resolve()
        with preset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"projection preset must contain an object: {preset_path}")
        config = cls(payload=payload, root=preset_path.parents[1])
        config.validate()
        return config

    def validate(self) -> None:
        required = [
            "image",
            "camera",
            "gateGeometry",
            "lut",
            "maskLayers",
            "bbox",
            "clipping",
            "innerVoids",
            "innerVoidQuadFit",
            "innerVoidEllipseFit",
            "innerVoidSolvePnP",
            "innerVoidInstanceTracking",
            "globalGateMapping",
        ]
        missing = [key for key in required if key not in self.payload]
        if missing:
            raise ValueError(f"projection preset missing sections: {missing}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image width and height must be positive")
        if not self.enabled_bits:
            raise ValueError("at least one mask layer must be enabled")
        profiles = self.payload.get("profiles", [])
        if profiles is not None and not isinstance(profiles, list):
            raise ValueError("projection profiles must be an array")
        for profile in profiles or []:
            if not isinstance(profile, dict):
                raise ValueError("projection profile entries must be objects")
            profile_id = str(profile.get("profileId") or profile.get("profile_id") or "").strip()
            if not profile_id:
                raise ValueError("projection profile is missing profileId")
            overrides = profile.get("overrides", profile.get("settings", {}))
            if not isinstance(overrides, dict):
                raise ValueError(f"projection profile {profile_id} overrides must be an object")
            for key, value in overrides.items():
                if key not in self.payload or not isinstance(value, dict) or not isinstance(self.payload.get(key), dict):
                    raise ValueError(f"unsupported projection profile override section for {profile_id}: {key}")

    def section(self, key: str) -> dict[str, Any]:
        value = self.payload.get(key)
        return dict(value) if isinstance(value, dict) else {}

    @property
    def image(self) -> dict[str, Any]:
        return self.section("image")

    @property
    def camera(self) -> dict[str, Any]:
        return self.section("camera")

    @property
    def image_width(self) -> int:
        return int(self.image.get("width", 640))

    @property
    def image_height(self) -> int:
        return int(self.image.get("height", 360))

    @property
    def lut_path(self) -> Path:
        raw_path = str(self.section("lut").get("path") or "assets/color_lut_v1.npz")
        path = Path(raw_path)
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def mask_layers(self) -> list[dict[str, Any]]:
        layers = self.payload.get("maskLayers")
        return [dict(item) for item in layers] if isinstance(layers, list) else []

    @property
    def enabled_bits(self) -> list[int]:
        return [int(layer["bit"]) for layer in self.mask_layers if bool(layer.get("enabled", True))]

    @property
    def gate_geometry(self) -> dict[str, Any]:
        return self.section("gateGeometry")

    @property
    def profiles(self) -> list[dict[str, Any]]:
        raw_profiles = self.payload.get("profiles")
        if not isinstance(raw_profiles, list) or not raw_profiles:
            return [
                {
                    "profile_id": "production",
                    "profile_label": "Production",
                    "overrides": {},
                }
            ]
        profiles: list[dict[str, Any]] = []
        for profile in raw_profiles:
            profile_id = str(profile.get("profileId") or profile.get("profile_id") or "").strip()
            profiles.append(
                {
                    "profile_id": profile_id,
                    "profile_label": str(profile.get("profileLabel") or profile.get("profile_label") or f"Config {profile_id}"),
                    "overrides": dict(profile.get("overrides", profile.get("settings", {}))),
                }
            )
        return profiles

    def profile_config(self, profile: dict[str, Any]) -> "ProjectionConfig":
        return self.with_overrides(profile.get("overrides") if isinstance(profile, dict) else {})

    @property
    def inner_square_size_m(self) -> float:
        solve = self.section("innerVoidSolvePnP")
        geometry = self.gate_geometry
        return float(solve.get("innerSquareSizeM", geometry.get("innerSquareSizeM", 1.5)))

    @property
    def ellipse_diameter_m(self) -> float:
        solve = self.section("innerVoidSolvePnP")
        geometry = self.gate_geometry
        return float(solve.get("ellipseDiameterM", geometry.get("ellipseDiameterM", 1.5)))

    def with_overrides(self, overrides: dict[str, Any] | None) -> "ProjectionConfig":
        if not overrides:
            return self
        merged = json.loads(json.dumps(self.payload))
        for key, value in overrides.items():
            if key not in merged or not isinstance(value, dict) or not isinstance(merged.get(key), dict):
                raise ValueError(f"unsupported projection config override section: {key}")
            merged[key].update(value)
        config = ProjectionConfig(payload=merged, root=self.root)
        config.validate()
        return config
