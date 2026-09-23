"""Frame-by-frame gate pose regression from CNN raw logits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .cnn_ingress import RawLogitsFrame
from .model import build_model, load_compatible_state_dict
from .quad_geometry import MODEL_HEIGHT, MODEL_WIDTH, fit_contour_quad_from_pixels, model_to_source_xy


REGRESSOR_SCHEMA_VERSION = "gate_position_multihead_stride4_v1"
DEFAULT_REGRESSOR_CHECKPOINT = Path(__file__).resolve().parent / "regressor_last.pt"
SOURCE_IMAGE_WIDTH = 640
SOURCE_IMAGE_HEIGHT = 360
CAMERA_FX_PX = 320.0
CAMERA_FY_PX = 320.0
CAMERA_CX_PX = 320.0
CAMERA_CY_PX = 180.0
ADAPTER_MAP_CHANNELS = [
    "visible_gate_mask",
    "circumscribed_centerpoint_heatmap",
    "gate_depth_linear_reward",
    "distractor_mask",
]
DEFAULT_SCALAR_FEATURES = [
    "center_x_norm",
    "center_y_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "component_area_frac",
    "component_confidence",
    "component_mean_probability",
    "center_peak_dx_norm",
    "center_peak_dy_norm",
    "center_peak_score",
    "pose_confidence_at_center",
    "base_position_x_norm_at_center",
    "base_position_y_norm_at_center",
    "base_position_z_norm_at_center",
]
DEFAULT_MASK_GEOMETRY_FEATURES = [
    "mask_weight_frac",
    "mask_centroid_dx_norm",
    "mask_centroid_dy_norm",
    "mask_major_sigma_norm",
    "mask_minor_sigma_norm",
    "mask_major_p90_norm",
    "mask_minor_p90_norm",
    "mask_axis_ratio_log",
    "mask_anisotropy",
    "mask_axis_cos2",
    "mask_axis_sin2",
]
DEFAULT_CROP_SIZE = 48
POSITION_BOUNDS_M = {"x": (-30.0, 30.0), "y": (-20.0, 20.0), "z": (0.0, 45.0)}


@dataclass(frozen=True)
class GateRegression:
    center_model_px: tuple[float, float]
    center_px: tuple[float, float]
    quad_center_model_px: tuple[float, float]
    quad_model_px: list[list[float]]
    position_xyz: tuple[float, float, float]
    orientation_xyz: tuple[float, float, float]
    confidence: float
    distance_camera_m: float
    quad_fit_source: str
    quad_is_fallback: bool


@dataclass(frozen=True)
class FrameRegression:
    frame_id: int
    sim_time_ns: int
    source_width: int
    source_height: int
    gates: list[GateRegression]


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _denormalize_tensor(values: torch.Tensor, low: float, high: float, *, clamp: float = 1.5) -> torch.Tensor:
    clipped = values.clamp(-float(clamp), float(clamp))
    return float(low) + ((clipped + 1.0) * 0.5 * float(high - low))


def _denormalize_z_tensor(z_norm: torch.Tensor) -> torch.Tensor:
    low, high = POSITION_BOUNDS_M["z"]
    return _denormalize_tensor(z_norm[:, 0:1], low, high)


def _source_xy_z_to_camera_xyz(
    x_px: float,
    y_px: float,
    z_m: float,
    *,
    source_width: float,
    source_height: float,
) -> tuple[float, float, float]:
    sx = float(source_width) / float(SOURCE_IMAGE_WIDTH)
    sy = float(source_height) / float(SOURCE_IMAGE_HEIGHT)
    fx = CAMERA_FX_PX * sx
    fy = CAMERA_FY_PX * sy
    cx = CAMERA_CX_PX * sx
    cy = CAMERA_CY_PX * sy
    z = float(z_m)
    x = ((float(x_px) - cx) / max(fx, 1.0e-6)) * z
    y = ((cy - float(y_px)) / max(fy, 1.0e-6)) * z
    return x, y, z


def _model_xy_z_to_camera_xyz(
    center_model_px: tuple[float, float],
    z_m: float,
    *,
    source_width: int,
    source_height: int,
) -> tuple[float, float, float]:
    source_x, source_y = model_to_source_xy(center_model_px[0], center_model_px[1], source_width, source_height)
    return _source_xy_z_to_camera_xyz(source_x, source_y, z_m, source_width=source_width, source_height=source_height)


def _decode_gate_components(
    visible_gate_probability: torch.Tensor,
    *,
    threshold: float,
    min_component_area: int,
    max_candidates: int,
    source_width: int,
    source_height: int,
) -> list[dict[str, Any]]:
    probability = visible_gate_probability.detach().cpu().numpy().astype(np.float32, copy=False)
    binary = (probability >= float(threshold)).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[dict[str, Any]] = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < int(min_component_area):
            continue
        ys, xs = np.nonzero(labels == component_id)
        if xs.size <= 0:
            continue
        values = probability[ys, xs]
        x_min = int(stats[component_id, cv2.CC_STAT_LEFT])
        y_min = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        component_center_model = [float(xs.mean()), float(ys.mean())]
        component_center_px = model_to_source_xy(component_center_model[0], component_center_model[1], source_width, source_height)
        source_min = model_to_source_xy(float(x_min), float(y_min), source_width, source_height)
        source_max = model_to_source_xy(float(x_min + width), float(y_min + height), source_width, source_height)
        quad = fit_contour_quad_from_pixels(
            xs.astype(np.float32),
            ys.astype(np.float32),
            source_width=source_width,
            source_height=source_height,
        )
        quad_center_model = [float(value) for value in quad["quad_center_model_px"]]
        quad_center_px = [float(value) for value in quad["quad_center_px"]]
        components.append(
            {
                "component_id": int(component_id),
                "area_model_px": int(area),
                "bbox_model_px": [float(x_min), float(y_min), float(width), float(height)],
                "bbox_px": [
                    float(source_min[0]),
                    float(source_min[1]),
                    float(source_max[0] - source_min[0]),
                    float(source_max[1] - source_min[1]),
                ],
                "component_center_model_px": component_center_model,
                "component_center_px": [float(component_center_px[0]), float(component_center_px[1])],
                "center_model_px": quad_center_model,
                "center_px": quad_center_px,
                "confidence": float(values.max()),
                "mean_probability": float(values.mean()),
                **quad,
            }
        )

    components.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return components[: max(0, int(max_candidates))]


def _contour_quad_center_heatmap(
    candidates: list[dict[str, Any]],
    *,
    sigma_model_px: float = 2.25,
) -> torch.Tensor:
    heatmap = np.zeros((MODEL_HEIGHT, MODEL_WIDTH), dtype=np.float32)
    yy, xx = np.mgrid[0:MODEL_HEIGHT, 0:MODEL_WIDTH].astype(np.float32)
    sigma = max(0.25, float(sigma_model_px))
    for candidate in candidates:
        cx, cy = [float(value) for value in candidate["quad_center_model_px"]]
        confidence = float(candidate["confidence"])
        gaussian = confidence * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2.0 * sigma * sigma))
        heatmap = np.maximum(heatmap, gaussian.astype(np.float32))
    return torch.from_numpy(np.clip(heatmap, 0.0, 1.0))


def _center_peak_from_map(center_probability: torch.Tensor) -> dict[str, Any]:
    flat = int(torch.argmax(center_probability).item())
    width = int(center_probability.shape[-1])
    x = flat % width
    y = flat // width
    return {"model_px": [int(x), int(y)], "score": float(center_probability[y, x].item())}


def _channel_value(values: torch.Tensor, x: float, y: float) -> float:
    xx = max(0, min(int(values.shape[-1]) - 1, int(round(float(x)))))
    yy = max(0, min(int(values.shape[-2]) - 1, int(round(float(y)))))
    return float(values[yy, xx].item())


def _scalar_features_for_candidate(
    maps: torch.Tensor,
    map_channels: list[str],
    candidate: dict[str, Any],
    *,
    center_peak: dict[str, Any],
) -> dict[str, float]:
    channel_index = {name: index for index, name in enumerate(map_channels)}
    center_x, center_y = candidate["center_model_px"]
    bbox_w = float(candidate["bbox_model_px"][2])
    bbox_h = float(candidate["bbox_model_px"][3])
    peak_x = float(center_peak["model_px"][0])
    peak_y = float(center_peak["model_px"][1])
    peak_score = float(center_peak["score"])

    def sample(name: str, default: float = 0.0) -> float:
        index = channel_index.get(name)
        if index is None:
            return float(default)
        return _channel_value(maps[index], center_x, center_y)

    return {
        "center_x_norm": (float(center_x) / max(float(MODEL_WIDTH - 1), 1.0)) * 2.0 - 1.0,
        "center_y_norm": (float(center_y) / max(float(MODEL_HEIGHT - 1), 1.0)) * 2.0 - 1.0,
        "bbox_w_norm": bbox_w / float(MODEL_WIDTH),
        "bbox_h_norm": bbox_h / float(MODEL_HEIGHT),
        "component_area_frac": float(candidate["area_model_px"]) / float(MODEL_WIDTH * MODEL_HEIGHT),
        "component_confidence": float(candidate["confidence"]),
        "component_mean_probability": float(candidate["mean_probability"]),
        "center_peak_dx_norm": (float(center_x) - peak_x) / float(MODEL_WIDTH),
        "center_peak_dy_norm": (float(center_y) - peak_y) / float(MODEL_HEIGHT),
        "center_peak_score": peak_score,
        "pose_confidence_at_center": sample("pose_confidence_logit"),
        "base_position_x_norm_at_center": sample("position_camera_x_norm"),
        "base_position_y_norm_at_center": sample("position_camera_y_norm"),
        "base_position_z_norm_at_center": sample("position_camera_z_norm"),
    }


def _crop_maps_around_center(maps: torch.Tensor, center_x: float, center_y: float, crop_size: int) -> torch.Tensor:
    size = int(crop_size)
    half = size // 2
    center_x_i = int(round(float(center_x)))
    center_y_i = int(round(float(center_y)))
    padded = F.pad(maps, (half, half, half, half), mode="constant", value=0.0)
    crop = padded[:, center_y_i : center_y_i + size, center_x_i : center_x_i + size]
    if crop.shape[-2:] == (size, size):
        return crop
    fixed = torch.zeros((maps.shape[0], size, size), dtype=maps.dtype)
    fixed[:, : crop.shape[-2], : crop.shape[-1]] = crop
    return fixed


def _weighted_percentile(values: torch.Tensor, weights: torch.Tensor, percentile: float) -> float:
    if values.numel() <= 0:
        return 0.0
    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = torch.cumsum(sorted_weights, dim=0)
    cutoff = float(percentile) * float(cumulative[-1].item())
    index = int(torch.searchsorted(cumulative, torch.tensor(cutoff, dtype=cumulative.dtype)).clamp(0, len(sorted_values) - 1).item())
    return float(sorted_values[index].item())


def _mask_geometry_features_for_candidate(mask_probability: torch.Tensor, candidate: dict[str, Any]) -> dict[str, float]:
    bbox_x, bbox_y, bbox_w, bbox_h = [int(round(float(value))) for value in candidate["bbox_model_px"]]
    x0 = max(0, bbox_x)
    y0 = max(0, bbox_y)
    x1 = min(int(mask_probability.shape[-1]), bbox_x + max(1, bbox_w))
    y1 = min(int(mask_probability.shape[-2]), bbox_y + max(1, bbox_h))
    defaults = {name: 0.0 for name in DEFAULT_MASK_GEOMETRY_FEATURES}
    if x1 <= x0 or y1 <= y0:
        return defaults
    roi = mask_probability[y0:y1, x0:x1].to(dtype=torch.float32)
    weights = roi.reshape(-1).clamp_min(0.0)
    weight_sum = float(weights.sum().item())
    if weight_sum <= 1.0e-6:
        return defaults
    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, dtype=torch.float32),
        torch.arange(x0, x1, dtype=torch.float32),
        indexing="ij",
    )
    xs = xx.reshape(-1)
    ys = yy.reshape(-1)
    mean_x = float(((weights * xs).sum() / weights.sum()).item())
    mean_y = float(((weights * ys).sum() / weights.sum()).item())
    dx = xs - mean_x
    dy = ys - mean_y
    cov_xx = float(((weights * dx * dx).sum() / weights.sum()).item())
    cov_yy = float(((weights * dy * dy).sum() / weights.sum()).item())
    cov_xy = float(((weights * dx * dy).sum() / weights.sum()).item())
    trace = cov_xx + cov_yy
    delta = math.sqrt(max(0.0, ((cov_xx - cov_yy) * (cov_xx - cov_yy)) + (4.0 * cov_xy * cov_xy)))
    major_var = max(0.0, 0.5 * (trace + delta))
    minor_var = max(0.0, 0.5 * (trace - delta))
    major_sigma = math.sqrt(major_var)
    minor_sigma = math.sqrt(minor_var)
    if abs(cov_xy) > 1.0e-9 or abs(major_var - cov_xx) > 1.0e-9:
        vx = cov_xy
        vy = major_var - cov_xx
        norm = math.sqrt((vx * vx) + (vy * vy))
        if norm <= 1.0e-9:
            vx, vy = 1.0, 0.0
        else:
            vx, vy = vx / norm, vy / norm
    else:
        vx, vy = 1.0, 0.0
    axis_angle = math.atan2(vy, vx)
    major_proj = dx * float(vx) + dy * float(vy)
    minor_proj = -dx * float(vy) + dy * float(vx)
    major_p90 = _weighted_percentile(major_proj, weights, 0.95) - _weighted_percentile(major_proj, weights, 0.05)
    minor_p90 = _weighted_percentile(minor_proj, weights, 0.95) - _weighted_percentile(minor_proj, weights, 0.05)
    component_center = candidate.get("center_model_px") or candidate.get("component_center_model_px") or [mean_x, mean_y]
    axis_ratio = major_sigma / max(minor_sigma, 1.0e-6)
    anisotropy = 0.0 if trace <= 1.0e-6 else (major_var - minor_var) / trace
    return {
        "mask_weight_frac": weight_sum / float(MODEL_WIDTH * MODEL_HEIGHT),
        "mask_centroid_dx_norm": (mean_x - float(component_center[0])) / float(MODEL_WIDTH),
        "mask_centroid_dy_norm": (mean_y - float(component_center[1])) / float(MODEL_HEIGHT),
        "mask_major_sigma_norm": major_sigma / float(MODEL_WIDTH),
        "mask_minor_sigma_norm": minor_sigma / float(MODEL_HEIGHT),
        "mask_major_p90_norm": float(major_p90) / float(MODEL_WIDTH),
        "mask_minor_p90_norm": float(minor_p90) / float(MODEL_HEIGHT),
        "mask_axis_ratio_log": math.log(max(axis_ratio, 1.0e-6)),
        "mask_anisotropy": float(max(0.0, min(1.0, anisotropy))),
        "mask_axis_cos2": math.cos(2.0 * axis_angle),
        "mask_axis_sin2": math.sin(2.0 * axis_angle),
    }


class LogitRegressor:
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_REGRESSOR_CHECKPOINT,
        *,
        device: str | torch.device = "auto",
        gate_threshold: float = 0.50,
        confidence_threshold: float = 0.50,
        min_component_area: int = 3,
        max_candidates: int = 32,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Regressor checkpoint not found: {self.checkpoint_path}")
        self.device = select_device(str(device)) if not isinstance(device, torch.device) else device
        self.gate_threshold = float(gate_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.min_component_area = int(min_component_area)
        self.max_candidates = int(max_candidates)
        self.model, self.config = self._load_model(self.checkpoint_path)

    def _load_model(self, checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        if config.get("schema_version") != REGRESSOR_SCHEMA_VERSION:
            raise ValueError(
                "Regressor checkpoint schema is incompatible: "
                f"checkpoint={config.get('schema_version')} expected={REGRESSOR_SCHEMA_VERSION}."
            )
        model = build_model(
            len(config.get("feature_channels", ADAPTER_MAP_CHANNELS)),
            len(config.get("scalar_features", DEFAULT_SCALAR_FEATURES)),
            len(config.get("mask_geometry_features") or DEFAULT_MASK_GEOMETRY_FEATURES),
        ).to(self.device)
        load_compatible_state_dict(model, checkpoint["model_state_dict"])
        model.eval()
        return model, config

    def run_frame(self, frame: RawLogitsFrame) -> FrameRegression:
        source_width = SOURCE_IMAGE_WIDTH
        source_height = SOURCE_IMAGE_HEIGHT
        mask_probs = torch.sigmoid(frame.mask_logits.detach().cpu()).to(dtype=torch.float32)
        depth_probs = torch.sigmoid(frame.depth_logits.detach().cpu()).to(dtype=torch.float32)
        gate_map = mask_probs[0].clamp(0.0, 1.0)
        distractor_map = mask_probs[1].clamp(0.0, 1.0)
        depth_map = depth_probs[0].clamp(0.0, 1.0)

        candidates = _decode_gate_components(
            gate_map,
            threshold=self.gate_threshold,
            min_component_area=self.min_component_area,
            max_candidates=self.max_candidates,
            source_width=source_width,
            source_height=source_height,
        )
        candidates = [item for item in candidates if float(item["confidence"]) >= self.confidence_threshold]
        center_heatmap = _contour_quad_center_heatmap(candidates)
        maps = torch.stack([gate_map, center_heatmap, depth_map, distractor_map], dim=0).to(dtype=torch.float32)
        center_peak = _center_peak_from_map(center_heatmap)
        map_channels = list(ADAPTER_MAP_CHANNELS)

        if not candidates:
            return FrameRegression(
                frame_id=int(frame.frame_id),
                sim_time_ns=int(frame.sim_time_ns),
                source_width=source_width,
                source_height=source_height,
                gates=[],
            )

        feature_channels = list(self.config.get("feature_channels", ADAPTER_MAP_CHANNELS))
        scalar_feature_names = list(self.config.get("scalar_features", DEFAULT_SCALAR_FEATURES))
        mask_geometry_feature_names = list(self.config.get("mask_geometry_features") or DEFAULT_MASK_GEOMETRY_FEATURES)
        crop_size = int(self.config.get("crop_size", DEFAULT_CROP_SIZE))
        channel_index = {name: index for index, name in enumerate(map_channels)}
        feature_indices = [channel_index[name] for name in feature_channels]
        selected_maps = maps[feature_indices]

        crops: list[torch.Tensor] = []
        scalars: list[torch.Tensor] = []
        mask_geometries: list[torch.Tensor] = []
        for candidate in candidates:
            candidate["scalar_features"] = _scalar_features_for_candidate(
                maps,
                map_channels,
                candidate,
                center_peak=center_peak,
            )
            center_x, center_y = candidate["center_model_px"]
            crops.append(_crop_maps_around_center(selected_maps, center_x, center_y, crop_size))
            scalars.append(
                torch.tensor(
                    [float(candidate["scalar_features"].get(name, 0.0)) for name in scalar_feature_names],
                    dtype=torch.float32,
                )
            )
            mask_geometry = _mask_geometry_features_for_candidate(gate_map, candidate)
            mask_geometries.append(
                torch.tensor(
                    [float(mask_geometry.get(name, 0.0)) for name in mask_geometry_feature_names],
                    dtype=torch.float32,
                )
            )

        with torch.no_grad():
            outputs = self.model(
                torch.stack(crops, dim=0).to(self.device),
                torch.stack(scalars, dim=0).to(self.device),
                torch.stack(mask_geometries, dim=0).to(self.device),
            )
            z_m = _denormalize_z_tensor(outputs["z_norm"]).detach().cpu()
            orientation_axis = F.normalize(outputs["orientation_axis_raw"], p=2, dim=1, eps=1.0e-6).detach().cpu()

        gates: list[GateRegression] = []
        for index, candidate in enumerate(candidates):
            center_model = tuple(float(value) for value in candidate["quad_center_model_px"])
            center_px = tuple(float(value) for value in candidate["quad_center_px"])
            z_value = float(z_m[index, 0].item())
            position = _model_xy_z_to_camera_xyz(
                center_model,
                z_value,
                source_width=source_width,
                source_height=source_height,
            )
            orientation = tuple(float(value) for value in orientation_axis[index].tolist())
            distance = math.sqrt(sum(float(value) * float(value) for value in position))
            gates.append(
                GateRegression(
                    center_model_px=center_model,
                    center_px=center_px,
                    quad_center_model_px=center_model,
                    quad_model_px=[[float(value) for value in point] for point in candidate["quad_model_px"]],
                    position_xyz=position,
                    orientation_xyz=orientation,
                    confidence=float(candidate["confidence"]),
                    distance_camera_m=float(distance),
                    quad_fit_source=str(candidate["quad_fit_source"]),
                    quad_is_fallback=bool(candidate["quad_is_fallback"]),
                )
            )

        gates.sort(key=lambda item: item.distance_camera_m)
        return FrameRegression(
            frame_id=int(frame.frame_id),
            sim_time_ns=int(frame.sim_time_ns),
            source_width=source_width,
            source_height=source_height,
            gates=gates,
        )


__all__ = [
    "DEFAULT_REGRESSOR_CHECKPOINT",
    "FrameRegression",
    "GateRegression",
    "LogitRegressor",
    "select_device",
]
