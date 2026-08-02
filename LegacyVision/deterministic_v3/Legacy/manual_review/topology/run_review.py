"""Replay a historic run into isolated topology review evidence.

The production preprocessing and topology classifier run first and are never
modified by this helper. The local topology snapshot then evaluates the same
``FrameObservation`` so future experiments can be compared without entering
the production import path.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable

import cv2
import numpy as np

from ...preprocessing import (
    DEFAULT_CONFIG,
    component_mask,
    decode_jpeg,
    load_lut,
    preprocess_frame,
)
from ...schema import ComponentObservation, FrameObservation, TopologyDecision
from ...topology import assess_frame as assess_production_frame
from .topology import (
    DEFAULT_POLICY as CANDIDATE_POLICY,
    assess_frame as assess_candidate_frame,
)


REVIEW_FORMAT_VERSION = 1
WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "review_assets" / "generated"
PRODUCTION_TOPOLOGY_PATH = WORKSPACE_ROOT.parents[1] / "topology.py"
CANDIDATE_TOPOLOGY_PATH = WORKSPACE_ROOT / "topology.py"
PANEL_SIZE = 248
PANEL_TITLE_HEIGHT = 42
IMAGE_GAP = 8
REVIEW_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """One validated historic frame-ingress record."""

    run_id: str
    frame_id: int
    sim_time_ns: int
    jpeg_size: int
    relative_path: str
    source_path: Path

    def jpeg_bytes(self) -> bytes:
        payload = self.source_path.read_bytes()
        if len(payload) != self.jpeg_size:
            raise ValueError(
                f"JPEG size changed for {self.relative_path}: "
                f"{len(payload)} != {self.jpeg_size}")
        return payload


def historic_frames(run_path: str | Path) -> tuple[SourceFrame, ...]:
    """Validate and return ordered ``frames.jsonl`` ingress records."""
    run_dir = Path(run_path).resolve()
    manifest_path = run_dir / "frames.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"historic frame manifest not found: {manifest_path}")

    records: list[SourceFrame] = []
    seen_ids: set[int] = set()
    previous_time = -1
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                frame_id = int(item["frame_id"])
                sim_time_ns = int(item["sim_time_ns"])
                jpeg_size = int(item["jpeg_size"])
                relative_path = str(item["path"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid frame record at line {line_number}") from error
            if frame_id in seen_ids:
                raise ValueError(f"duplicate frame_id in manifest: {frame_id}")
            if sim_time_ns < previous_time:
                raise ValueError("frame manifest timing is not monotonic")
            relative = Path(relative_path)
            if (relative.is_absolute() or not relative.parts
                    or relative.parts[0] != "vision_frames"):
                raise ValueError(
                    f"frame path must be relative to vision_frames: "
                    f"{relative_path}")
            source_path = (run_dir / relative).resolve()
            if not source_path.is_relative_to(run_dir):
                raise ValueError(f"frame path escapes run: {relative_path}")
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            seen_ids.add(frame_id)
            previous_time = sim_time_ns
            records.append(SourceFrame(
                run_id=run_dir.name,
                frame_id=frame_id,
                sim_time_ns=sim_time_ns,
                jpeg_size=jpeg_size,
                relative_path=relative.as_posix(),
                source_path=source_path,
            ))
    return tuple(records)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert topology records while retaining schema field terminology."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported review value: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _square_source(
    image: np.ndarray, component: ComponentObservation
) -> np.ndarray:
    height, width = component.analysis_shape
    origin_x, origin_y = component.image_origin_uv
    canvas = np.zeros((height, width, 3), np.uint8)
    source_height, source_width = image.shape[:2]
    source_x0, source_y0 = max(origin_x, 0), max(origin_y, 0)
    source_x1 = min(origin_x + width, source_width)
    source_y1 = min(origin_y + height, source_height)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return canvas
    destination_x0 = source_x0 - origin_x
    destination_y0 = source_y0 - origin_y
    canvas[
        destination_y0:destination_y0 + source_y1 - source_y0,
        destination_x0:destination_x0 + source_x1 - source_x0,
    ] = image[source_y0:source_y1, source_x0:source_x1]
    return canvas


def _normalize_distance(
    distance: np.ndarray, *, scale_peak: float | None = None
) -> np.ndarray:
    peak = (float(distance.max()) if scale_peak is None else float(scale_peak))
    if peak <= 0:
        return np.zeros(distance.shape, np.uint8)
    return np.rint(np.clip(distance / peak, 0.0, 1.0) * 255.0).astype(
        np.uint8)


def _distance_heatmap(
    distance: np.ndarray, *, scale_peak: float | None = None
) -> np.ndarray:
    normalized = _normalize_distance(distance, scale_peak=scale_peak)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    heatmap[normalized == 0] = 0
    return heatmap


def _hierarchy_overlay(
    image: np.ndarray,
    frame: FrameObservation,
    component: ComponentObservation,
) -> np.ndarray:
    overlay = _square_source(image, component)
    origin = np.array(component.image_origin_uv, np.int32)
    for node in frame.closed_contour_nodes:
        if node.component_id != component.component_id:
            continue
        local = node.points_uv.copy()
        local[:, 0, :] -= origin
        color = (255, 0, 255) if node.is_hole else (0, 255, 0)
        cv2.polylines(overlay, [local], True, color, 2, cv2.LINE_AA)
        anchor_x, anchor_y = (int(value) for value in local[0, 0])
        cv2.putText(
            overlay,
            f"{node.contour_id}:d{node.depth}",
            (max(anchor_x, 2), max(anchor_y, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _place_maximum(
    canvas: np.ndarray,
    patch: np.ndarray,
    destination_x: int,
    destination_y: int,
) -> None:
    canvas_height, canvas_width = canvas.shape[:2]
    patch_height, patch_width = patch.shape[:2]
    x0, y0 = max(destination_x, 0), max(destination_y, 0)
    x1 = min(destination_x + patch_width, canvas_width)
    y1 = min(destination_y + patch_height, canvas_height)
    if x1 <= x0 or y1 <= y0:
        return
    patch_x0, patch_y0 = x0 - destination_x, y0 - destination_y
    existing = canvas[y0:y1, x0:x1]
    incoming = patch[
        patch_y0:patch_y0 + y1 - y0,
        patch_x0:patch_x0 + x1 - x0,
    ]
    np.maximum(existing, incoming, out=existing)


def _recreate_aperture_distance(
    frame: FrameObservation,
    component: ComponentObservation,
    decision: TopologyDecision,
) -> tuple[np.ndarray, np.ndarray]:
    """Recreate only transforms proven to have run in ``_aperture_center``."""
    distance_canvas = np.zeros(component.analysis_shape, np.float32)
    core_canvas = np.zeros(component.analysis_shape, np.uint8)
    origin_x, origin_y = component.image_origin_uv
    node_by_id = {node.contour_id: node for node in frame.closed_contour_nodes}

    for evidence in decision.aperture_center_evidence:
        node = node_by_id[evidence.child_contour_id]
        contour = node.points_uv
        x, y, width, height = cv2.boundingRect(contour)
        local = contour.copy()
        local[:, :, 0] -= x - 1
        local[:, :, 1] -= y - 1
        child_mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.drawContours(child_mask, [local], -1, 1, cv2.FILLED)
        distance = cv2.distanceTransform(child_mask, cv2.DIST_L2, 5)
        peak = float(distance.max())
        if not np.isclose(peak, evidence.distance_peak_px, atol=1e-5):
            raise AssertionError(
                "review aperture distance drifted from TopologyDecision: "
                f"{peak} != {evidence.distance_peak_px}")

        core_y, core_x = np.nonzero(distance >= 0.90 * peak)
        weights = distance[core_y, core_x]
        mass = float(weights.sum())
        if mass > 0:
            peak_x = float(np.dot(core_x, weights) / mass)
            peak_y = float(np.dot(core_y, weights) / mass)
        else:
            peak_x, peak_y = 1.0, 1.0
        recreated_center = (x + peak_x - 1.0, y + peak_y - 1.0)
        if not np.allclose(recreated_center, evidence.center_uv, atol=1e-5):
            raise AssertionError(
                "review aperture center drifted from TopologyDecision: "
                f"{recreated_center} != {evidence.center_uv}")

        destination_x = x - 1 - origin_x
        destination_y = y - 1 - origin_y
        _place_maximum(
            distance_canvas, distance, destination_x, destination_y)
        core = (distance >= 0.90 * peak).astype(np.uint8)
        _place_maximum(core_canvas, core, destination_x, destination_y)
    return distance_canvas, core_canvas


def _recreate_parent_distance(
    frame: FrameObservation,
    component: ComponentObservation,
    decision: TopologyDecision,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Recreate the candidate's pair-local full-parent distance evidence."""
    distance_canvas = np.zeros(component.analysis_shape, np.float32)
    core_canvas = np.zeros(component.analysis_shape, np.uint8)
    origin_x, origin_y = component.image_origin_uv
    node_by_id = {node.contour_id: node for node in frame.closed_contour_nodes}
    evidence_records: list[dict[str, Any]] = []
    threshold = CANDIDATE_POLICY.standard_minimum_parent_distance_depth_ratio

    for aperture in decision.aperture_center_evidence:
        parent = node_by_id[aperture.parent_contour_id]
        contour = parent.points_uv
        x, y, width, height = cv2.boundingRect(contour)
        local = contour.copy()
        local[:, :, 0] -= x - 1
        local[:, :, 1] -= y - 1
        parent_mask = np.zeros((height + 2, width + 2), np.uint8)
        cv2.drawContours(parent_mask, [local], -1, 1, cv2.FILLED)
        distance = cv2.distanceTransform(parent_mask, cv2.DIST_L2, 5)
        _, peak, _, maximum_xy = cv2.minMaxLoc(distance)
        local_center = (
            float(aperture.center_uv[0] - x + 1.0),
            float(aperture.center_uv[1] - y + 1.0),
        )
        center_depth = float(cv2.getRectSubPix(
            distance, (1, 1), local_center)[0, 0])
        depth_ratio = center_depth / peak if peak > 0 else 0.0
        accepted = depth_ratio >= threshold
        failed_token = "parent_distance_depth_ratio_below_minimum"
        if (failed_token in aperture.failed_checks) == accepted:
            raise AssertionError(
                "review parent-distance result drifted from candidate topology")

        destination_x = x - 1 - origin_x
        destination_y = y - 1 - origin_y
        _place_maximum(
            distance_canvas, distance, destination_x, destination_y)
        core = (distance >= threshold * peak).astype(np.uint8)
        _place_maximum(core_canvas, core, destination_x, destination_y)
        maximum_uv = (
            float(x + maximum_xy[0] - 1),
            float(y + maximum_xy[1] - 1),
        )
        evidence_records.append({
            "parent_contour_id": aperture.parent_contour_id,
            "child_contour_id": aperture.child_contour_id,
            "parent_distance_peak_px": float(peak),
            "parent_distance_at_child_center_px": center_depth,
            "parent_distance_depth_ratio": depth_ratio,
            "minimum_parent_distance_depth_ratio": threshold,
            "parent_distance_maximum_uv": maximum_uv,
            "child_aperture_center_uv": aperture.center_uv,
            "accepted": accepted,
        })
    return distance_canvas, core_canvas, evidence_records


def _parent_distance_heatmap(
    frame: FrameObservation,
    component: ComponentObservation,
    decision: TopologyDecision,
) -> tuple[np.ndarray, list[dict[str, Any]], float | None]:
    """Show the complete filled-parent field used by the new standard gate."""
    height, width = component.analysis_shape
    if not decision.aperture_center_evidence:
        panel = np.zeros((height, width, 3), np.uint8)
        cv2.putText(
            panel,
            "N/A: branch did not run test",
            (8, max(height // 2, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        return panel, [], None

    distance, core, records = _recreate_parent_distance(
        frame, component, decision)
    scale_peak = max(
        item["parent_distance_peak_px"] for item in records)
    heatmap = _distance_heatmap(distance, scale_peak=scale_peak)
    core_contours, _ = cv2.findContours(
        core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heatmap, core_contours, -1, (255, 255, 255), 1)
    origin_x, origin_y = component.image_origin_uv
    for item in records:
        child_point = (
            int(round(item["child_aperture_center_uv"][0] - origin_x)),
            int(round(item["child_aperture_center_uv"][1] - origin_y)),
        )
        maximum_point = (
            int(round(item["parent_distance_maximum_uv"][0] - origin_x)),
            int(round(item["parent_distance_maximum_uv"][1] - origin_y)),
        )
        cv2.line(
            heatmap, maximum_point, child_point, (0, 255, 255), 1,
            cv2.LINE_AA)
        cv2.circle(heatmap, maximum_point, 4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(
            heatmap, child_point, (0, 0, 0), cv2.MARKER_CROSS, 13, 3,
            cv2.LINE_AA)
        cv2.drawMarker(
            heatmap, child_point, (255, 255, 255), cv2.MARKER_CROSS, 11, 1,
            cv2.LINE_AA)
    return heatmap, records, scale_peak


def _aperture_heatmap(
    frame: FrameObservation,
    component: ComponentObservation,
    decision: TopologyDecision,
    *,
    scale_peak: float | None = None,
) -> np.ndarray:
    height, width = component.analysis_shape
    if not decision.aperture_center_evidence:
        panel = np.zeros((height, width, 3), np.uint8)
        cv2.putText(
            panel,
            "N/A: branch did not run test",
            (8, max(height // 2, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        return panel

    distance, core = _recreate_aperture_distance(
        frame, component, decision)
    heatmap = _distance_heatmap(distance, scale_peak=scale_peak)
    heatmap[core != 0] = (255, 255, 255)
    origin_x, origin_y = component.image_origin_uv
    for evidence in decision.aperture_center_evidence:
        point = (
            int(round(evidence.center_uv[0] - origin_x)),
            int(round(evidence.center_uv[1] - origin_y)),
        )
        cv2.drawMarker(
            heatmap, point, (0, 0, 0), cv2.MARKER_CROSS, 13, 3,
            cv2.LINE_AA)
        cv2.drawMarker(
            heatmap, point, (255, 255, 255), cv2.MARKER_CROSS, 11, 1,
            cv2.LINE_AA)
    return heatmap


def _panel(image: np.ndarray, title: str, *, nearest: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    resized = cv2.resize(
        image, (PANEL_SIZE, PANEL_SIZE), interpolation=interpolation)
    canvas = np.full(
        (PANEL_SIZE + PANEL_TITLE_HEIGHT, PANEL_SIZE, 3), 24, np.uint8)
    canvas[PANEL_TITLE_HEIGHT:] = resized
    for line_index, line in enumerate(title.split("\n")):
        cv2.putText(
            canvas,
            line,
            (6, 15 + line_index * 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _header_lines(
    source: SourceFrame,
    component: ComponentObservation,
    production: TopologyDecision,
    candidate: TopologyDecision,
    parent_distance_evidence: list[dict[str, Any]],
) -> list[str]:
    lines = [
        f"{source.run_id} | frame={source.frame_id} | "
        f"sim_time_ns={source.sim_time_ns} | component={component.component_id}",
        f"candidate={candidate.topology_label} route={candidate.route} "
        f"accepted={candidate.accepted} rule={candidate.classification_rule}",
        f"production={production.topology_label} "
        f"changed={candidate != production} bbox={component.bbox_xywh}",
    ]
    if candidate.rejection_reason:
        lines.append(f"rejection_reason={candidate.rejection_reason}")
    for item in parent_distance_evidence:
        lines.append(
            f"parent={item['parent_contour_id']} child="
            f"{item['child_contour_id']} parent_peak="
            f"{item['parent_distance_peak_px']:.2f} at_child="
            f"{item['parent_distance_at_child_center_px']:.2f} ratio="
            f"{item['parent_distance_depth_ratio']:.3f} min="
            f"{item['minimum_parent_distance_depth_ratio']:.3f} "
            f"pass={item['accepted']}")
    for evidence in candidate.aperture_center_evidence:
        lines.append(
            f"child={evidence.child_contour_id} peak={evidence.distance_peak_px:.2f} "
            f"diameter={evidence.diameter_ratio:.3f} "
            f"offset={evidence.center_offset_ratio:.3f} "
            f"p25_balance={evidence.radial_balance_ratio:.3f} "
            f"median_balance={evidence.radial_balance_median:.3f} "
            f"pass={evidence.accepted} failures={list(evidence.failed_checks)}")
    return lines


def render_component_review(
    *,
    source: SourceFrame,
    image: np.ndarray,
    frame: FrameObservation,
    component: ComponentObservation,
    production: TopologyDecision,
    candidate: TopologyDecision,
    output_path: Path,
) -> dict[str, Any]:
    """Write one complete visual audit card and return transform metadata."""
    source_panel = _square_source(image, component)
    closed = component_mask(frame, component, stage="closed")
    closed_panel = cv2.cvtColor(closed * 255, cv2.COLOR_GRAY2BGR)
    hierarchy_panel = _hierarchy_overlay(image, frame, component)
    foreground_distance = component.distance_transform
    parent_panel, parent_evidence, shared_peak = _parent_distance_heatmap(
        frame, component, candidate)
    aperture_panel = _aperture_heatmap(
        frame, component, candidate, scale_peak=shared_peak)

    panels = (
        _panel(source_panel, "historic frame\nsource BGR crop"),
        _panel(
            closed_panel,
            "preprocessing.py\nfinal closed component",
            nearest=True,
        ),
        _panel(
            hierarchy_panel,
            "preprocessing.py\nFrameObservation.closed_contour_nodes",
        ),
        _panel(
            parent_panel,
            "topology.py:_parent_distance_depth_ratio\nfilled parent distance",
        ),
        _panel(
            aperture_panel,
            "topology.py:_aperture_center\nchild distance transform",
        ),
    )
    content_width = len(panels) * PANEL_SIZE + (len(panels) - 1) * IMAGE_GAP
    lines = _header_lines(
        source, component, production, candidate, parent_evidence)
    header_height = 18 + 20 * len(lines)
    canvas = np.full(
        (header_height + panels[0].shape[0], content_width, 3),
        16,
        np.uint8,
    )
    for line_index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (8, 19 + line_index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    x = 0
    for panel in panels:
        canvas[header_height:, x:x + PANEL_SIZE] = panel
        x += PANEL_SIZE + IMAGE_GAP

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
            str(output_path), canvas, (cv2.IMWRITE_PNG_COMPRESSION, 6)):
        raise OSError(f"unable to write review image: {output_path}")
    return {
        "foreground_distance_peak_px": float(foreground_distance.max()),
        "parent_transform_count": len(parent_evidence),
        "parent_distance_evidence": parent_evidence,
        "aperture_transform_count": len(candidate.aperture_center_evidence),
        "aperture_distance_peaks_px": [
            item.distance_peak_px
            for item in candidate.aperture_center_evidence
        ],
    }


def _write_html_pages(output_dir: Path, cards: Iterable[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_cards = list(cards)
    for card in all_cards:
        grouped[card["candidate_topology_label"]].append(card)

    style = """
body{margin:20px;background:#111;color:#ddd;font:13px system-ui,sans-serif}
a{color:#8ecbff}.summary{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}
.card{margin:0 0 22px;padding:10px;background:#1b1b1b;border:1px solid #383838}
.card.changed{border-color:#ffb347}.meta{margin-bottom:8px;font:12px monospace}
img{display:block;max-width:100%;height:auto;background:#000}
"""

    def page(title: str, page_cards: list[dict[str, Any]]) -> str:
        blocks = []
        for card in page_cards:
            changed_class = " changed" if card["decision_changed"] else ""
            identity = html.escape(card["identity"])
            label = html.escape(card["candidate_topology_label"])
            rule = html.escape(card["classification_rule"])
            path = html.escape(card["visualization_path"])
            blocks.append(
                f'<article class="card{changed_class}">'
                f'<div class="meta">{identity} | {label} | {rule}</div>'
                f'<a href="{path}"><img loading="lazy" src="{path}" '
                f'alt="{identity} topology review"></a></article>')
        navigation = ' · '.join(
            f'<a href="{html.escape(label)}.html">{html.escape(label)} '
            f'({len(items)})</a>'
            for label, items in sorted(grouped.items()))
        return (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(title)}</title><style>{style}</style></head>"
            f"<body><h1>{html.escape(title)}</h1>"
            f"<p><a href=\"index.html\">summary</a> · "
            f"<a href=\"all.html\">all ({len(all_cards)})</a> · {navigation}</p>"
            f"{''.join(blocks)}</body></html>"
        )

    summary = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        f"<title>Topology manual review</title><style>{style}</style></head>",
        "<body><h1>Topology manual review</h1>",
        "<p>Predictions are subjective review candidates, not ground truth. ",
        '<a href="../../../LABELING_INSTRUCTIONS.md">Labeling instructions</a></p>',
        '<div class="summary"><a href="all.html">all components '
        f"({len(all_cards)})</a>",
    ]
    for label, items in sorted(grouped.items()):
        summary.append(
            f'<a href="{html.escape(label)}.html">{html.escape(label)} '
            f'({len(items)})</a>')
    summary.append("</div></body></html>")
    (output_dir / "index.html").write_text(
        "".join(summary), encoding="utf-8")
    (output_dir / "all.html").write_text(
        page("All topology components", all_cards), encoding="utf-8")
    for label, items in grouped.items():
        (output_dir / f"{label}.html").write_text(
            page(f"Topology: {label}", items), encoding="utf-8")


def generate_review(
    run_path: str | Path,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    review_id: str | None = None,
    mode: str = "baseline",
    limit: int | None = None,
    lut: np.ndarray | None = None,
) -> Path:
    """Generate complete topology decisions and visual evidence for one run."""
    if mode not in {"baseline", "candidate"}:
        raise ValueError("mode must be 'baseline' or 'candidate'")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    review_id = review_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ")
    if REVIEW_ID_PATTERN.fullmatch(review_id) is None:
        raise ValueError("review_id may contain only letters, numbers, . _ and -")

    records = historic_frames(run_path)
    selected = records if limit is None else records[:limit]
    run_id = Path(run_path).resolve().name
    output_dir = Path(output_root).resolve() / run_id / review_id
    if output_dir.exists():
        raise FileExistsError(f"review output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    production_sha = _sha256(PRODUCTION_TOPOLOGY_PATH)
    candidate_sha = _sha256(CANDIDATE_TOPOLOGY_PATH)
    if mode == "baseline" and production_sha != candidate_sha:
        raise RuntimeError(
            "baseline mode requires a byte-for-byte production topology copy; "
            "use --mode candidate after editing the isolated copy")

    active_lut = load_lut() if lut is None else lut
    production_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    frame_entries: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    changed_decisions = 0
    timing_ns: Counter[str] = Counter()
    component_count = 0
    records_path = output_dir / "topology-records.jsonl"

    with records_path.open("w", encoding="utf-8") as record_stream:
        for index, source in enumerate(selected, 1):
            jpeg = source.jpeg_bytes()
            source_sha = hashlib.sha256(jpeg).hexdigest()
            started = perf_counter_ns()
            image = decode_jpeg(jpeg)
            frame = preprocess_frame(
                frame_id=source.frame_id,
                sim_time_ns=source.sim_time_ns,
                image=image,
                lut=active_lut,
                config=DEFAULT_CONFIG,
            )
            timing_ns["preprocessing"] += perf_counter_ns() - started

            started = perf_counter_ns()
            production = assess_production_frame(frame)
            timing_ns["production_topology"] += perf_counter_ns() - started
            started = perf_counter_ns()
            candidate = assess_candidate_frame(frame)
            timing_ns["candidate_topology"] += perf_counter_ns() - started
            if len(production) != len(frame.components):
                raise AssertionError("production topology lost component coverage")
            if len(candidate) != len(frame.components):
                raise AssertionError("candidate topology lost component coverage")
            if mode == "baseline" and candidate != production:
                raise AssertionError(
                    f"baseline decisions differ on frame {source.frame_id}")

            frame_label_counts: Counter[str] = Counter()
            for component, production_item, candidate_item in zip(
                    frame.components, production, candidate):
                component_count += 1
                decision_changed = candidate_item != production_item
                changed_decisions += int(decision_changed)
                production_counts[production_item.topology_label] += 1
                candidate_counts[candidate_item.topology_label] += 1
                frame_label_counts[candidate_item.topology_label] += 1
                rule_counts[candidate_item.classification_rule] += 1
                filename = (
                    f"frame-{source.frame_id:08d}-{source.sim_time_ns}-"
                    f"component-{component.component_id:04d}.png")
                relative_visual = (
                    Path("components")
                    / candidate_item.topology_label
                    / filename)
                started = perf_counter_ns()
                distance_metadata = render_component_review(
                    source=source,
                    image=image,
                    frame=frame,
                    component=component,
                    production=production_item,
                    candidate=candidate_item,
                    output_path=output_dir / relative_visual,
                )
                timing_ns["render_and_write"] += perf_counter_ns() - started
                identity = (
                    f"{source.run_id}/{source.frame_id}/"
                    f"{source.sim_time_ns}/{component.component_id}")
                record = {
                    "identity": {
                        "run_id": source.run_id,
                        "frame_id": source.frame_id,
                        "sim_time_ns": source.sim_time_ns,
                        "component_id": component.component_id,
                        "source_relative_path": source.relative_path,
                        "source_jpeg_sha256": source_sha,
                    },
                    "component_evidence": {
                        "bbox_xywh": _json_value(component.bbox_xywh),
                        "image_origin_uv": _json_value(
                            component.image_origin_uv),
                        "analysis_shape": _json_value(
                            component.analysis_shape),
                        "touches_frame": component.touches_frame,
                        "area_px": component.area_px,
                        "fill_ratio": component.fill_ratio,
                        "solidity": component.solidity,
                        "parent_contour_ids": _json_value(
                            component.contours.parent_contour_ids),
                        "parent_child_contour_ids": _json_value(
                            component.contours.parent_child_contour_ids),
                    },
                    "production_TopologyDecision": _json_value(
                        production_item),
                    "candidate_TopologyDecision": _json_value(candidate_item),
                    "decision_changed": decision_changed,
                    "distance_transform_review": distance_metadata,
                    "visualization_path": relative_visual.as_posix(),
                }
                record_stream.write(json.dumps(record) + "\n")
                cards.append({
                    "identity": identity,
                    "candidate_topology_label":
                        candidate_item.topology_label,
                    "classification_rule":
                        candidate_item.classification_rule,
                    "decision_changed": decision_changed,
                    "visualization_path": relative_visual.as_posix(),
                })

            frame_entries.append({
                "frame_id": source.frame_id,
                "sim_time_ns": source.sim_time_ns,
                "source_relative_path": source.relative_path,
                "source_jpeg_sha256": source_sha,
                "component_count": len(frame.components),
                "gated": frame.gated,
                "rejection_reason": frame.rejection_reason,
                "candidate_topology_label_counts": dict(frame_label_counts),
            })
            if index == 1 or index % 25 == 0 or index == len(selected):
                print(f"topology review frames: {index}/{len(selected)}")

    _write_html_pages(output_dir, cards)
    processed_frames = len(selected)
    manifest = {
        "review_format_version": REVIEW_FORMAT_VERSION,
        "review_id": review_id,
        "mode": mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_path": str(Path(run_path).resolve()),
        "source_frame_count": len(records),
        "processed_frame_count": processed_frames,
        "complete": processed_frames == len(records),
        "preprocessing_version": DEFAULT_CONFIG.version,
        "topology_input": "FrameObservation.closed_mask and contour evidence",
        "density_evidence_used": False,
        "production_topology_sha256": production_sha,
        "candidate_topology_sha256": candidate_sha,
        "topology_source_files_byte_identical": production_sha == candidate_sha,
        "decision_equivalent_to_production": changed_decisions == 0,
        "changed_decision_count": changed_decisions,
        "component_count": component_count,
        "production_topology_label_counts": dict(production_counts),
        "candidate_topology_label_counts": dict(candidate_counts),
        "candidate_classification_rule_counts": dict(rule_counts),
        "visualization_count": len(cards),
        "records_path": records_path.relative_to(output_dir).as_posix(),
        "index_path": "index.html",
        "timing": {
            "preprocessing_mean_ms_per_frame": (
                timing_ns["preprocessing"] / max(processed_frames, 1) / 1e6),
            "production_topology_mean_ms_per_frame": (
                timing_ns["production_topology"]
                / max(processed_frames, 1) / 1e6),
            "candidate_topology_mean_ms_per_frame": (
                timing_ns["candidate_topology"]
                / max(processed_frames, 1) / 1e6),
            "render_and_write_mean_ms_per_component": (
                timing_ns["render_and_write"]
                / max(component_count, 1) / 1e6),
        },
        "frames": frame_entries,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path)
    parser.add_argument(
        "--mode", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--review-id")
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = generate_review(
        args.run_path,
        mode=args.mode,
        limit=args.limit,
        review_id=args.review_id,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
