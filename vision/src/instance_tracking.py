from __future__ import annotations
from math import hypot
import camera_odometry as codo
import final_2d_estimate
from schema import MotionCompensation, MotionForecast, MotionTrail, TrackMatch, TrailPoint, VoidAnalysis
from track_estimate import FRAME_DT_S, TrackEstimate

INSTANCE_EARN_STREAK = 10
INSTANCE_MAX_MISSING = 15
VOID_EARN_STREAK = 5
VOID_MAX_MISSING = 10
LIFETIME_AREA_AGREEMENT = 0.5
VOID_GRACE_MULTIPLIER = 1.6
TRAIL_WINDOW = 5
PREDICT_STEPS = 15
MAX_TRACKED_VOIDS = 5  # combined live + ghost void identities in flight at once


def _center(bbox):
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0

def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0); union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0

def _distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])

def _area_agreement(a, b):
    return min(a, b) / max(a, b) if a and b else 0.0


def _new_track(id_, center, area, bbox=None, frame_path=None, clipped=False):
    streak = 0 if clipped else 1
    return {"id": id_, "bbox": bbox, "center": center, "area": area, "area_ema": area,
            "streak": streak, "best_streak": streak, "missing": 0, "reliable": False,
            "last_frame_path": frame_path, "clipping": clipped,
            "motion_trail": None, "last_cv_confidence": None, "region_track_id": None}


def _advance_track(track, center, area, bbox, earn_streak, frame_path=None, clipped=False):
    track["center"], track["area"], track["bbox"] = center, area, bbox
    track["last_frame_path"] = frame_path
    track["missing"] = 0
    track["clipping"] = clipped
    if clipped:
        return  # a clipped observation still keeps the track alive, but proves nothing about its quality
    track["area_ema"] = track["area_ema"] * 0.8 + area * 0.2
    track["streak"] += 1
    track["best_streak"] = max(track["best_streak"], track["streak"])
    track["reliable"] = track["reliable"] or track["best_streak"] >= earn_streak


def _age_track(track):
    track["streak"] = 0
    track["missing"] += 1


def _build_match(pieces, streak):
    iou, raw_distance, area, lifetime_area, score, mc = pieces
    return TrackMatch(iou=iou, raw_distance_px=raw_distance, area_agreement=area,
                       lifetime_area_agreement=lifetime_area, score=score, streak=streak,
                       motion_compensation=mc)


class InstanceTracker:
    def __init__(self, min_iou=0.08, max_region_center_px=35.0,
                 max_void_center_px=25.0, min_area_agreement=0.45):
        self.tracks = []
        self.next_instance = 1
        self.next_void = 1
        self.min_iou = min_iou
        self.max_region_center_px = max_region_center_px
        self.max_void_center_px = max_void_center_px
        self.min_area_agreement = min_area_agreement
        self._telemetry_checked = False
        self._telemetry = None
        self._track_estimates: dict[str, TrackEstimate] = {}
        self._void_tracks: list[dict] = []
        self._void_ghost_pool: list[dict] = []

    def _new_id(self, prefix):
        attr = "next_instance" if prefix == "instance_track" else "next_void"
        value = getattr(self, attr)
        setattr(self, attr, value + 1)
        return f"{prefix}_{value:03d}"

    def _ensure_telemetry(self, frame_path):
        if self._telemetry_checked or frame_path is None:
            return
        self._telemetry_checked = True
        path = codo.find_telemetry_for_frame(frame_path)
        if path is not None:
            self._telemetry = codo.load_telemetry(path)

    def _build_trail(self, center, frame_path, upcoming_frame_paths):
        if self._telemetry is None or not upcoming_frame_paths:
            return None
        points = []
        for future_path in upcoming_frame_paths[:TRAIL_WINDOW]:
            predicted = codo.predict_center(self._telemetry, frame_path, center, future_path)
            if predicted is not None:
                points.append(TrailPoint(frame_path=future_path, predicted_center=predicted))
        return MotionTrail(points=points) if points else None

    def _build_forecast(self, track_id, center, t, mc_point):
        if t is None:
            return None
        est = self._track_estimates.setdefault(track_id, TrackEstimate())
        est.observe(t, center)
        if mc_point is not None:
            est.observe(t, mc_point)
        future = [est.project(t + step * FRAME_DT_S) for step in range(1, PREDICT_STEPS + 1)]
        if any(point is None for point in future):
            return None
        return MotionForecast(confidence=est.confidence(), history=list(est.history), forecast=future)

    def _check_ghost_pool(self, center, t):
        """A genuinely new void's first sighting, checked against recently-expired reliable
        voids' own continued projections. Never merges identity — just flags a possible match
        and removes that ghost so it can't be claimed twice."""
        tolerance = self.max_void_center_px * VOID_GRACE_MULTIPLIER
        best_index, best_id, best_error = None, None, None
        for i, ghost in enumerate(self._void_ghost_pool):
            estimate = final_2d_estimate.estimate_from_gap(
                ghost["track_id"], ghost["missing"], ghost["last_cv_confidence"], ghost["motion_trail"], ghost["estimate"], t)
            if estimate is None:
                continue
            error = _distance(center, estimate.center)
            if error <= tolerance and (best_error is None or error < best_error):
                best_index, best_id, best_error = i, ghost["track_id"], error
        if best_index is not None:
            self._void_ghost_pool.pop(best_index)
        return best_id, best_error

    def _make_room_for_new_void(self):
        """Called only when a genuinely new void needs a fresh mint and the combined live+ghost
        pool is already at MAX_TRACKED_VOIDS. Cheapest sacrifice first: an expired ghost, then the
        weakest not-yet-reliable live track — this is "discounting fragments" in code: proven
        tracks always outlive fresh, unproven ones under pressure. Never refuses a real detection
        an identity outright — a true 5-reliable-voids scene (never observed in this dataset) is
        allowed a temporary 6th slot rather than losing ground truth."""
        if len(self._void_tracks) + len(self._void_ghost_pool) < MAX_TRACKED_VOIDS:
            return
        if self._void_ghost_pool:
            oldest = max(self._void_ghost_pool, key=lambda g: g["missing"])
            self._void_ghost_pool.remove(oldest)
            return
        candidates = [v for v in self._void_tracks if not v["reliable"]]
        if candidates:
            weakest = min(candidates, key=lambda v: v["streak"])
            self._void_tracks.remove(weakest)

    def _effective_distance(self, prev_center, prev_frame, cur_center, cur_frame, base_limit, allow_mc=True):
        raw_distance = _distance(prev_center, cur_center)
        result = None
        if allow_mc and self._telemetry is not None and prev_frame and cur_frame:
            result = codo.compensated_match(self._telemetry, prev_frame, prev_center, cur_frame, cur_center)
        if result is None:
            return raw_distance, raw_distance, base_limit, None
        compensated_distance, predicted_center, scale = result
        tolerance = base_limit * scale
        mc = MotionCompensation(predicted_center=predicted_center, source_frame_path=prev_frame,
                                 compensated_distance_px=compensated_distance, translation_scale=scale,
                                 tolerance_px=tolerance)
        return compensated_distance, raw_distance, tolerance, mc

    def _match_region(self, region, used, frame_path):
        center = _center(region.bbox)
        best, best_score, best_pieces = None, -1.0, None
        for track in self.tracks:
            if track["id"] in used:
                continue
            overlap = _iou(region.bbox, track["bbox"])
            allow_mc = not track["clipping"] and not region.mask_clipping
            distance, raw_distance, limit, mc = self._effective_distance(
                track["center"], track["last_frame_path"], center, frame_path, self.max_region_center_px, allow_mc)
            area = _area_agreement(region.pixel_count, track["area"])
            if area < self.min_area_agreement:
                continue
            if overlap < self.min_iou and distance > limit:
                continue
            lifetime_area = None
            if track["reliable"] and not region.mask_clipping:
                lifetime_area = _area_agreement(region.pixel_count, track["area_ema"])
                if lifetime_area < LIFETIME_AREA_AGREEMENT:
                    continue
            score = overlap * 3.0 + (1.0 - min(distance / 80.0, 1.0)) + area
            if score > best_score:
                best, best_score = track, score
                best_pieces = (overlap, raw_distance, area, lifetime_area, score, mc)
        return best, best_pieces

    def _match_void(self, void, voids, used, frame_path):
        best, best_score, best_pieces = None, -1.0, None
        for track in voids:
            if track["id"] in used:
                continue
            base_limit = self.max_void_center_px * (VOID_GRACE_MULTIPLIER if track["reliable"] else 1.0)
            allow_mc = not track["clipping"] and not void.mask_clipping
            distance, raw_distance, limit, mc = self._effective_distance(
                track["center"], track["last_frame_path"], void.center, frame_path, base_limit, allow_mc)
            area = _area_agreement(void.pixel_count, track["area"])
            if distance > limit or area < self.min_area_agreement:
                continue
            score = (1.0 - distance / limit) + area
            if score > best_score:
                best, best_score = track, score
                best_pieces = (None, raw_distance, area, None, score, mc)
        return best, best_pieces

    @staticmethod
    def _keep(track, used, max_missing):
        if track["id"] in used:
            return True
        _age_track(track)
        return track["reliable"] and track["missing"] <= max_missing

    def update(self, analysis: VoidAnalysis, frame_path: str | None = None, upcoming_frame_paths=()) -> VoidAnalysis:
        self._ensure_telemetry(frame_path)
        t = codo.frame_epoch_seconds(frame_path) if frame_path else None
        used_regions = set()
        regions_by_size = sorted(analysis.regions, key=lambda item: item.pixel_count, reverse=True)

        # Pass 1: instance/region matching. Voids are untouched here.
        for region in regions_by_size:
            track, pieces = self._match_region(region, used_regions, frame_path)
            if track is None:
                track = _new_track(self._new_id("instance_track"), _center(region.bbox), region.pixel_count, region.bbox, frame_path, region.mask_clipping)
                self.tracks.append(track)
            else:
                _advance_track(track, _center(region.bbox), region.pixel_count, region.bbox, INSTANCE_EARN_STREAK, frame_path, region.mask_clipping)
            used_regions.add(track["id"])
            region.track_id = track["id"]
            region.reliable = track["reliable"] and not region.mask_clipping
            region.track_match = _build_match(pieces, track["streak"]) if pieces else None
            region.motion_trail = self._build_trail(_center(region.bbox), frame_path, upcoming_frame_paths)
            region_mc_point = pieces[5].predicted_center if pieces and pieces[5] else None
            region.forecast = self._build_forecast(track["id"], _center(region.bbox), t, region_mc_point)
            region_confidence = final_2d_estimate.smooth_confidence(
                track["last_cv_confidence"], final_2d_estimate.cv_confidence(region.track_match))
            track["last_cv_confidence"] = region_confidence
            track["motion_trail"] = region.motion_trail
            region_estimate = final_2d_estimate.estimate_from_match(track["id"], _center(region.bbox), region_confidence)
            analysis.final_estimates.append(region_estimate)

        # Pass 2: voids, matched as one flat pool independent of whichever region they
        # currently sit inside — this is what survives an instance-level merge/split.
        used_void_ids = set()
        for region in regions_by_size:
            for void in region.voids:
                vtrack, vpieces = self._match_void(void, self._void_tracks, used_void_ids, frame_path)
                reconnect_id = reconnect_error = None
                if vtrack is None:
                    if t is not None:
                        reconnect_id, reconnect_error = self._check_ghost_pool(void.center, t)
                    self._make_room_for_new_void()
                    vtrack = _new_track(self._new_id("void_track"), void.center, void.pixel_count, frame_path=frame_path, clipped=void.mask_clipping)
                    self._void_tracks.append(vtrack)
                else:
                    _advance_track(vtrack, void.center, void.pixel_count, None, VOID_EARN_STREAK, frame_path, void.mask_clipping)
                vtrack["region_track_id"] = region.track_id
                used_void_ids.add(vtrack["id"])
                void.track_id = vtrack["id"]
                void.reliable = vtrack["reliable"] and not void.mask_clipping
                void.region_track_id = region.track_id
                void.track_match = _build_match(vpieces, vtrack["streak"]) if vpieces else None
                void.motion_trail = self._build_trail(void.center, frame_path, upcoming_frame_paths)
                void.possible_reconnection = reconnect_id
                void.reconnection_error_px = reconnect_error
                mc_point = vpieces[5].predicted_center if vpieces and vpieces[5] else None
                void.forecast = self._build_forecast(vtrack["id"], void.center, t, mc_point)
                void_confidence = final_2d_estimate.smooth_confidence(
                    vtrack["last_cv_confidence"], final_2d_estimate.cv_confidence(void.track_match))
                vtrack["last_cv_confidence"] = void_confidence
                vtrack["motion_trail"] = void.motion_trail
                void_estimate = final_2d_estimate.estimate_from_match(vtrack["id"], void.center, void_confidence, void.reliable)
                if void_estimate is not None:
                    analysis.final_estimates.append(void_estimate)

        # Void aging/pruning/ghost-pool population runs unconditionally, every frame, over
        # every live void — no void can freeze just because its parent instance went unmatched.
        kept_voids, pruned_voids = [], []
        for v in self._void_tracks:
            (kept_voids if self._keep(v, used_void_ids, VOID_MAX_MISSING) else pruned_voids).append(v)
        self._void_tracks = kept_voids
        for v in pruned_voids:
            if v["reliable"]:
                self._void_ghost_pool.append({
                    "track_id": v["id"], "last_cv_confidence": v["last_cv_confidence"],
                    "motion_trail": v["motion_trail"], "estimate": self._track_estimates.get(v["id"]),
                    "missing": 1,
                })

        self.tracks = [tr for tr in self.tracks if self._keep(tr, used_regions, INSTANCE_MAX_MISSING)]

        for ghost in self._void_ghost_pool:
            ghost["missing"] += 1
        self._void_ghost_pool = [g for g in self._void_ghost_pool if g["missing"] <= final_2d_estimate.MAX_FINAL_GAP_FRAMES]

        if t is not None:
            for track in self.tracks:
                if track["id"] not in used_regions:
                    estimate = final_2d_estimate.estimate_from_gap(
                        track["id"], track["missing"], track["last_cv_confidence"], track["motion_trail"],
                        self._track_estimates.get(track["id"]), t)
                    if estimate is not None:
                        analysis.final_estimates.append(estimate)
            for vtrack in self._void_tracks:
                if vtrack["id"] not in used_void_ids:
                    estimate = final_2d_estimate.estimate_from_gap(
                        vtrack["id"], vtrack["missing"], vtrack["last_cv_confidence"], vtrack["motion_trail"],
                        self._track_estimates.get(vtrack["id"]), t)
                    if estimate is not None:
                        analysis.final_estimates.append(estimate)

        return analysis
