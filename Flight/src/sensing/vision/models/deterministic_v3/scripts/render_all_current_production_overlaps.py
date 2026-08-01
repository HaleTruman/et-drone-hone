"""Export every current-run multi-hole overlap through production processing."""

import json
from pathlib import Path

import cv2
import numpy as np

from sensing.vision.models.deterministic_v3.scripts.render_current_production_review import (
    RUN,
    render,
    scan,
)
from sensing.vision.models.deterministic_v3.src.inverse_density_quadrilateral_production import (
    INVERSE_GAMMA,
    P90_PERCENTILE,
    RADIUS_PROFILES,
    RELATIVE_CAP,
    RIDGE_GAMMA,
    fit_p90_quadrilateral,
    inverse_density_field,
    load_lut,
)


OUTPUT = Path(
    "Flight/src/sensing/vision/models/deterministic_v3/production_samples/"
    "current-production-all-overlaps")
ROWS_PER_PAGE = 8


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def fit_status(candidate):
    component = candidate.component
    profile = component.profile
    field = inverse_density_field(
        component.mask, profile.density_radius, profile.ridge_radius)
    return fit_p90_quadrilateral(component.mask, field) is not None


def main():
    instances = OUTPUT / "instances"
    contacts = OUTPUT / "contact-sheets"
    instances.mkdir(parents=True, exist_ok=True)
    contacts.mkdir(parents=True, exist_ok=True)

    lut = load_lut()
    paths = sorted(RUN.glob("frame-*"))
    _, overlaps, _ = scan(paths, lut)
    overlaps.sort(
        key=lambda item: (-item.dimension, item.frame_index,
                          item.component.label))

    records = []
    page_rows = []
    page_number = 0
    bucket_counts = {profile.name: 0 for profile in RADIUS_PROFILES}
    hole_counts = {}
    fitted_count = 0

    for index, candidate in enumerate(overlaps, 1):
        component = candidate.component
        profile = component.profile
        frame_id = candidate.path.name.split("-")[1]
        fitted = fit_status(candidate)
        fitted_count += int(fitted)
        bucket_counts[profile.name] += 1
        hole_counts[str(candidate.holes)] = (
            hole_counts.get(str(candidate.holes), 0) + 1)
        filename = (
            f"{index:05d}-{candidate.dimension:03d}px-frame-{frame_id}-"
            f"label-{component.label:03d}-{candidate.holes}holes.png")
        row = render(candidate, "overlap-multi-hole", lut)
        output = instances / filename
        if not cv2.imwrite(str(output), row):
            raise OSError(f"Unable to write {output}")
        page_rows.append(row)

        records.append({
            "file": str(Path("instances") / filename),
            "frame": candidate.path.name,
            "label": component.label,
            "maximum_dimension_px": candidate.dimension,
            "bbox": list(component.bbox),
            "area_px": component.area,
            "significant_holes": candidate.holes,
            "solidity": candidate.solidity,
            "touches_frame": candidate.touches_frame,
            "radius_profile": profile.name,
            "density_radius": profile.density_radius,
            "ridge_radius": profile.ridge_radius,
            "p90_fit": fitted,
        })

        if len(page_rows) == ROWS_PER_PAGE or index == len(overlaps):
            page_number += 1
            page = np.vstack(page_rows)
            page_path = contacts / f"page-{page_number:03d}.png"
            if not cv2.imwrite(str(page_path), page):
                raise OSError(f"Unable to write {page_path}")
            print(page_path)
            page_rows.clear()

    manifest = {
        "run": str(RUN),
        "ordering": "maximum bounding-box dimension descending",
        "overlap_definition": (
            "one connected component with at least two enclosed contours "
            "having contour area >= 12 px"),
        "instances": len(overlaps),
        "p90_fitted_instances": fitted_count,
        "bucket_counts": bucket_counts,
        "significant_hole_counts": hole_counts,
        "profile": {
            "inverse_gamma": INVERSE_GAMMA,
            "ridge_gamma": RIDGE_GAMMA,
            "relative_cap": RELATIVE_CAP,
            "percentile": P90_PERCENTILE,
            "radius_profiles": [
                {
                    "name": profile.name,
                    "maximum_dimension": profile.maximum_dimension,
                    "density_radius": profile.density_radius,
                    "ridge_radius": profile.ridge_radius,
                }
                for profile in RADIUS_PROFILES
            ],
        },
        "records": records,
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=json_default) + "\n")
    print(manifest_path)
    print(json.dumps({
        "instances": len(overlaps),
        "pages": page_number,
        "p90_fitted_instances": fitted_count,
        "bucket_counts": bucket_counts,
        "significant_hole_counts": hole_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
