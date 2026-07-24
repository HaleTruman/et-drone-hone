from pathlib import Path
import json

import numpy as np
from PIL import Image
import pytest

from vision.src.config import DEFAULT_PRESET_PATH, ProjectionConfig
from vision.src.pipeline import PipelineOptions, run_pipeline


def _nonzero_rgb(config: ProjectionConfig) -> tuple[int, int, int]:
    with np.load(config.lut_path, allow_pickle=False) as payload:
        key = int(np.flatnonzero(payload["lut"])[0])
    return ((key >> 16) & 0xFF, (key >> 8) & 0xFF, key & 0xFF)


def _write_frame(path: Path) -> None:
    config = ProjectionConfig.from_path(DEFAULT_PRESET_PATH)
    rgb = _nonzero_rgb(config)
    image = Image.new("RGB", (640, 360), (0, 0, 0))
    for x in range(260, 380):
        for y in range(120, 240):
            if not (300 <= x < 340 and 160 <= y < 200):
                image.putpixel((x, y), rgb)
    image.save(path, quality=100, subsampling=0)


def test_pipeline_one_frame_production_and_debug(tmp_path: Path):
    source_dir = tmp_path / "frames"
    source_dir.mkdir()
    _write_frame(source_dir / "frame_000000.jpg")

    prod_manifest = run_pipeline(
        PipelineOptions(
            mode="batch",
            source_dir=source_dir,
            output_root=tmp_path / "prod",
            run_id="prod-run",
            debug=False,
            preset_path=DEFAULT_PRESET_PATH,
        )
    )
    prod_root = Path(prod_manifest.run_root)
    assert (prod_root / "frames" / "frame_000000.json").exists()
    assert (prod_root / "latest.json").exists()
    assert (prod_root / "status.json").exists()
    assert (prod_root / "run_manifest.json").exists()
    assert not (prod_root / "debug").exists()

    debug_manifest = run_pipeline(
        PipelineOptions(
            mode="batch",
            source_dir=source_dir,
            output_root=tmp_path / "debug",
            run_id="debug-run",
            debug=True,
            preset_path=DEFAULT_PRESET_PATH,
        )
    )
    debug_root = Path(debug_manifest.run_root)
    assert len(list((debug_root / "debug" / "color_mask" / "frames").glob("*.json"))) == 1
    for profile_id in ["A", "B", "C"]:
        for stage in ["bbox", "inner_voids", "geometry_fits", "solve_pnp", "tracking"]:
            files = list((debug_root / "debug" / "profiles" / profile_id / stage / "frames").glob("*.json"))
            assert len(files) == 1
    assert (debug_root / "debug" / "color_mask" / "masks" / "frame_000000.bin").exists()
    assert (debug_root / "debug" / "global_mapping" / "run.json").exists()
    frame_payload = json.loads((debug_root / "frames" / "frame_000000.json").read_text(encoding="utf-8"))
    assert frame_payload["stage_counts"]["profiles"] == 3
    assert "vision_results" in frame_payload


def test_pipeline_runs_available_repo_frame(tmp_path: Path):
    source_dir = Path(__file__).resolve().parents[2] / "run-20260720T023225Z" / "vision_frames"
    if not source_dir.exists():
        source_dir = Path(__file__).resolve().parents[1] / "assets"
    if not any(source_dir.glob("*.jpg")):
        pytest.skip("repo frame resource is not available")
    manifest = run_pipeline(
        PipelineOptions(
            mode="batch",
            source_dir=source_dir,
            output_root=tmp_path / "real-frame",
            run_id="real-frame-run",
            debug=False,
            max_frames=1,
            preset_path=DEFAULT_PRESET_PATH,
        )
    )
    root = Path(manifest.run_root)
    payload = json.loads((root / "frames" / "frame_000000.json").read_text(encoding="utf-8"))
    assert manifest.frame_count == 1
    assert payload["stage_counts"]["profiles"] == 3
    assert isinstance(payload["vision_results"]["gates"], list)
