from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vision.src.cnn import lightmask_model
from vision.src.cnn.rgb_inference import DEFAULT_CHECKPOINT, LightmaskInference
from vision.src.cnn.rgb_normalizer import jpeg_bytes_to_tensor
from vision.src.io.udp_protocol import sorted_jpeg_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_FRAMES_DIR = (
    REPO_ROOT
    / "vision/tools/sample_runs/universe_75m75g50d_nearest_front_facing_75gates_fullrun_retry720_20260605"
)


def _require_checkpoint() -> None:
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip(f"checkpoint missing: {DEFAULT_CHECKPOINT}")


def test_cnn_runtime_no_longer_imports_training_pipeline() -> None:
    cnn_runtime_files = [
        REPO_ROOT / "vision/src/cnn/rgb_inference.py",
        REPO_ROOT / "vision/src/cnn/lightmask_model.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in cnn_runtime_files)

    assert "training_pipeline" not in combined
    assert "from lib." not in combined
    assert "sys.path" not in combined
    assert "Convolutional_Neural_Network" not in combined


def test_cnn_checkpoint_schema_matches_standalone_runtime_contract() -> None:
    _require_checkpoint()
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu")
    config = checkpoint.get("config", {})

    assert config["schema_version"] == lightmask_model.SCHEMA_VERSION
    assert config["backbone"] in lightmask_model.BACKBONES
    assert config["depth_head_enabled"] is True
    assert config["image"]["width"] == lightmask_model.IMAGE_WIDTH
    assert config["image"]["height"] == lightmask_model.IMAGE_HEIGHT
    assert config["output"]["stride"] == lightmask_model.OUTPUT_STRIDE
    assert config["output"]["heatmap_width"] == lightmask_model.HEATMAP_WIDTH
    assert config["output"]["heatmap_height"] == lightmask_model.HEATMAP_HEIGHT
    assert tuple(config["output"]["mask_channels"]) == lightmask_model.MASK_CHANNELS
    assert tuple(config["output"]["depth_channels"]) == lightmask_model.DEPTH_CHANNELS
    assert "model_state_dict" in checkpoint


def test_standalone_lightmask_model_loads_checkpoint_strictly() -> None:
    _require_checkpoint()
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu")
    config = checkpoint["config"]
    model = lightmask_model.build_model(
        backbone=str(config.get("backbone", "mobilenet_v3_small")),
        output_channels=len(lightmask_model.MASK_CHANNELS),
        weights="none",
        decoder_channels=int(config.get("decoder_channels", 32)),
        decoder_blocks=int(config.get("decoder_blocks", 2)),
        dropout=float(config.get("dropout", 0.0)),
        enable_depth_head=bool(config.get("depth_head_enabled", False)),
    )

    lightmask_model.load_compatible_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 3, lightmask_model.IMAGE_HEIGHT, lightmask_model.IMAGE_WIDTH), dtype=torch.float32))

    assert tuple(output["mask_logits"].shape) == (
        1,
        len(lightmask_model.MASK_CHANNELS),
        lightmask_model.HEATMAP_HEIGHT,
        lightmask_model.HEATMAP_WIDTH,
    )
    assert tuple(output["depth_logits"].shape) == (
        1,
        len(lightmask_model.DEPTH_CHANNELS),
        lightmask_model.HEATMAP_HEIGHT,
        lightmask_model.HEATMAP_WIDTH,
    )
    assert torch.isfinite(output["mask_logits"]).all()
    assert torch.isfinite(output["depth_logits"]).all()


def test_lightmask_inference_runs_sample_frame_with_local_model() -> None:
    _require_checkpoint()
    samples = sorted_jpeg_paths(SAMPLE_FRAMES_DIR)[:1]
    if not samples:
        pytest.skip(f"sample frame missing: {SAMPLE_FRAMES_DIR}")

    runner = LightmaskInference(DEFAULT_CHECKPOINT, device="cpu")
    result = runner.run_frame(jpeg_bytes_to_tensor(samples[0].read_bytes()))

    assert tuple(result.mask_logits.shape) == (
        len(lightmask_model.MASK_CHANNELS),
        lightmask_model.HEATMAP_HEIGHT,
        lightmask_model.HEATMAP_WIDTH,
    )
    assert tuple(result.depth_logits.shape) == (
        len(lightmask_model.DEPTH_CHANNELS),
        lightmask_model.HEATMAP_HEIGHT,
        lightmask_model.HEATMAP_WIDTH,
    )
    assert result.mask_logits.dtype == torch.float32
    assert result.depth_logits.dtype == torch.float32
    assert torch.isfinite(result.mask_logits).all()
    assert torch.isfinite(result.depth_logits).all()
