from __future__ import annotations

from pathlib import Path

from vision.src.io.udp_protocol import frame_id_from_path, sorted_jpeg_paths


def test_frame_id_from_path_accepts_decoded_rgb_png_convention() -> None:
    assert frame_id_from_path(Path("frame_000001_rgb.png")) == 1
    assert frame_id_from_path(Path("sample_frame_000002.jpg")) == 2
    assert frame_id_from_path(Path("sample_frame_003.jpeg")) == 3
    assert frame_id_from_path(Path("frames/frame_000004/rgb.jpg")) == 4


def test_sorted_jpeg_paths_accepts_png_and_legacy_jpeg_names(tmp_path: Path) -> None:
    for name in (
        "frame_000003_rgb.png",
        "frame_000004_instance.png",
        "sample_frame_000001.jpg",
        "sample_frame_000002.jpeg",
        "ignore.txt",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")

    paths = sorted_jpeg_paths(tmp_path)

    assert [path.name for path in paths] == [
        "sample_frame_000001.jpg",
        "sample_frame_000002.jpeg",
        "frame_000003_rgb.png",
    ]


def test_sorted_jpeg_paths_accepts_manifest_frame_directories(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    for frame_id in (3, 1, 2):
        frame_dir = frames_dir / f"frame_{frame_id:06d}"
        frame_dir.mkdir(parents=True)
        (frame_dir / "rgb.jpg").write_text("x", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        """
{
  "frames": [
    {"frame_id": 3, "rgb": "frames/frame_000003/rgb.jpg"},
    {"frame_id": 1, "rgb": "frames/frame_000001/rgb.jpg"},
    {"frame_id": 2, "rgb": "frames/frame_000002/rgb.jpg"}
  ]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    paths = sorted_jpeg_paths(tmp_path)

    assert [path.parent.name for path in paths] == ["frame_000001", "frame_000002", "frame_000003"]
