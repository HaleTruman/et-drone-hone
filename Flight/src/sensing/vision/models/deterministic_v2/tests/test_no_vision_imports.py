from pathlib import Path


def test_runtime_python_files_do_not_import_legacy_workbench_modules():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    runtime_paths = [root / "main.py", *sorted((root / "src").rglob("*.py"))]
    for path in runtime_paths:
        text = path.read_text(encoding="utf-8")
        banned = [
            "from review_vision_ui",
            "import review_vision_ui",
            "from serve_vision_review",
            "import serve_vision_review",
            "from flight_bridge",
            "import flight_bridge",
            "from pose_estimation",
            "import pose_estimation",
            "from instance_tracking",
            "import instance_tracking",
            "from color_masking",
            "import color_masking",
            "from bboxing",
            "import bboxing",
            "from contouring",
            "import contouring",
        ]
        if any(item in text for item in banned):
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
