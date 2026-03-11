import argparse
import shutil
from pathlib import Path

SCRIPT_NAME = "sync_plugin_source"
SCRIPT_VERSION = "0.1.0"
SCRIPT_DATE = "2026-03-11"

IGNORED_DIRS = {"Binaries", "Intermediate", "Saved"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direction",
        choices=["tooling_to_project", "project_to_tooling"],
        default="tooling_to_project",
        help="Copy direction for plugin source sync.",
    )
    return parser.parse_args()


def _copytree_filtered(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*IGNORED_DIRS),
    )


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    tooling_plugin = repo_root / "UE_Tooling/UE_Build/WebSocket/Plugin_Source/DroneWebSocket"
    ue_plugin = repo_root / "UE_Drone_Env/Plugins/DroneWebSocket"

    if args.direction == "tooling_to_project":
        source = tooling_plugin
        target = ue_plugin
    else:
        source = ue_plugin
        target = tooling_plugin

    if not source.exists():
        print(f"[{SCRIPT_NAME}] source does not exist: {source}")
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    _copytree_filtered(source, target)
    print(f"[{SCRIPT_NAME}] synced {source} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
