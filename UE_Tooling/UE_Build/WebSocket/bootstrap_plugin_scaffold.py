import shutil
from pathlib import Path

SCRIPT_NAME = "bootstrap_plugin_scaffold"
SCRIPT_VERSION = "0.1.0"
SCRIPT_DATE = "2026-03-11"

IGNORED_DIRS = {"Binaries", "Intermediate", "Saved"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _copytree_filtered(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*IGNORED_DIRS),
    )


def main() -> int:
    repo_root = _repo_root()
    tooling_plugin = repo_root / "UE_Tooling/UE_Build/WebSocket/Plugin_Source/DroneWebSocket"
    ue_plugin = repo_root / "UE_Drone_Env/Plugins/DroneWebSocket"

    if tooling_plugin.exists():
        print(f"[{SCRIPT_NAME}] tooling source already exists: {tooling_plugin}")
        return 0

    if ue_plugin.exists():
        tooling_plugin.parent.mkdir(parents=True, exist_ok=True)
        _copytree_filtered(ue_plugin, tooling_plugin)
        print(f"[{SCRIPT_NAME}] bootstrapped tooling source from UE plugin: {ue_plugin}")
        return 0

    (tooling_plugin / "Source/DroneWebSocket/Public").mkdir(parents=True, exist_ok=True)
    (tooling_plugin / "Source/DroneWebSocket/Private").mkdir(parents=True, exist_ok=True)
    (tooling_plugin / "README.md").write_text(
        "# DroneWebSocket plugin source scaffold\n",
        encoding="utf-8",
    )
    print(f"[{SCRIPT_NAME}] created empty scaffold: {tooling_plugin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
