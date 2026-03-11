import shutil
from pathlib import Path

SCRIPT_NAME = "deploy_prebuilt_plugin"
SCRIPT_VERSION = "0.1.0"
SCRIPT_DATE = "2026-03-11"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    repo_root = _repo_root()
    prebuilt_plugin = repo_root / "UE_Tooling/UE_Build/WebSocket/Prebuilt/DroneWebSocket"
    ue_plugin = repo_root / "UE_Drone_Env/Plugins/DroneWebSocket"

    if not prebuilt_plugin.exists():
        print(f"[{SCRIPT_NAME}] missing prebuilt plugin payload: {prebuilt_plugin}")
        return 1

    ue_plugin.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(prebuilt_plugin / "DroneWebSocket.uplugin", ue_plugin / "DroneWebSocket.uplugin")
    _copy_if_exists(prebuilt_plugin / "Binaries", ue_plugin / "Binaries")
    _copy_if_exists(prebuilt_plugin / "Resources", ue_plugin / "Resources")
    _copy_if_exists(prebuilt_plugin / "Content", ue_plugin / "Content")

    print(f"[{SCRIPT_NAME}] deployed prebuilt payload into: {ue_plugin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
