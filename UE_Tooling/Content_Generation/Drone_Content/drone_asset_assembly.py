import unreal

SCRIPT_NAME = "drone_asset_assembly"
SCRIPT_VERSION = "0.1.0"
SCRIPT_DATE = "2026-03-10"


def log(message: str) -> None:
    unreal.log(f"[DroneAssetAssembly] {message}")


def log_script_metadata() -> None:
    log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def ensure_directory(asset_dir: str) -> None:
    if unreal.EditorAssetLibrary.does_directory_exist(asset_dir):
        return
    require(unreal.EditorAssetLibrary.make_directory(asset_dir), f"Failed to create directory: {asset_dir}")


def save_asset(asset_path: str) -> None:
    require(
        unreal.EditorAssetLibrary.save_asset(asset_path, only_if_is_dirty=False),
        f"Failed to save asset: {asset_path}",
    )


def create_or_load_asset(asset_path: str, asset_name: str, asset_dir: str, asset_class, factory) -> tuple[object, bool]:
    ensure_directory(asset_dir)
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        asset = unreal.EditorAssetLibrary.load_asset(asset_path)
        require(asset is not None, f"Failed to load asset: {asset_path}")
        return asset, False
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    asset = asset_tools.create_asset(asset_name, asset_dir, asset_class, factory)
    require(asset is not None, f"Failed to create asset: {asset_path}")
    return asset, True


def placeholder_note(target_asset_path: str) -> None:
    log(f"TODO implement create/update logic for: {target_asset_path}")
