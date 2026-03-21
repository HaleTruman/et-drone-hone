"""
Desired behavior:
- Validate that the named websocket plugin in the UE project is structurally aligned, engine-compatible, built, and startup-ready.
- Report a clear pass/fail result with exact validation failures.
- Own post-bootstrap readiness checks rather than rebuilding the plugin.

Interfaces:
- Checks `UE_Tooling/UE_Build/WebSocket/Plugin_Source/<PluginName>` and `UE_Drone_Env/Plugins/<PluginName>`.
- Validates descriptor compatibility, source/project sync, expected build outputs, and headless editor startup.
- Uses `assemble_websocket_assets.py` for shared validation helpers and result reporting.

Assumptions:
- Bootstrap has already prepared or rebuilt the named project plugin as needed.
- Startup readiness requires more than file presence alone.
- Current required feature scope is limited to the known transport/protocol/handshake files.

Success conditions:
- Validation passes only when source and project trees are structurally valid, in sync, engine-compatible, contain expected build outputs, and survive a headless editor startup probe.
- Validation failures return non-zero and identify the failed checks precisely.
"""

from __future__ import annotations

import argparse
import sys

from assemble_websocket_assets import (
    SCRIPT_DATE as ASSEMBLY_SCRIPT_DATE,
    SCRIPT_VERSION as ASSEMBLY_SCRIPT_VERSION,
    compare_source_snapshots,
    emit_result,
    expected_binary_relative_paths,
    log,
    plugin_descriptor_validation,
    plugin_paths,
    read_json,
    startup_probe,
    validate_expected_binaries,
    validate_required_files,
)

SCRIPT_NAME = "plugin_validate"
SCRIPT_VERSION = "2.0.0"
SCRIPT_DATE = "2026-03-14"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-name", default="DroneWebSocket")
    parser.add_argument("--project", default="")
    parser.add_argument("--engine-root", default="")
    parser.add_argument("--editor-binary", default="")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--startup-map", default="/Engine/Maps/Entry")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        paths = plugin_paths(
            plugin_name=str(args.plugin_name),
            project_override=str(args.project) or None,
            engine_root_override=str(args.engine_root) or None,
            editor_binary_override=str(args.editor_binary) or None,
        )
        log(
            f"script={SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE} "
            f"assembly_version={ASSEMBLY_SCRIPT_VERSION} assembly_date={ASSEMBLY_SCRIPT_DATE}"
        )

        failures: list[str] = []
        project_descriptor = read_json(paths.project_file)
        source_required = validate_required_files(paths.source_root, paths.plugin_name)
        project_required = validate_required_files(paths.project_root, paths.plugin_name)
        source_sync = compare_source_snapshots(paths.source_root, paths.project_root)
        descriptor_validation = plugin_descriptor_validation(paths.project_root, paths.plugin_name, project_descriptor)
        binary_validation = validate_expected_binaries(paths.project_root, paths.plugin_name, paths.host_platform)

        if not source_required["is_valid"]:
            failures.append("source_required_files")
        if not project_required["is_valid"]:
            failures.append("project_required_files")
        if not source_sync["is_in_sync"]:
            failures.append("source_project_sync")
        if not descriptor_validation["is_valid"]:
            failures.append("descriptor_compatibility")
        if not binary_validation["is_valid"]:
            failures.append("build_outputs")

        startup_validation = {
            "skipped": bool(failures),
            "reason": "startup probe skipped due to earlier validation failures" if failures else "",
            "expected_binary_paths": list(expected_binary_relative_paths(paths.plugin_name, paths.host_platform)),
        }
        if not failures:
            startup_validation = startup_probe(
                paths,
                timeout_seconds=max(60, int(args.timeout_seconds)),
                startup_map=str(args.startup_map),
            )
            if not startup_validation["success"]:
                failures.append("startup_probe")

        details = {
            "plugin_name": paths.plugin_name,
            "project_root": str(paths.project_root),
            "source_root": str(paths.source_root),
            "project_file": str(paths.project_file),
            "host_platform": paths.host_platform,
            "source_required": source_required,
            "project_required": project_required,
            "source_sync": source_sync,
            "descriptor_validation": descriptor_validation,
            "binary_validation": binary_validation,
            "startup_validation": startup_validation,
            "failures": failures,
        }
    except Exception as exc:
        emit_result(
            SCRIPT_NAME,
            "failed",
            {
                "plugin_name": str(args.plugin_name),
                "error": str(exc),
            },
        )
        print(f"[{SCRIPT_NAME}] ERROR: {exc}", file=sys.stderr)
        return 1

    if failures:
        emit_result(SCRIPT_NAME, "failed", details)
        print(f"[{SCRIPT_NAME}] ERROR: validation failed: {', '.join(failures)}", file=sys.stderr)
        return 1

    emit_result(SCRIPT_NAME, "success", details)
    log(f"validation passed for plugin_name={paths.plugin_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
