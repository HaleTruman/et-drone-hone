"""
Desired behavior:
- Canonically bootstrap the named websocket plugin into the UE project.
- Keep an existing named project plugin by default, and rebuild it only when explicitly requested or when missing.
- Fail fast on blocking bootstrap/build errors without performing post-build validation.

Interfaces:
- Reads authoritative source from `UE_Tooling/UE_Build/WebSocket/Plugin_Source/<PluginName>`.
- Stages source into `UE_Drone_Env/Plugins/<PluginName>` as build input when needed.
- Invokes the UE project editor build through Unreal's build tooling.

Assumptions:
- `Plugin_Source/<PluginName>` is the only source-of-truth.
- Bootstrap owns project plugin staging and build invocation, not startup-readiness validation.
- Validation is performed separately by `plugin_validate.py`.

Success conditions:
- If the named project plugin already exists and overwrite is not requested, bootstrap reports it as available and exits cleanly.
- If the named project plugin is missing or overwrite is requested, bootstrap stages source and completes the UE-side project build.
- Build failures return non-zero with the exact build command and output.
"""

from __future__ import annotations

import argparse
import sys

from assemble_websocket_assets import (
    SCRIPT_DATE as ASSEMBLY_SCRIPT_DATE,
    SCRIPT_VERSION as ASSEMBLY_SCRIPT_VERSION,
    build_project_plugin,
    emit_result,
    ensure_bootstrap_inputs,
    log,
    plugin_paths,
    stage_source_in_project,
)

SCRIPT_NAME = "bootstrap_plugin"
SCRIPT_VERSION = "2.0.0"
SCRIPT_DATE = "2026-03-14"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-name", default="DroneWebSocket")
    parser.add_argument("--project", default="")
    parser.add_argument("--engine-root", default="")
    parser.add_argument("--build-script", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = plugin_paths(
        plugin_name=str(args.plugin_name),
        project_override=str(args.project) or None,
        engine_root_override=str(args.engine_root) or None,
        build_script_override=str(args.build_script) or None,
    )
    log(
        f"script={SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE} "
        f"assembly_version={ASSEMBLY_SCRIPT_VERSION} assembly_date={ASSEMBLY_SCRIPT_DATE}"
    )

    try:
        bootstrap_inputs = ensure_bootstrap_inputs(paths)
        action = stage_source_in_project(paths.source_root, paths.project_root, overwrite=bool(args.overwrite))

        build_result = None
        if action != "kept_existing":
            build_result = build_project_plugin(paths, timeout_seconds=max(60, int(args.timeout_seconds)))
            sys.stdout.write(build_result["stdout"])
            sys.stderr.write(build_result["stderr"])
            if not build_result["success"]:
                raise RuntimeError(
                    f"Project build failed for {paths.project_editor_target} with exit code {build_result['return_code']}"
                )
    except Exception as exc:
        emit_result(
            SCRIPT_NAME,
            "failed",
            {
                "plugin_name": paths.plugin_name if "paths" in locals() else str(args.plugin_name),
                "overwrite": bool(args.overwrite),
                "error": str(exc),
                "project_root": str(paths.project_root) if "paths" in locals() else "",
                "source_root": str(paths.source_root) if "paths" in locals() else "",
                "build_result": build_result if "build_result" in locals() else None,
            },
        )
        print(f"[{SCRIPT_NAME}] ERROR: {exc}", file=sys.stderr)
        return 1

    details = {
        "plugin_name": paths.plugin_name,
        "overwrite": bool(args.overwrite),
        "action": action,
        "source_root": str(paths.source_root),
        "project_root": str(paths.project_root),
        "project_file": str(paths.project_file),
        "project_editor_target": paths.project_editor_target,
        "bootstrap_inputs": bootstrap_inputs,
        "build_invoked": action != "kept_existing",
        "build_result": build_result,
    }
    emit_result(SCRIPT_NAME, "success", details)
    log(f"completed action={action} plugin_name={paths.plugin_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
