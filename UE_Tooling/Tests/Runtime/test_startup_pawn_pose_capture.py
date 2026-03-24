"""
Status: in development.
Minimal single-pose startup pawn capture test.

Purpose:
- Temporarily move the canonical preplaced startup pawn.
- Verify the saved level kept that transform.
- Run the real `run_unreal_io.py` flow once.
- Confirm the accepted PNG looks like a torus-visible capture.
- Restore the original startup pawn transform.

Boundaries:
- Test-only.
- One pose per invocation.
- No build ownership changes.
- No runtime ownership changes.
- No permanent default edits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency path
    Image = None


SCRIPT_NAME = "test_startup_pawn_pose_capture"
SCRIPT_VERSION = "0.3.0"
SCRIPT_DATE = "2026-03-23"

ACTOR_LABEL = "BP_DronePawn_Startup_Main"
DEFAULT_EDITOR_TIMEOUT_SECONDS = 180
DEFAULT_RUNTIME_TIMEOUT_SECONDS = 420
DEFAULT_NON_BLACK_THRESHOLD = 0.0005
DEFAULT_RED_DOMINANT_THRESHOLD = 0.0002

HELPER_SCRIPT_BODY = textwrap.dedent(
    r"""
    import argparse
    import json
    from pathlib import Path

    import unreal


    def _parse_args():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--mode", choices=("inspect", "apply"), required=True)
        parser.add_argument("--level", required=True)
        parser.add_argument("--actor-label", default="BP_DronePawn_Startup_Main")
        parser.add_argument("--json-out", required=True)
        parser.add_argument("--x", type=float, default=0.0)
        parser.add_argument("--y", type=float, default=0.0)
        parser.add_argument("--z", type=float, default=0.0)
        parser.add_argument("--pitch", type=float, default=0.0)
        parser.add_argument("--yaw", type=float, default=0.0)
        parser.add_argument("--roll", type=float, default=0.0)
        args, _ = parser.parse_known_args()
        return args


    def _find_actor_by_label(label):
        for actor in unreal.EditorLevelLibrary.get_all_level_actors() or []:
            try:
                if str(actor.get_actor_label()) == str(label):
                    return actor
            except Exception:
                continue
        return None


    def _record_actor(actor):
        location = actor.get_actor_location()
        rotation = actor.get_actor_rotation()
        return {
            "name": str(actor.get_name()),
            "label": str(actor.get_actor_label()),
            "location": [float(location.x), float(location.y), float(location.z)],
            "rotation": [float(rotation.pitch), float(rotation.yaw), float(rotation.roll)],
        }


    def main():
        args = _parse_args()
        json_out = Path(str(args.json_out))
        json_out.parent.mkdir(parents=True, exist_ok=True)
        if unreal.EditorLoadingAndSavingUtils.load_map(str(args.level)) is None:
            raise RuntimeError("Failed to load level: " + str(args.level))
        actor = _find_actor_by_label(str(args.actor_label))
        if actor is None:
            raise RuntimeError("Actor label not found: " + str(args.actor_label))

        payload = {
            "status": "success",
            "mode": str(args.mode),
            "level": str(args.level),
            "actor_label": str(args.actor_label),
            "before": _record_actor(actor),
        }

        if str(args.mode) == "apply":
            actor.set_actor_location(
                unreal.Vector(float(args.x), float(args.y), float(args.z)),
                False,
                False,
            )
            actor.set_actor_rotation(
                unreal.Rotator(float(args.roll), float(args.pitch), float(args.yaw)),
                False,
            )
            payload["save_current_level"] = bool(unreal.EditorLevelLibrary.save_current_level())
            payload["save_dirty_packages"] = bool(
                unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
            )
            if unreal.EditorLoadingAndSavingUtils.load_map(str(args.level)) is None:
                raise RuntimeError("Failed to reload level after save: " + str(args.level))
            actor = _find_actor_by_label(str(args.actor_label))
            if actor is None:
                raise RuntimeError("Actor label missing after reload: " + str(args.actor_label))

        payload["after"] = _record_actor(actor)
        json_out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        unreal.log("[test_startup_pawn_pose_capture_helper] " + json.dumps(payload))


    if __name__ == "__main__":
        main()
    """
).strip() + "\n"


@dataclass(frozen=True)
class Transform:
    location: tuple[float, float, float]
    rotation: tuple[float, float, float]


@dataclass(frozen=True)
class PoseSpec:
    name: str
    transform: Transform


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Failed to locate repo root.")


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _sanitize(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.strip())
    return cleaned.strip("_")


def _parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-manifest", required=True)
    parser.add_argument(
        "--pose",
        required=True,
        help="Required pose in the form name,x,y,z,pitch,yaw,roll",
    )
    parser.add_argument("--test-run-id", default="")
    parser.add_argument("--unreal-editor", default="")
    parser.add_argument("--editor-timeout-seconds", type=int, default=DEFAULT_EDITOR_TIMEOUT_SECONDS)
    parser.add_argument("--runtime-timeout-seconds", type=int, default=DEFAULT_RUNTIME_TIMEOUT_SECONDS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--non-black-threshold", type=float, default=DEFAULT_NON_BLACK_THRESHOLD)
    parser.add_argument("--red-dominant-threshold", type=float, default=DEFAULT_RED_DOMINANT_THRESHOLD)
    parser.add_argument(
        "--artifacts-root",
        default=str((repo_root / "UE_Tooling/Artifacts/tests").resolve()),
    )
    return parser.parse_args()


def _resolve_existing_path(path_value: str, repo_root: Path) -> Path:
    raw = Path(path_value).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(repo_root / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Path not found: {path_value}")


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _extract_authoritative_level(build_payload: dict) -> str:
    direct = str(build_payload.get("authoritative_course_level_path", "")).strip()
    if direct:
        return direct
    handoff = build_payload.get("runtime_handoff", {})
    if isinstance(handoff, dict):
        level = str(handoff.get("authoritative_level_path", "")).strip()
        if level:
            return level
    raise RuntimeError("Build manifest does not contain authoritative course level path.")


def _default_test_run_id() -> str:
    return f"startup_pawn_pose_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _write_helper_script() -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="startup_pawn_pose_capture_"))
    helper_path = temp_dir / "ue_startup_pawn_pose_capture_helper.py"
    helper_path.write_text(HELPER_SCRIPT_BODY, encoding="utf-8")
    return helper_path


def _run_logged_command(command: list[str], *, cwd: Path, log_path: Path, timeout_seconds: int) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _now_utc_iso()
    started_monotonic = time.monotonic()
    with log_path.open("wb") as handle:
        process = subprocess.Popen(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
        try:
            return_code = process.wait(timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
    return {
        "command": command,
        "log_path": str(log_path),
        "started_at_utc": started_at,
        "finished_at_utc": _now_utc_iso(),
        "duration_seconds": round(max(0.0, time.monotonic() - started_monotonic), 3),
        "return_code": int(return_code),
        "timed_out": bool(timed_out),
        "status": "success" if int(return_code) == 0 and not timed_out else "failed",
    }


def _build_editor_command(
    *,
    repo_root: Path,
    helper_script: Path,
    project_path: Path,
    unreal_editor: str,
    userdir: Path,
    ue_log_path: Path,
    timeout_seconds: int,
    level: str,
    mode: str,
    json_out: Path,
    transform: Transform | None,
) -> list[str]:
    build_executor = repo_root / "UE_Tooling/UE_Build/run_unreal_build_gen.py"
    command = [
        sys.executable,
        str(build_executor),
        "--script",
        str(helper_script),
        "--project",
        str(project_path),
        "--log",
        str(ue_log_path),
        "--userdir",
        str(userdir),
        "--timeout-seconds",
        str(int(timeout_seconds)),
        "--",
        "--mode",
        mode,
        "--level",
        level,
        "--actor-label",
        ACTOR_LABEL,
        "--json-out",
        str(json_out),
    ]
    if unreal_editor:
        command.extend(["--unreal-editor", unreal_editor])
    if transform is not None:
        x, y, z = transform.location
        pitch, yaw, roll = transform.rotation
        command.extend(
            [
                "--x",
                str(x),
                "--y",
                str(y),
                "--z",
                str(z),
                "--pitch",
                str(pitch),
                "--yaw",
                str(yaw),
                "--roll",
                str(roll),
            ]
        )
    return command


def _run_editor_helper(
    *,
    repo_root: Path,
    helper_script: Path,
    project_path: Path,
    unreal_editor: str,
    timeout_seconds: int,
    level: str,
    mode: str,
    json_out: Path,
    log_path: Path,
    userdir: Path,
    transform: Transform | None = None,
) -> tuple[dict, dict]:
    ue_log_path = log_path.with_suffix(".ue.log")
    command = _build_editor_command(
        repo_root=repo_root,
        helper_script=helper_script,
        project_path=project_path,
        unreal_editor=unreal_editor,
        userdir=userdir,
        ue_log_path=ue_log_path,
        timeout_seconds=timeout_seconds,
        level=level,
        mode=mode,
        json_out=json_out,
        transform=transform,
    )
    execution = _run_logged_command(
        command,
        cwd=repo_root,
        log_path=log_path,
        timeout_seconds=timeout_seconds + 60,
    )
    payload = _load_json(json_out) if json_out.exists() else {}
    execution["ue_log_path"] = str(ue_log_path)
    return execution, payload


def _transform_from_record(record: dict) -> Transform:
    location = record.get("location", [0.0, 0.0, 0.0])
    rotation = record.get("rotation", [0.0, 0.0, 0.0])
    return Transform(
        location=(float(location[0]), float(location[1]), float(location[2])),
        rotation=(float(rotation[0]), float(rotation[1]), float(rotation[2])),
    )


def _matches(lhs: Transform, rhs: Transform, tolerance: float = 0.01) -> bool:
    pairs = list(zip(lhs.location, rhs.location)) + list(zip(lhs.rotation, rhs.rotation))
    return all(abs(float(a) - float(b)) <= tolerance for a, b in pairs)


def _parse_pose(raw_value: str) -> PoseSpec:
    parts = [part.strip() for part in str(raw_value).split(",")]
    if len(parts) != 7:
        raise RuntimeError("Pose must use the form name,x,y,z,pitch,yaw,roll.")
    name = _sanitize(parts[0]) or "pose"
    x, y, z, pitch, yaw, roll = (float(item) for item in parts[1:])
    return PoseSpec(
        name=name,
        transform=Transform(
            location=(x, y, z),
            rotation=(pitch, yaw, roll),
        ),
    )


def _build_runtime_command(
    *,
    repo_root: Path,
    build_manifest_path: Path,
    runtime_run_id: str,
    runtime_artifacts_root: Path,
    host: str,
    port: int,
) -> list[str]:
    runtime_script = repo_root / "UE_Tooling/run_unreal_io.py"
    return [
        sys.executable,
        str(runtime_script),
        "--build-manifest",
        str(build_manifest_path),
        "--runtime-run-id",
        runtime_run_id,
        "--artifacts-root",
        str(runtime_artifacts_root),
        "--host",
        host,
        "--port",
        str(int(port)),
        "--strict-validation",
        "--require-client-connection",
        "--skip-phase8-spawn-probe",
    ]


def _resolve_accepted_sample_path(runtime_manifest: dict) -> str:
    phase_results = runtime_manifest.get("phase_results", {})
    if not isinstance(phase_results, dict):
        return ""
    phase3 = phase_results.get("phase3_bridge_supervision", {})
    phase7 = phase_results.get("phase7_sampler_smoke_capture", {})
    if not isinstance(phase3, dict) or not isinstance(phase7, dict):
        return ""
    bridge_summary_path = Path(str(phase3.get("bridge_summary_path", "")))
    if not bridge_summary_path.exists():
        return ""
    capture_evidence = phase7.get("capture_evidence", {})
    if not isinstance(capture_evidence, dict):
        return ""
    accepted_relative = str(capture_evidence.get("accepted_sample_file", "")).strip()
    if not accepted_relative:
        return ""
    return str((bridge_summary_path.parent / "samples" / accepted_relative).resolve())


def _analyze_image(path: Path, *, non_black_threshold: float, red_dominant_threshold: float) -> dict:
    result = {
        "status": "missing",
        "path": str(path),
        "exists": path.exists(),
        "width": 0,
        "height": 0,
        "pixel_count": 0,
        "non_black_ratio": 0.0,
        "red_dominant_ratio": 0.0,
        "torus_visibility_heuristic_passed": False,
        "heuristic_reason": "",
    }
    if not path.exists():
        result["heuristic_reason"] = "accepted_sample_missing"
        return result
    if Image is None:
        result["status"] = "analysis_unavailable"
        result["heuristic_reason"] = "Pillow unavailable."
        return result

    image = Image.open(path).convert("RGB")
    width, height = image.size
    pixel_count = width * height
    non_black_count = 0
    red_dominant_count = 0
    for red, green, blue in image.getdata():
        if max(red, green, blue) >= 16:
            non_black_count += 1
        if red >= 24 and red >= green + 12 and red >= blue + 12:
            red_dominant_count += 1
    non_black_ratio = float(non_black_count) / float(max(pixel_count, 1))
    red_dominant_ratio = float(red_dominant_count) / float(max(pixel_count, 1))
    passed = non_black_ratio >= float(non_black_threshold) and red_dominant_ratio >= float(red_dominant_threshold)
    result.update(
        {
            "status": "success",
            "width": int(width),
            "height": int(height),
            "pixel_count": int(pixel_count),
            "non_black_ratio": non_black_ratio,
            "red_dominant_ratio": red_dominant_ratio,
            "torus_visibility_heuristic_passed": bool(passed),
            "heuristic_reason": (
                "image passed torus visibility heuristic"
                if passed
                else "image did not meet non-black/red-dominant torus thresholds"
            ),
        }
    )
    return result


def main() -> int:
    repo_root = _repo_root()
    args = _parse_args(repo_root)
    pose = _parse_pose(str(args.pose))
    build_manifest_path = _resolve_existing_path(str(args.build_manifest), repo_root)
    build_payload = _load_json(build_manifest_path)
    authoritative_level = _extract_authoritative_level(build_payload)
    project_path = _resolve_existing_path("UE_Drone_Env/UE_Drone_Env.uproject", repo_root)

    test_run_id = _sanitize(str(args.test_run_id)) or _default_test_run_id()
    artifacts_root = _resolve_existing_path(str(args.artifacts_root), repo_root.parent) if Path(str(args.artifacts_root)).exists() else Path(str(args.artifacts_root)).expanduser()
    if not artifacts_root.is_absolute():
        artifacts_root = (repo_root / artifacts_root).resolve()
    test_dir = artifacts_root / test_run_id
    logs_dir = test_dir / "logs"
    json_dir = test_dir / "json"
    runtime_artifacts_root = test_dir / "runtime"
    for directory in (test_dir, logs_dir, json_dir, runtime_artifacts_root):
        directory.mkdir(parents=True, exist_ok=True)

    helper_script = _write_helper_script()
    summary = {
        "script": {"name": SCRIPT_NAME, "version": SCRIPT_VERSION, "date": SCRIPT_DATE},
        "status": "in_progress",
        "started_at_utc": _now_utc_iso(),
        "finished_at_utc": "",
        "test_run_id": test_run_id,
        "test_dir": str(test_dir),
        "build_manifest_path": str(build_manifest_path),
        "authoritative_level_path": authoritative_level,
        "actor_label": ACTOR_LABEL,
        "requested_pose": {
            "name": pose.name,
            "location": list(pose.transform.location),
            "rotation": list(pose.transform.rotation),
        },
        "initial_inspect": {},
        "apply": {},
        "verify": {},
        "runtime": {},
        "image_analysis": {},
        "restore": {},
        "errors": [],
    }

    original_transform = None

    try:
        inspect_exec, inspect_payload = _run_editor_helper(
            repo_root=repo_root,
            helper_script=helper_script,
            project_path=project_path,
            unreal_editor=str(args.unreal_editor).strip(),
            timeout_seconds=max(1, int(args.editor_timeout_seconds)),
            level=authoritative_level,
            mode="inspect",
            json_out=json_dir / "initial_inspect.json",
            log_path=logs_dir / "initial_inspect.log",
            userdir=test_dir / "editor_userdir" / "initial_inspect",
        )
        summary["initial_inspect"] = {"execution": inspect_exec, "payload": inspect_payload}
        if inspect_exec.get("status") != "success":
            raise RuntimeError("Initial startup pawn inspection failed.")
        original_transform = _transform_from_record(inspect_payload.get("before", {}))
        summary["original_transform"] = {
            "location": list(original_transform.location),
            "rotation": list(original_transform.rotation),
        }

        apply_exec, apply_payload = _run_editor_helper(
            repo_root=repo_root,
            helper_script=helper_script,
            project_path=project_path,
            unreal_editor=str(args.unreal_editor).strip(),
            timeout_seconds=max(1, int(args.editor_timeout_seconds)),
            level=authoritative_level,
            mode="apply",
            json_out=json_dir / "apply.json",
            log_path=logs_dir / "apply.log",
            userdir=test_dir / "editor_userdir" / "apply",
            transform=pose.transform,
        )
        summary["apply"] = {"execution": apply_exec, "payload": apply_payload}
        if apply_exec.get("status") != "success":
            raise RuntimeError("Apply step failed.")

        verify_exec, verify_payload = _run_editor_helper(
            repo_root=repo_root,
            helper_script=helper_script,
            project_path=project_path,
            unreal_editor=str(args.unreal_editor).strip(),
            timeout_seconds=max(1, int(args.editor_timeout_seconds)),
            level=authoritative_level,
            mode="inspect",
            json_out=json_dir / "verify.json",
            log_path=logs_dir / "verify.log",
            userdir=test_dir / "editor_userdir" / "verify",
        )
        persisted_transform = _transform_from_record(verify_payload.get("before", {}))
        persisted = verify_exec.get("status") == "success" and _matches(pose.transform, persisted_transform)
        summary["verify"] = {
            "execution": verify_exec,
            "payload": verify_payload,
            "persisted_transform_matches_requested": persisted,
        }
        if not persisted:
            raise RuntimeError("Requested startup pawn transform did not persist.")

        runtime_run_id = f"{test_run_id}_{pose.name}"
        runtime_exec = _run_logged_command(
            _build_runtime_command(
                repo_root=repo_root,
                build_manifest_path=build_manifest_path,
                runtime_run_id=runtime_run_id,
                runtime_artifacts_root=runtime_artifacts_root,
                host=str(args.host),
                port=int(args.port),
            ),
            cwd=repo_root,
            log_path=logs_dir / "run_unreal_io.log",
            timeout_seconds=max(1, int(args.runtime_timeout_seconds)),
        )
        runtime_manifest_path = runtime_artifacts_root / runtime_run_id / "runtime_session_manifest.json"
        runtime_payload = _load_json(runtime_manifest_path) if runtime_manifest_path.exists() else {}
        accepted_sample_raw = _resolve_accepted_sample_path(runtime_payload) if runtime_payload else ""
        accepted_sample_path = Path(accepted_sample_raw) if accepted_sample_raw else None
        image_analysis = (
            _analyze_image(
                accepted_sample_path,
                non_black_threshold=float(args.non_black_threshold),
                red_dominant_threshold=float(args.red_dominant_threshold),
            )
            if accepted_sample_path is not None
            else {
                "status": "missing",
                "path": "",
                "exists": False,
                "torus_visibility_heuristic_passed": False,
                "heuristic_reason": "accepted_sample_path_unavailable",
            }
        )
        summary["runtime"] = {
            "execution": runtime_exec,
            "runtime_run_id": runtime_run_id,
            "runtime_manifest_path": str(runtime_manifest_path),
            "runtime_manifest_status": str(runtime_payload.get("status", "")) if runtime_payload else "",
            "accepted_sample_path": str(accepted_sample_path) if accepted_sample_path else "",
        }
        summary["image_analysis"] = image_analysis
        if runtime_exec.get("status") != "success" or runtime_payload.get("status") != "success":
            raise RuntimeError("Runtime capture flow failed.")
        if not bool(image_analysis.get("torus_visibility_heuristic_passed", False)):
            raise RuntimeError("Accepted image did not pass torus visibility heuristic.")

    except Exception as exc:
        summary["errors"].append(str(exc))
    finally:
        if original_transform is not None:
            restore_exec, restore_payload = _run_editor_helper(
                repo_root=repo_root,
                helper_script=helper_script,
                project_path=project_path,
                unreal_editor=str(args.unreal_editor).strip(),
                timeout_seconds=max(1, int(args.editor_timeout_seconds)),
                level=authoritative_level,
                mode="apply",
                json_out=json_dir / "restore_apply.json",
                log_path=logs_dir / "restore_apply.log",
                userdir=test_dir / "editor_userdir" / "restore_apply",
                transform=original_transform,
            )
            restore_verify_exec, restore_verify_payload = _run_editor_helper(
                repo_root=repo_root,
                helper_script=helper_script,
                project_path=project_path,
                unreal_editor=str(args.unreal_editor).strip(),
                timeout_seconds=max(1, int(args.editor_timeout_seconds)),
                level=authoritative_level,
                mode="inspect",
                json_out=json_dir / "restore_verify.json",
                log_path=logs_dir / "restore_verify.log",
                userdir=test_dir / "editor_userdir" / "restore_verify",
            )
            restored_transform = _transform_from_record(restore_verify_payload.get("before", {}))
            restored = restore_verify_exec.get("status") == "success" and _matches(original_transform, restored_transform)
            summary["restore"] = {
                "apply_execution": restore_exec,
                "verify_execution": restore_verify_exec,
                "restored_matches_original": restored,
            }
            if not restored:
                summary["errors"].append("Original startup pawn transform was not restored successfully.")

    summary["status"] = "success" if not summary["errors"] else "failed"
    summary["finished_at_utc"] = _now_utc_iso()
    summary_path = test_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "status": summary["status"]}, indent=2))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
