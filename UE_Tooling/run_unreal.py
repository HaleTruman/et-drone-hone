"""
TODO: Rename this runner to `run_unreal_build_gen.py` when runtime-control launch paths are
split into a dedicated runner.

Current scope:
- Launch Unreal for editor-time generation/build scripts.
- Default script target is under `UE_Tooling/UE_Build/Content_Generation`.

Current dependencies:
- UnrealEditor binary (`--unreal-editor` or `UNREAL_EDITOR`).
- `UE_Drone_Env/UE_Drone_Env.uproject`.
- A valid UE Python script path passed through `--script`.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _default_unreal_editor_path() -> Optional[Path]:
    candidates = [
        Path("/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    repo_root = _repo_root()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unreal-editor",
        default=os.environ.get("UNREAL_EDITOR") or "",
        help="Path to UnrealEditor binary (or set UNREAL_EDITOR).",
    )
    parser.add_argument(
        "--project",
        default=str((repo_root / "UE_Drone_Env/UE_Drone_Env.uproject").resolve()),
        help="Path to .uproject",
    )
    parser.add_argument(
        "--script",
        default=str((repo_root / "UE_Tooling/UE_Build/Content_Generation/Course_Content/Maps/gen_l_coursetorus.py").resolve()),
        help="Path to UE Python script to run via -ExecutePythonScript",
    )
    parser.add_argument(
        "--level",
        default="/Game/Course_Content/Maps/L_CourseTorus",
        help="Map package path passed to the UE script via --level unless already present in script args.",
    )
    parser.add_argument(
        "--log",
        default=str(Path("/tmp/ue_tooling_run_unreal.log")),
        help="Path to write combined stdout/stderr log.",
    )
    parser.add_argument(
        "--userdir",
        default="",
        help="Optional -userdir=... override (defaults to a timestamped folder under UE_Drone_Env/Saved).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Kill Unreal if it runs longer than this.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to the UE Python script.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()

    unreal_editor = Path(args.unreal_editor) if args.unreal_editor else (_default_unreal_editor_path() or Path(""))
    if not unreal_editor.exists():
        print("ERROR: UnrealEditor binary not found. Pass --unreal-editor or set UNREAL_EDITOR.", file=sys.stderr)
        return 2

    project = Path(args.project)
    if not project.exists():
        print(f"ERROR: Project not found: {project}", file=sys.stderr)
        return 2

    script = Path(args.script)
    if not script.exists():
        print(f"ERROR: Script not found: {script}", file=sys.stderr)
        return 2

    log_path = Path(args.log)

    if args.userdir:
        userdir = Path(args.userdir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        userdir = repo_root / "UE_Drone_Env/Saved/AutomationUserDir" / timestamp
    userdir = userdir.resolve()
    userdir.mkdir(parents=True, exist_ok=True)

    script_args = args.script_args or []
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    if "--level" not in script_args:
        script_args = ["--level", str(args.level)] + list(script_args)

    execute_value = " ".join([str(script)] + list(script_args))
    execute_arg = f"-ExecutePythonScript={execute_value}"

    cmd = [
        str(unreal_editor),
        str(project),
        "-unattended",
        "-nop4",
        "-nosplash",
        "-nullrhi",
        "-nosound",
        "-forcelogflush",
        "-stdout",
        "-FullStdOutLogOutput",
        "-AllowStdOutLogVerbosity",
        "-CrashForUAT",
        f"-userdir={userdir}",
        "-ScriptErrorsAreFatal",
        execute_arg,
    ]

    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        try:
            return proc.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                return proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                return 124


if __name__ == "__main__":
    raise SystemExit(main())
