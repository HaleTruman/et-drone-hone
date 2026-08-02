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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unreal-editor",
        default=os.environ.get("UNREAL_EDITOR") or "",
        help="Path to UnrealEditor binary (or set UNREAL_EDITOR).",
    )
    parser.add_argument(
        "--project",
        default=str(Path("UE_Drone_Env_1/UE_Drone_Env_1.uproject").resolve()),
        help="Path to .uproject",
    )
    parser.add_argument(
        "--script",
        default=str(Path("UE_Drone_Env_1/Scripts/create_red_sphere_course.py").resolve()),
        help="Path to UE Python script to run via -ExecutePythonScript",
    )
    parser.add_argument(
        "--log",
        default=str(Path("/tmp/ue_automation.log")),
        help="Path to write combined stdout/stderr log.",
    )
    parser.add_argument(
        "--userdir",
        default="",
        help="Optional -userdir=... override (defaults to a timestamped folder under UE_Drone_Env_1/Saved).",
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
        help="Arguments after -- are passed to the UE Python script (e.g. -- --diameter-cm 100 --color 1,0,0).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

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
        userdir = Path("UE_Drone_Env_1/Saved/AutomationUserDir") / timestamp
    userdir = userdir.resolve()
    userdir.mkdir(parents=True, exist_ok=True)

    script_args = args.script_args or []
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    # On macOS, Unreal reconstructs the stored command line from argv, and if an argv element contains
    # spaces and an '=', it will reformat it as ArgName="ArgValue" (quoting only the value). So for
    # -ExecutePythonScript to receive script args, we must *not* embed quotes ourselves here; we pass a
    # single argv element with spaces in the value, and let the engine quote the value portion.
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
