"""
Compatibility shim for the relocated build launcher.

Canonical location:
- UE_Tooling/UE_Build/run_unreal_build_gen.py

This shim preserves legacy invocation paths during the build-layer refactor.
"""

from pathlib import Path
import runpy


def main() -> int:
    target = Path(__file__).resolve().parent / "UE_Build/run_unreal_build_gen.py"
    if not target.exists():
        raise RuntimeError(f"Missing relocated launcher: {target}")
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
