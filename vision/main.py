"""CLI for the projection production runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline import options_from_args, run_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the projection production vision pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Process one frame or a JPEG directory.")
    run.add_argument("--source-dir", type=Path)
    run.add_argument("--single-frame", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--run-id")
    run.add_argument("--debug", action="store_true")
    run.add_argument("--max-frames", type=int)
    run.add_argument("--preset-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_pipeline(options_from_args(args))
        print(manifest.run_root)
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
