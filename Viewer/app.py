"""Local entrypoint for the flight log review UI."""

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def main() -> int:
    _ensure_src_on_path()
    from ui.server import main as ui_main

    return ui_main()


if __name__ == "__main__":
    raise SystemExit(main())
