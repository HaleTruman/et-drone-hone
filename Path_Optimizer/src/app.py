"""Local entrypoint for the racing-stack run explorer."""

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def main() -> int:
    _ensure_src_on_path()
    from app.dash_app import run

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
