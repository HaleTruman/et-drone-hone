"""Compare Flight deterministic vision backends for a recorded flight run."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Vision.review_pipeline.compare_flight_backends import main


if __name__ == "__main__":
    main()
