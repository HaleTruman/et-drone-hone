"""Generate LegacyVision review artifacts for a flight run."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Vision.review_pipeline.replay_historic_run import main


if __name__ == "__main__":
    main()
