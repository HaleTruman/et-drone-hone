"""Start the LegacyVision deterministic-v3 review UI."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LegacyVision.ui.backend.serve_review_ui import main


if __name__ == "__main__":
    main()
