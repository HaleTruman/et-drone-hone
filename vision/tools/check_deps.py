#!/usr/bin/env python3
"""Verify required local dependencies for the 0721 decode and pose tools."""

from __future__ import annotations

import json


def main() -> None:
    import numpy as np

    cv = {"ok": True}
    try:
        import cv2

        cv["opencv"] = cv2.__version__
    except Exception as error:
        cv = {"ok": False, "error": str(error)}

    print(
        json.dumps(
            {
                "ok": True,
                "analysisMode": "fresh-layer-decode",
                "pythonDeps": {"numpy": np.__version__},
                "opencv": cv,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
