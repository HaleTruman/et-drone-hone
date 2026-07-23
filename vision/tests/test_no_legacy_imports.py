from __future__ import annotations

import unittest
from pathlib import Path


VISION_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".py", ".txt"}
OLD_IDENTITY_TOKENS = (
    "second" + "gen",
    "second" + "_gen",
    "second-" + "generation",
    "second " + "generation",
)


class IdentityCleanupTests(unittest.TestCase):
    def test_vision_files_do_not_contain_old_identity_terms(self) -> None:
        offenders: list[str] = []
        for path in VISION_ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if "__pycache__" in path.relative_to(VISION_ROOT).parts:
                continue
            text = path.read_text(encoding="utf-8").lower()
            for token in OLD_IDENTITY_TOKENS:
                if token in text:
                    offenders.append(str(path.relative_to(VISION_ROOT)))
                    break
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
