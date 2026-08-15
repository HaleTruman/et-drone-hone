"""Compatibility wrapper for the relocated review validator."""

from Vision.review_pipeline.validate_schema_review_dump import *  # noqa: F401,F403
from Vision.review_pipeline.validate_schema_review_dump import main


if __name__ == "__main__":
    main()
