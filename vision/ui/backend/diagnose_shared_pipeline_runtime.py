"""Compatibility wrapper for the relocated review diagnostics."""

from Vision.review_pipeline.diagnose_shared_pipeline_runtime import *  # noqa: F401,F403
from Vision.review_pipeline.diagnose_shared_pipeline_runtime import main


if __name__ == "__main__":
    main()
