"""Compatibility wrapper for the relocated review pipeline."""

from LegacyVision.review_pipeline.replay_historic_run import *  # noqa: F401,F403
from LegacyVision.review_pipeline.replay_historic_run import main


if __name__ == "__main__":
    main()
