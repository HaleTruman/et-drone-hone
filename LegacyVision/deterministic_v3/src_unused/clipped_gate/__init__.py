"""Offline clipped-mask discovery helpers.

This package is intentionally not registered with the deterministic-v3 runtime
pipeline.  Its first public surface consumes already-classified historic review
records only.
"""

from .process import PROCESS_VERSION, process_clipped_mask

__all__ = ("PROCESS_VERSION", "process_clipped_mask")
