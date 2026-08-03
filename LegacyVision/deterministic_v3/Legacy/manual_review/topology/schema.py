"""Schema bridge used by the byte-for-byte production topology snapshot.

The local :mod:`topology` file deliberately retains its original ``.schema``
import.  Re-exporting only those contracts here keeps the snapshot executable
without copying or redefining production data classes.
"""

from ...schema import (
    CLIPPED_TOPOLOGY,
    C_SHAPE_ROUTE,
    C_SHAPE_TOPOLOGY,
    MULTI_GATE_ROUTE,
    MULTI_VOID_TOPOLOGY,
    STANDARD_ROUTE,
    STANDARD_TOPOLOGY,
    UNKNOWN_TOPOLOGY,
    ApertureCenterEvidence,
    ComponentObservation,
    ContourNodeEvidence,
    FrameObservation,
    TopologyDecision,
)

__all__ = (
    "CLIPPED_TOPOLOGY",
    "C_SHAPE_ROUTE",
    "C_SHAPE_TOPOLOGY",
    "MULTI_GATE_ROUTE",
    "MULTI_VOID_TOPOLOGY",
    "STANDARD_ROUTE",
    "STANDARD_TOPOLOGY",
    "UNKNOWN_TOPOLOGY",
    "ApertureCenterEvidence",
    "ComponentObservation",
    "ContourNodeEvidence",
    "FrameObservation",
    "TopologyDecision",
)

