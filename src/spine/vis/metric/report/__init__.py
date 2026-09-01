"""Batch reduction and rendering recipes for SPINE metric CSV files.

The package exposes the recipe interface, built-in metric implementations and
the :func:`build_report` orchestrator used by the standalone ``spine-report``
entry point. Importing it does not load Torch, LArCV or the SPINE driver.
"""

from .base import REPORT_SCHEMA_VERSION, ReportRecipe
from .cluster import ClusterSummaryRecipe
from .manager import build_report
from .node import NodeSummaryRecipe, quality_cut_mask
from .point import PointProposalRecipe
from .segment import SegmentConfusionRecipe

__all__ = [
    "ClusterSummaryRecipe",
    "NodeSummaryRecipe",
    "PointProposalRecipe",
    "REPORT_SCHEMA_VERSION",
    "ReportRecipe",
    "SegmentConfusionRecipe",
    "build_report",
    "quality_cut_mask",
]
