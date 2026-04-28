"""Canonical SPINE tensor column conventions.

This module defines the fixed column layouts used throughout SPINE for sparse
input tensors, cluster labels, coordinate labels, and PPN targets/predictions.
These are not physics categories; they are schema conventions for tensors.

Design notes
------------
- Single columns are represented as :class:`enum.IntEnum` members for readable
  access such as ``SparseTensorCol.X`` or ``ClusterLabelCol.PARTICLE``.
- Multi-column selectors are represented as plain tuples so they can be used
  directly with NumPy and PyTorch indexing without introducing a numerical
  dependency in the constants package.
- Backward-compatible aliases such as ``BATCH_COL`` and ``COORD_COLS`` are kept
  at module scope so existing code can migrate incrementally.
"""

from enum import IntEnum
from typing import Final

__all__ = [
    "SparseTensorCol",
    "ClusterLabelCol",
    "CoordLabelCol",
    "PPNLabelCol",
    "PPNPredCol",
    "COORD_COLS",
    "COORD_COLS_LO",
    "COORD_COLS_HI",
    "VTX_COLS",
    "VTX_COLS_LO",
    "VTX_COLS_HI",
    "COORD_START_COLS",
    "COORD_START_COLS_LO",
    "COORD_START_COLS_HI",
    "COORD_END_COLS",
    "COORD_END_COLS_LO",
    "COORD_END_COLS_HI",
    "PPN_ROFF_COLS",
    "PPN_ROFF_COLS_LO",
    "PPN_ROFF_COLS_HI",
    "PPN_RTYPE_COLS",
    "PPN_RTYPE_COLS_LO",
    "PPN_RTYPE_COLS_HI",
    "PPN_RPOS_COLS",
    "PPN_RPOS_COLS_LO",
    "PPN_RPOS_COLS_HI",
    "PPN_SCORE_COLS",
    "PPN_SCORE_COLS_LO",
    "PPN_SCORE_COLS_HI",
    "PPN_CLASS_COLS",
    "PPN_CLASS_COLS_LO",
    "PPN_CLASS_COLS_HI",
    "PPN_END_COLS",
    "PPN_END_COLS_LO",
    "PPN_END_COLS_HI",
    # Backward-compatible aliases
    "BATCH_COL",
    "VALUE_COL",
    "CLUST_COL",
    "PART_COL",
    "GROUP_COL",
    "ANCST_COL",
    "INTER_COL",
    "NU_COL",
    "PID_COL",
    "PRGRP_COL",
    "PRINT_COL",
    "MOM_COL",
    "SHAPE_COL",
    "COORD_TIME_COL",
    "PPN_LTYPE_COL",
    "PPN_LPART_COL",
    "PPN_LENDP_COL",
    "PPN_OCC_COL",
    "PPN_SHAPE_COL",
]


# Sparse tensor layout
# --------------------
# Canonical sparse tensors are expected to store:
# - one batch column
# - three spatial coordinate columns
# - one value/features column start
class SparseTensorCol(IntEnum):
    """Columns of the canonical sparse tensor layout."""

    BATCH = 0
    X = 1
    Y = 2
    Z = 3
    VALUE = 4


# Cluster label tensor layout
# ---------------------------
# This layout is used for voxel-wise truth/label tensors carrying fragment,
# particle, interaction, and ancestry information.
class ClusterLabelCol(IntEnum):
    """Columns of the canonical cluster label tensor layout."""

    BATCH = 0
    X = 1
    Y = 2
    Z = 3
    VALUE = 4
    CLUSTER = 5
    PARTICLE = 6
    GROUP = 7
    ANCESTOR = 8
    INTERACTION = 9
    NU = 10
    PID = 11
    GROUP_PRIMARY = 12
    INTER_PRIMARY = 13
    VTX_X = 14
    VTX_Y = 15
    VTX_Z = 16
    MOMENTUM = 17
    SHAPE = 18


# Coordinate label tensor layout
# ------------------------------
# These columns store start/end point targets and optional timing information.
class CoordLabelCol(IntEnum):
    """Columns of the canonical coordinate-label tensor layout."""

    BATCH = 0
    START_X = 1
    START_Y = 2
    START_Z = 3
    END_X = 4
    END_Y = 5
    END_Z = 6
    TIME = 7
    SHAPE = 8


# PPN label tensor layout
# -----------------------
# Columns used in the PPN supervision tensor.
class PPNLabelCol(IntEnum):
    """Columns of the canonical PPN label tensor layout."""

    BATCH = 0
    X = 1
    Y = 2
    Z = 3
    TYPE = 4
    PARTICLE = 5
    ENDPOINT = 6


# PPN prediction tensor layout
# ----------------------------
# Columns used in the canonical post-network PPN prediction tensor.
class PPNPredCol(IntEnum):
    """Columns of the canonical PPN prediction tensor layout."""

    RAW_DX = 0
    RAW_DY = 1
    RAW_DZ = 2
    RAW_TYPE_0 = 3
    RAW_TYPE_1 = 4
    RAW_TYPE_2 = 5
    RAW_TYPE_3 = 6
    RAW_TYPE_4 = 7
    RAW_POS_0 = 8
    RAW_POS_1 = 9
    SCORE_NEG = 4
    SCORE_POS = 5
    OCCUPANCY = 6
    CLASS_0 = 7
    CLASS_1 = 8
    CLASS_2 = 9
    CLASS_3 = 10
    CLASS_4 = 11
    SHAPE = 12
    END_NEG = 13
    END_POS = 14


# Common grouped selectors
# ------------------------
# These are convenience tuples for direct array slicing.
# For contiguous column groups, matching lower/upper bounds are also provided so
# Numba kernels can use simple slices instead of advanced indexing.
COORD_COLS: Final[tuple[int, int, int]] = (
    int(SparseTensorCol.X),
    int(SparseTensorCol.Y),
    int(SparseTensorCol.Z),
)
COORD_COLS_LO: Final[int] = COORD_COLS[0]
COORD_COLS_HI: Final[int] = COORD_COLS[-1] + 1
VTX_COLS: Final[tuple[int, int, int]] = (
    int(ClusterLabelCol.VTX_X),
    int(ClusterLabelCol.VTX_Y),
    int(ClusterLabelCol.VTX_Z),
)
VTX_COLS_LO: Final[int] = VTX_COLS[0]
VTX_COLS_HI: Final[int] = VTX_COLS[-1] + 1
COORD_START_COLS: Final[tuple[int, int, int]] = (
    int(CoordLabelCol.START_X),
    int(CoordLabelCol.START_Y),
    int(CoordLabelCol.START_Z),
)
COORD_START_COLS_LO: Final[int] = COORD_START_COLS[0]
COORD_START_COLS_HI: Final[int] = COORD_START_COLS[-1] + 1
COORD_END_COLS: Final[tuple[int, int, int]] = (
    int(CoordLabelCol.END_X),
    int(CoordLabelCol.END_Y),
    int(CoordLabelCol.END_Z),
)
COORD_END_COLS_LO: Final[int] = COORD_END_COLS[0]
COORD_END_COLS_HI: Final[int] = COORD_END_COLS[-1] + 1
PPN_ROFF_COLS: Final[tuple[int, int, int]] = (
    int(PPNPredCol.RAW_DX),
    int(PPNPredCol.RAW_DY),
    int(PPNPredCol.RAW_DZ),
)
PPN_ROFF_COLS_LO: Final[int] = PPN_ROFF_COLS[0]
PPN_ROFF_COLS_HI: Final[int] = PPN_ROFF_COLS[-1] + 1
PPN_RTYPE_COLS: Final[tuple[int, int, int, int, int]] = (
    int(PPNPredCol.RAW_TYPE_0),
    int(PPNPredCol.RAW_TYPE_1),
    int(PPNPredCol.RAW_TYPE_2),
    int(PPNPredCol.RAW_TYPE_3),
    int(PPNPredCol.RAW_TYPE_4),
)
PPN_RTYPE_COLS_LO: Final[int] = PPN_RTYPE_COLS[0]
PPN_RTYPE_COLS_HI: Final[int] = PPN_RTYPE_COLS[-1] + 1
PPN_RPOS_COLS: Final[tuple[int, int]] = (
    int(PPNPredCol.RAW_POS_0),
    int(PPNPredCol.RAW_POS_1),
)
PPN_RPOS_COLS_LO: Final[int] = PPN_RPOS_COLS[0]
PPN_RPOS_COLS_HI: Final[int] = PPN_RPOS_COLS[-1] + 1
PPN_SCORE_COLS: Final[tuple[int, int]] = (
    int(PPNPredCol.SCORE_NEG),
    int(PPNPredCol.SCORE_POS),
)
PPN_SCORE_COLS_LO: Final[int] = PPN_SCORE_COLS[0]
PPN_SCORE_COLS_HI: Final[int] = PPN_SCORE_COLS[-1] + 1
PPN_CLASS_COLS: Final[tuple[int, int, int, int, int]] = (
    int(PPNPredCol.CLASS_0),
    int(PPNPredCol.CLASS_1),
    int(PPNPredCol.CLASS_2),
    int(PPNPredCol.CLASS_3),
    int(PPNPredCol.CLASS_4),
)
PPN_CLASS_COLS_LO: Final[int] = PPN_CLASS_COLS[0]
PPN_CLASS_COLS_HI: Final[int] = PPN_CLASS_COLS[-1] + 1
PPN_END_COLS: Final[tuple[int, int]] = (
    int(PPNPredCol.END_NEG),
    int(PPNPredCol.END_POS),
)
PPN_END_COLS_LO: Final[int] = PPN_END_COLS[0]
PPN_END_COLS_HI: Final[int] = PPN_END_COLS[-1] + 1

# Backward-compatible aliases
# ---------------------------
# Keep the legacy names available while the rest of the codebase and external
# notebooks migrate to the new enum-centric API.
BATCH_COL = int(SparseTensorCol.BATCH)
VALUE_COL = int(SparseTensorCol.VALUE)

CLUST_COL = int(ClusterLabelCol.CLUSTER)
PART_COL = int(ClusterLabelCol.PARTICLE)
GROUP_COL = int(ClusterLabelCol.GROUP)
ANCST_COL = int(ClusterLabelCol.ANCESTOR)
INTER_COL = int(ClusterLabelCol.INTERACTION)
NU_COL = int(ClusterLabelCol.NU)
PID_COL = int(ClusterLabelCol.PID)
PRGRP_COL = int(ClusterLabelCol.GROUP_PRIMARY)
PRINT_COL = int(ClusterLabelCol.INTER_PRIMARY)
MOM_COL = int(ClusterLabelCol.MOMENTUM)

SHAPE_COL: Final[int] = -1

COORD_TIME_COL = int(CoordLabelCol.TIME)

PPN_LTYPE_COL = int(PPNLabelCol.TYPE)
PPN_LPART_COL = int(PPNLabelCol.PARTICLE)
PPN_LENDP_COL = int(PPNLabelCol.ENDPOINT)

PPN_OCC_COL = int(PPNPredCol.OCCUPANCY)
PPN_SHAPE_COL = int(PPNPredCol.SHAPE)
