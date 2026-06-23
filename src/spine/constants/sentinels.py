"""Sentinel values and invalid IDs used by SPINE.

These constants represent canonical invalid or missing values inherited from
LArCV conventions and from long-standing SPINE internal handling.
"""

__all__ = [
    "LARCV_INVALID_INDEX",
    "LARCV_INVALID_INSTANCE_ID",
    "LARCV_INVALID_UINT",
    "INVALID_PDG",
    "INVAL_IDX",
    "INVAL_ID",
    "INVAL_TID",
    "INVAL_PDG",
]


# Canonical invalid values
# ------------------------
LARCV_INVALID_INDEX = 65535
LARCV_INVALID_INSTANCE_ID = 9223372036854775807
LARCV_INVALID_UINT = 4294967295
INVALID_PDG = 0

# Backward-compatible aliases
# ---------------------------
INVAL_IDX = LARCV_INVALID_INDEX
INVAL_ID = LARCV_INVALID_INSTANCE_ID
INVAL_TID = LARCV_INVALID_UINT
INVAL_PDG = INVALID_PDG
