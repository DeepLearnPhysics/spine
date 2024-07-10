"""Module which contains enumerated variables shared across the project."""

from enum import IntEnum

from .globals import *

__all__ = ['ClusterLabelEnum']


class ClusterLabelEnum(IntEnum):
    """Enumerates all possible columns of the cluster label tensor."""
    CLUSTER = CLUST_COL
    PARTICLE = PART_COL
    GROUP = GROUP_COL
    INTERACTION = INTER_COL
    NU = NU_COL
    PID = PID_COL
    GROUP_PRIMARY = PRGRP_COL
    INTER_PRIMARY = PRINT_COL
    MOMENTUM = MOM_COL


class ShapeEnum(IntEnum):
    """Enumerates all possible shape values."""
    SHOWER = SHOWR_SHP
    TRACK = TRACK_SHP
    MICHEL = MICHL_SHP
    DELTA = DELTA_SHP
    LOWE = LOWES_SHP
    GHOST = GHOST_SHP


class PIDEnum(IntEnum):
    """Enumerates all possible particle species values."""
    PHOTON = PHOT_PID
    ELECTRON = ELEC_PID
    MUON = MUON_PID
    PION = PION_PID
    PROTON = PROT_PID
    KAON = KAON_PID
