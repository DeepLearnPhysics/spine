"""Module which contains enumerated variables shared across the project."""

from enum import IntEnum

from .globals import *

__all__ = ["enum_factory"]


def enum_factory(enum, value):
    """Parses an enumerated object from string name(s) to value(s).

    Parameters
    ----------
    enum : str
        Name of the enumerated type
    value : Union[str, List[str]]
        Name or names of the enumerated objects (from config)

    Returns
    -------
    Union[int, List[int]]
        Value or values of the enumerated objects
    """
    # Get the enumerated type
    ENUM_DICT = {"cluster": ClusterLabelEnum, "shape": ShapeEnum, "pid": PIDEnum}
    assert enum in ENUM_DICT, (
        "Enumerated type not recognized: {enum}. Must be one of "
        "{list(ENUM_DICT.keys())}."
    )
    enum = ENUM_DICT[enum]

    # Translate enumerated strings into values
    if isinstance(value, str):
        if not hasattr(enum, value.upper()):
            raise ValueError(
                f"Enumerated object not recognized: {value}. Must be one "
                f"of {[e.name for e in enum]}."
            )

        return getattr(enum, value.upper()).value

    else:
        values = []
        for v in value:
            if not hasattr(enum, v.upper()):
                raise ValueError(
                    f"Enumerated object not recognized: {v}. Must be one "
                    f"of {[e.name for e in enum]}."
                )
            values.append(getattr(enum, v.upper()).value)

        return values


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
