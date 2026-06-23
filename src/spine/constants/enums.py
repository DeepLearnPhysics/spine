"""Canonical SPINE categorical enumerations.

This module defines enumerated integer codes used for semantic categories and
physics/reconstruction labels. Unlike :mod:`spine.constants.columns`, which
describes tensor schema positions, this module only contains categories whose
values are part of the semantic content of SPINE objects.

Typical uses include:
- semantic shape categories such as shower/track/Michel/delta
- canonical SPINE particle-ID categories
- neutrino current and interaction categories

These enums are intended to be the source of truth for:
- field metadata validation in ``spine.data``
- human-readable label maps in :mod:`spine.constants.labels`
"""

from enum import IntEnum

__all__ = [
    "ParticleShape",
    "ParticlePID",
    "NuCurrentType",
    "NuInteractionScheme",
    "LArSoftNuInteractionType",
    "GenieNuInteractionType",
    # Backward-compatible aliases
    "SHOWR_SHP",
    "TRACK_SHP",
    "MICHL_SHP",
    "DELTA_SHP",
    "LOWES_SHP",
    "GHOST_SHP",
    "UNKWN_SHP",
    "PHOT_PID",
    "ELEC_PID",
    "MUON_PID",
    "PION_PID",
    "PROT_PID",
    "KAON_PID",
]


# Semantic shape categories
# -------------------------
class ParticleShape(IntEnum):
    """Enumerates semantic voxel/particle categories."""

    UNKNOWN = -1
    SHOWER = 0
    TRACK = 1
    MICHEL = 2
    DELTA = 3
    LOWE = 4
    GHOST = 5
    LARCV_UNKNOWN = 6


# Canonical SPINE particle species IDs
# ------------------------------------
class ParticlePID(IntEnum):
    """Enumerates canonical SPINE particle species IDs."""

    UNKNOWN = -1
    PHOTON = 0
    ELECTRON = 1
    MUON = 2
    PION = 3
    PROTON = 4
    KAON = 5


# Neutrino current categories
# ---------------------------
class NuCurrentType(IntEnum):
    """Enumerates neutrino current types."""

    UNKNOWN = -1
    CC = 0
    NC = 1


# Neutrino interaction-code conventions
# -------------------------------------
class NuInteractionScheme(IntEnum):
    """Enumerates source conventions for neutrino interaction code fields."""

    UNKNOWN = -1
    LARSOFT = 0
    GENIE = 1


# LArSoft/SBN neutrino interaction categories
# -------------------------------------------
# Source:
# nusimdata/SimulationBase/MCNeutrino.h :: simb::int_type_
# https://code-doc.larsoft.org/docs/latest/html/MCNeutrino_8h.html
class LArSoftNuInteractionType(IntEnum):
    """Enumerates LArSoft neutrino interaction categories."""

    UNKNOWN = -1
    QE = 0
    RES = 1
    DIS = 2
    COH = 3
    COHELASTIC = 4
    ELECTRONSCATTERING = 5
    IMDANNIHILATION = 6
    INVERSEBETADECAY = 7
    GLASHOWRESONANCE = 8
    AMNUGAMMA = 9
    MEC = 10
    DIFFRACTIVE = 11
    EM = 12
    WEAKMIX = 13
    NUANCEOFFSET = 1000
    CCQE = 1001
    NCQE = 1002
    RESCCNUPROTONPIPLUS = 1003
    RESCCNUNEUTRONPI0 = 1004
    RESCCNUNEUTRONPIPLUS = 1005
    RESNCNUPROTONPI0 = 1006
    RESNCNUPROTONPIPLUS = 1007
    RESNCNUNEUTRONPI0 = 1008
    RESNCNUNEUTRONPIMINUS = 1009
    RESCCNUBARNEUTRONPIMINUS = 1010
    RESCCNUBARPROTONPI0 = 1011
    RESCCNUBARPROTONPIMINUS = 1012
    RESNCNUBARPROTONPI0 = 1013
    RESNCNUBARPROTONPIPLUS = 1014
    RESNCNUBARNEUTRONPI0 = 1015
    RESNCNUBARNEUTRONPIMINUS = 1016
    RESCCNUDELTAPLUSPIPLUS = 1017
    RESCCNUDELTA2PLUSPIMINUS = 1021
    RESCCNUBARDELTA0PIMINUS = 1028
    RESCCNUBARDELTAMINUSPIPLUS = 1032
    RESCCNUPROTONRHOPLUS = 1039
    RESCCNUNEUTRONRHOPLUS = 1041
    RESCCNUBARNEUTRONRHOMINUS = 1046
    RESCCNUBARNEUTRONRHO0 = 1048
    RESCCNUSIGMAPLUSKAONPLUS = 1053
    RESCCNUSIGMAPLUSKAON0 = 1055
    RESCCNUBARSIGMAMINUSKAON0 = 1060
    RESCCNUBARSIGMA0KAON0 = 1062
    RESCCNUPROTONETA = 1067
    RESCCNUBARNEUTRONETA = 1070
    RESCCNUKAONPLUSLAMBDA0 = 1073
    RESCCNUBARKAON0LAMBDA0 = 1076
    RESCCNUPROTONPIPLUSPIMINUS = 1079
    RESCCNUPROTONPI0PI0 = 1080
    RESCCNUBARNEUTRONPIPLUSPIMINUS = 1085
    RESCCNUBARNEUTRONPI0PI0 = 1086
    RESCCNUBARPROTONPI0PI0 = 1090
    CCDIS = 1091
    NCDIS = 1092
    UNUSED1 = 1093
    UNUSED2 = 1094
    CCQEHYPERON = 1095
    NCCOH = 1096
    CCCOH = 1097
    NUELECTRONELASTIC = 1098
    INVERSEMUDECAY = 1099
    MEC2P2H = 1100


# GENIE/DUNE neutrino interaction categories
# ------------------------------------------
# Source:
# GENIE Framework/Interaction/ScatteringType.h :: genie::EScatteringType
# https://internal.dunescience.org/doxygen/classgenie_1_1ScatteringType.html
class GenieNuInteractionType(IntEnum):
    """Enumerates GENIE scattering categories."""

    UNKNOWN = -100
    NULL = 0
    QE = 1
    SINGLEKAON = 2
    DIS = 3
    RES = 4
    COH = 5
    DIFFRACTIVE = 6
    NUELECTRONELASTIC = 7
    INVERSEMUDECAY = 8
    AMNUGAMMA = 9
    MEC = 10
    COHELASTIC = 11
    INVERSEBETADECAY = 12
    GLASHOWRESONANCE = 13
    IMDANNIHILATION = 14
    PHOTONCOHERENT = 15
    PHOTONRESONANCE = 16
    SINGLEPION = 17
    DARKMATTERELASTIC = 101
    DARKMATTERDIS = 102
    DARKMATTERELECTRON = 103
    NORM = 104


# Temporary compatibility alias
# -----------------------------
# Until source-specific dispatch is added throughout the neutrino data flow,
# keep the historical symbol bound to the LArSoft/SBN convention.
NuInteractionType = LArSoftNuInteractionType


# Backward-compatible aliases
# ---------------------------
# These names mirror the long-standing globals API and make migration less
# abrupt while the rest of the code moves to explicit enum member access.
SHOWR_SHP = int(ParticleShape.SHOWER)
TRACK_SHP = int(ParticleShape.TRACK)
MICHL_SHP = int(ParticleShape.MICHEL)
DELTA_SHP = int(ParticleShape.DELTA)
LOWES_SHP = int(ParticleShape.LOWE)
GHOST_SHP = int(ParticleShape.GHOST)
UNKWN_SHP = int(ParticleShape.LARCV_UNKNOWN)

PHOT_PID = int(ParticlePID.PHOTON)
ELEC_PID = int(ParticlePID.ELECTRON)
MUON_PID = int(ParticlePID.MUON)
PION_PID = int(ParticlePID.PION)
PROT_PID = int(ParticlePID.PROTON)
KAON_PID = int(ParticlePID.KAON)
