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
    "GiBUUNuInteractionType",
    "NuWroNuInteractionType",
    "NeutNuInteractionType",
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
    GIBUU = 2
    NUWRO = 3
    NEUT = 4


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


# GiBUU neutrino interaction categories
# --------------------------------------
# Source: GiBUU's NuHepMC process table (NuHepMC.ProcessInfo metadata).
class GiBUUNuInteractionType(IntEnum):
    """Enumerates GiBUU NuHepMC process identifiers."""

    UNKNOWN = -1
    UNRECOGNIZED = 0
    CCQE = 200
    NCQE = 250
    CC_2P2H_QE = 300
    CC_2P2H_DELTA = 301
    NC_2P2H_QE = 350
    NC_2P2H_DELTA = 351
    CC_RES_DELTA = 400
    CC_RES_OTHER = 401
    NC_RES_DELTA = 450
    NC_RES_OTHER = 451
    CC_BKGD_NEUTRON = 500
    CC_BKGD_PROTON = 501
    CC_BKGD_2PI = 502
    NC_BKGD_NEUTRON = 550
    NC_BKGD_PROTON = 551
    NC_BKGD_2PI = 552
    CC_DIS = 600
    NC_DIS = 650


# NuWro neutrino interaction categories
# --------------------------------------
# Source: NuWro's nuwro2rootracker GetNeutChannel conversion. These are the
# simplified NEUT-style channel identifiers written to EvtCode.
class NuWroNuInteractionType(IntEnum):
    """Enumerates NuWro RooTracker interaction channel identifiers."""

    UNKNOWN = -1
    CCQE = 1
    CCMEC = 2
    CCRES = 11
    CCCOH = 16
    CCDIS = 26
    NCRES = 31
    NCCOH = 36
    NCDIS = 46
    NCQE = 51
    CCHYPERON = 100


# NEUT neutrino interaction categories
# -------------------------------------
# Source: neutvect-converter's NuHepMC.ProcessInfo table. Positive NuHepMC
# process IDs are used instead of signed NEUT mode values so that NEUT's -1
# CCQE antineutrino mode cannot collide with SPINE's missing-value sentinel.
class NeutNuInteractionType(IntEnum):
    """Enumerates NEUT NuHepMC process identifiers."""

    UNKNOWN = -1
    CC_COH_NU = 100
    CC_DIF_NU = 110
    CC_COH_NUBAR = 125
    CC_DIF_NUBAR = 135
    NC_COH_NU = 150
    NC_DIF_NU = 160
    NC_COH_NUBAR = 175
    NC_DIF_NUBAR = 185
    CC_QE_NU = 200
    CC_QE_PROTON_NUBAR = 225
    NC_ELASTIC_PROTON_NU = 250
    NC_ELASTIC_NEUTRON_NU = 251
    NC_ELASTIC_PROTON_NUBAR = 275
    NC_ELASTIC_NEUTRON_NUBAR = 276
    CC_2P2H_NU = 300
    CC_2P2H_NUBAR = 325
    CC_RES_PROTON_PI_PLUS_NU = 400
    CC_RES_PROTON_PI_ZERO_NU = 401
    CC_RES_NEUTRON_PI_PLUS_NU = 402
    CC_ETA_NU = 410
    CC_KAON_NU = 411
    CC_SINGLE_GAMMA_NU = 412
    CC_RES_NEUTRON_PI_MINUS_NUBAR = 425
    CC_RES_PROTON_PI_ZERO_NUBAR = 426
    CC_RES_PROTON_PI_MINUS_NUBAR = 427
    CC_ETA_NUBAR = 435
    CC_KAON_NUBAR = 436
    CC_SINGLE_GAMMA_NUBAR = 437
    NC_RES_NEUTRON_PI_ZERO_NU = 450
    NC_RES_PROTON_PI_ZERO_NU = 451
    NC_RES_NEUTRON_PI_PLUS_NU = 452
    NC_ETA_NEUTRON_NU = 460
    NC_ETA_PROTON_NU = 461
    NC_KAON_NEUTRON_NU = 462
    NC_KAON_PROTON_NU = 463
    NC_SINGLE_GAMMA_NEUTRON_NU = 464
    NC_SINGLE_GAMMA_PROTON_NU = 465
    NC_RES_NEUTRON_PI_ZERO_NUBAR = 475
    NC_RES_PROTON_PI_ZERO_NUBAR = 476
    NC_RES_PROTON_PI_MINUS_NUBAR = 478
    NC_RES_NEUTRON_PI_PLUS_NUBAR = 479
    NC_ETA_NEUTRON_NUBAR = 485
    NC_ETA_PROTON_NUBAR = 486
    NC_KAON_NEUTRON_NUBAR = 487
    NC_KAON_PROTON_NUBAR = 488
    NC_SINGLE_GAMMA_NEUTRON_NUBAR = 489
    NC_SINGLE_GAMMA_PROTON_NUBAR = 490
    CC_MULTI_PI_NU = 500
    CC_MULTI_PI_NUBAR = 525
    NC_MULTI_PI_NU = 550
    NC_MULTI_PI_NUBAR = 575
    CC_DIS_NU = 600
    NC_DIS_NU = 601
    CC_DIS_NUBAR = 625
    NC_DIS_NUBAR = 675


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
