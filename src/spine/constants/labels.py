"""Human-readable labels and categorical mappings for SPINE enums.

This module contains presentation-oriented dictionaries and compatibility maps
derived from the canonical enums in :mod:`spine.constants.enums`.

The intent is to keep these mappings separate from the enum definitions:
- enums define the canonical integer codes
- this module defines how those codes are displayed or related to other codes
"""

from spine.constants.enums import (
    LArSoftNuInteractionType,
    NuCurrentType,
    ParticlePID,
    ParticleShape,
)

__all__ = [
    "SHAPE_PREC",
    "SHAPE_LABELS",
    "PID_LABELS",
    "PID_TAGS",
    "NU_CURR_TYPE",
    "NU_INT_TYPE",
    "PDG_TO_PID",
    "PID_TO_PDG",
    "SHP_TO_PID",
    "SHP_TO_PRIMARY",
]


# ParticleShape display and precedence helpers
# -----------------------------------
SHAPE_PREC = (
    int(ParticleShape.TRACK),
    int(ParticleShape.MICHEL),
    int(ParticleShape.SHOWER),
    int(ParticleShape.DELTA),
    int(ParticleShape.LOWE),
    int(ParticleShape.LARCV_UNKNOWN),
)

SHAPE_LABELS = {
    int(ParticleShape.UNKNOWN): "Unknown",
    int(ParticleShape.SHOWER): "Shower",
    int(ParticleShape.TRACK): "Track",
    int(ParticleShape.MICHEL): "Michel",
    int(ParticleShape.DELTA): "Delta",
    int(ParticleShape.LOWE): "LE",
    int(ParticleShape.GHOST): "Ghost",
    int(ParticleShape.LARCV_UNKNOWN): "Unknown",
}

# Particle-ID labels and short tags
# ---------------------------------
PID_LABELS = {
    int(ParticlePID.UNKNOWN): "Unknown",
    int(ParticlePID.PHOTON): "Photon",
    int(ParticlePID.ELECTRON): "Electron",
    int(ParticlePID.MUON): "Muon",
    int(ParticlePID.PION): "Pion",
    int(ParticlePID.PROTON): "Proton",
    int(ParticlePID.KAON): "Kaon",
}

PID_TAGS = {
    int(ParticlePID.UNKNOWN): "?",
    int(ParticlePID.PHOTON): "g",
    int(ParticlePID.ELECTRON): "e",
    int(ParticlePID.MUON): "mu",
    int(ParticlePID.PION): "pi",
    int(ParticlePID.PROTON): "p",
    int(ParticlePID.KAON): "ka",
}

# Neutrino interaction labels
# ---------------------------
NU_CURR_TYPE = {
    int(NuCurrentType.UNKNOWN): "UnknownCurrent",
    int(NuCurrentType.CC): "CC",
    int(NuCurrentType.NC): "NC",
}

NU_INT_TYPE = {int(member): member.name for member in LArSoftNuInteractionType}
NU_INT_TYPE[int(LArSoftNuInteractionType.UNKNOWN)] = "UnknownInteraction"

# Crosswalks between PDG codes and SPINE particle IDs
# ---------------------------------------------------
PDG_TO_PID = {
    22: int(ParticlePID.PHOTON),
    11: int(ParticlePID.ELECTRON),
    -11: int(ParticlePID.ELECTRON),
    13: int(ParticlePID.MUON),
    -13: int(ParticlePID.MUON),
    211: int(ParticlePID.PION),
    -211: int(ParticlePID.PION),
    2212: int(ParticlePID.PROTON),
    321: int(ParticlePID.KAON),
    -321: int(ParticlePID.KAON),
}

PID_TO_PDG = {value: abs(key) for key, value in PDG_TO_PID.items()}
PID_TO_PDG[int(ParticlePID.UNKNOWN)] = -1

# ParticleShape-to-allowed-category helpers
# ---------------------------------
SHP_TO_PID = {
    int(ParticleShape.SHOWER): [int(ParticlePID.PHOTON), int(ParticlePID.ELECTRON)],
    int(ParticleShape.TRACK): [
        int(ParticlePID.MUON),
        int(ParticlePID.PION),
        int(ParticlePID.PROTON),
        int(ParticlePID.KAON),
    ],
    int(ParticleShape.DELTA): [int(ParticlePID.ELECTRON)],
    int(ParticleShape.MICHEL): [int(ParticlePID.ELECTRON)],
}

SHP_TO_PRIMARY = {
    int(ParticleShape.SHOWER): (0, 1),
    int(ParticleShape.TRACK): (0, 1),
    int(ParticleShape.DELTA): (0,),
    int(ParticleShape.MICHEL): (0,),
}
