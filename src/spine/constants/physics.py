"""Physical constants used by SPINE.

This module contains campaign-independent constants that define canonical
physics assumptions used in reconstruction utilities. Anything that can vary by
detector conditions, calibration campaign, or systematic choice should live in
configuration rather than here.
"""

from .enums import ParticlePID

__all__ = [
    "PHOT_MASS_MEV_C2",
    "ELEC_MASS_MEV_C2",
    "MUON_MASS_MEV_C2",
    "PION_MASS_MEV_C2",
    "PROT_MASS_MEV_C2",
    "KAON_MASS_MEV_C2",
    "PID_MASSES",
    "LAR_DENSITY_G_CM3",
    "LAR_Z",
    "LAR_A_G_MOL",
    "LAR_MEAN_EXCITATION_ENERGY_MEV",
    "LAR_RADIATION_LENGTH_CM",
    "LAR_WION_MEV",
    "LAR_CRITICAL_ENERGY_MEV",
    "LAR_MOLIERE_RADIUS_CM",
    "LAR_A",
    "LAR_MEE",
    "LAR_X0",
    "LAR_WION",
    "LAR_E_CRIT",
    "LAR_RM",
    "LAR_DE_A",
    "LAR_DE_K",
    "LAR_DE_X0",
    "LAR_DE_X1",
    "LAR_DE_CBAR",
    "LAR_DE_DELTA0",
    "PHOT_MASS",
    "ELEC_MASS",
    "MUON_MASS",
    "PION_MASS",
    "PROT_MASS",
    "KAON_MASS",
    "LAR_DENSITY",
]


# Rest masses
# -----------
PHOT_MASS_MEV_C2 = 0.0
ELEC_MASS_MEV_C2 = 0.511998
MUON_MASS_MEV_C2 = 105.658
PION_MASS_MEV_C2 = 139.570
PROT_MASS_MEV_C2 = 938.272
KAON_MASS_MEV_C2 = 493.677

PID_MASSES = {
    int(ParticlePID.PHOTON): PHOT_MASS_MEV_C2,
    int(ParticlePID.ELECTRON): ELEC_MASS_MEV_C2,
    int(ParticlePID.MUON): MUON_MASS_MEV_C2,
    int(ParticlePID.PION): PION_MASS_MEV_C2,
    int(ParticlePID.PROTON): PROT_MASS_MEV_C2,
    int(ParticlePID.KAON): KAON_MASS_MEV_C2,
}

# Liquid argon material constants
# -------------------------------
LAR_DENSITY_G_CM3 = 1.396
LAR_Z = 18
LAR_A_G_MOL = 39.9481
LAR_MEAN_EXCITATION_ENERGY_MEV = 188.0e-6
LAR_RADIATION_LENGTH_CM = 14.0
LAR_WION_MEV = 23.6e-6
LAR_CRITICAL_ENERGY_MEV = 32.84
LAR_MOLIERE_RADIUS_CM = 9.043

# Sternheimer density-effect parametrization for liquid argon
# -----------------------------------------------------------
LAR_DE_A = 0.19559
LAR_DE_K = 3.0000
LAR_DE_X0 = 0.2000
LAR_DE_X1 = 3.0000
LAR_DE_CBAR = 5.2146
LAR_DE_DELTA0 = 0.00

# Backward-compatible aliases
# ---------------------------
PHOT_MASS = PHOT_MASS_MEV_C2
ELEC_MASS = ELEC_MASS_MEV_C2
MUON_MASS = MUON_MASS_MEV_C2
PION_MASS = PION_MASS_MEV_C2
PROT_MASS = PROT_MASS_MEV_C2
KAON_MASS = KAON_MASS_MEV_C2

LAR_DENSITY = LAR_DENSITY_G_CM3
LAR_A = LAR_A_G_MOL
LAR_MEE = LAR_MEAN_EXCITATION_ENERGY_MEV
LAR_X0 = LAR_RADIATION_LENGTH_CM
LAR_WION = LAR_WION_MEV
LAR_E_CRIT = LAR_CRITICAL_ENERGY_MEV
LAR_RM = LAR_MOLIERE_RADIUS_CM
