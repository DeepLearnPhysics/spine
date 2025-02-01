"""Module which contains all global variables shared across the project."""

import numpy as np
from collections import defaultdict

# Column which specifies the batch ID in a sparse tensor
BATCH_COL = 0

# Columns which specify the voxel coordinates in a sparse tensor
COORD_COLS = np.array([1, 2, 3])

# Colum which specifies the first value of a voxel in a sparse tensor
VALUE_COL = 4

# Columns that specify each attribute in a cluster label tensor
CLUST_COL = 5                      # Fragment ID
PART_COL  = 6                      # Particle ID
GROUP_COL = 7                      # Group ID
INTER_COL = 8                      # Interaction ID
NU_COL    = 9                      # Neutrino ID
PID_COL   = 10                     # Particle species
PRGRP_COL = 11                     # Group primary flag
PRINT_COL = 12                     # Interaction primary flag
VTX_COLS  = np.array([13, 14, 15]) # Interaction vertex location
MOM_COL   = 16                     # Absolute momentum in GeV

# Column which specifies the shape ID of a voxel in a label tensor
SHAPE_COL = -1

# Columns that specify each value type in the coordinate label tensor
COORD_START_COLS = COORD_COLS
COORD_END_COLS   = np.array([4, 5, 6])
COORD_TIME_COL   = 7

# Columns that specify each value type in a PPN label/output tensors
PPN_LTYPE_COL  = 4                           # Class type label
PPN_LPART_COL  = 5                           # Particle ID label
PPN_LENDP_COL  = 6                           # Endpoint label

PPN_ROFF_COLS  = np.array([0, 1, 2])         # Raw offset
PPN_RTYPE_COLS = np.array([3, 4, 5, 6, 7])   # Raw class type scores
PPN_RPOS_COLS  = np.array([8, 9])            # Raw positive score

PPN_SCORE_COLS = np.array([4, 5])            # Softmax positive scores
PPN_OCC_COL    = 6                           # Occupancy score
PPN_CLASS_COLS = np.array([7, 8, 9, 10, 11]) # Softmax class scores
PPN_SHAPE_COL  = 12                          # Predicted shape
PPN_END_COLS   = np.array([13, 14])          # Softmax end point scores

# Shape ID of each type of voxel category
SHOWR_SHP = 0 # larcv.kShapeShower
TRACK_SHP = 1 # larcv.kShapeTrack
MICHL_SHP = 2 # larcv.kShapeMichel
DELTA_SHP = 3 # larcv.kShapeDelta
LOWES_SHP = 4 # larcv.kShapeLEScatter
GHOST_SHP = 5 # larcv.kShapeGhost
UNKWN_SHP = 6 # larcv.kShapeUnknown

# Shape precedence used in the cluster labeling process
SHAPE_PREC = (TRACK_SHP, MICHL_SHP, SHOWR_SHP, DELTA_SHP, LOWES_SHP, UNKWN_SHP)

# Shape labels
SHAPE_LABELS = {
   -1: 'Unknown',
   SHOWR_SHP: 'Shower',
   TRACK_SHP: 'Track',
   MICHL_SHP: 'Michel',
   DELTA_SHP: 'Delta',
   LOWES_SHP: 'LE',
   GHOST_SHP: 'Ghost',
}

# Invalid larcv.Particle labels
INVAL_IDX = 65535               # larcv.kINVALID_INDEX, or INVAL_ID before LArCV2 v2.2.0
INVAL_ID  = 9223372036854775807 # larcv.kINVALID_INSTANCEID
INVAL_TID = 4294967295          # larcv.kINVALID_UINT
INVAL_PDG = 0                   # Invalid particle PDG code

# Particle ID of each recognized particle species
PHOT_PID = 0
ELEC_PID = 1
MUON_PID = 2
PION_PID = 3
PROT_PID = 4
KAON_PID = 5

# Mapping between particle PDG code and particle ID labels
PHOT_PID = 0
PDG_TO_PID = defaultdict(lambda: -1)
PDG_TO_PID.update({
    22:   PHOT_PID,
    11:   ELEC_PID,
    -11:  ELEC_PID,
    13:   MUON_PID,
    -13:  MUON_PID,
    211:  PION_PID,
    -211: PION_PID,
    2212: PROT_PID,
    321:  KAON_PID,
    -321: KAON_PID
})

PID_TO_PDG = {v : abs(k) for k, v in PDG_TO_PID.items()}
PID_TO_PDG[-1] = -1

# Particle type labels
PID_LABELS = {
    -1: 'Unknown',
    PHOT_PID: 'Photon',
    ELEC_PID: 'Electron',
    MUON_PID: 'Muon',
    PION_PID: 'Pion',
    PROT_PID: 'Proton',
    KAON_PID: 'Kaon'
}

# Particle type tags
PID_TAGS = {
    -1: '?',
    PHOT_PID: 'g',
    ELEC_PID: 'e',
    MUON_PID: 'mu',
    PION_PID: 'pi',
    PROT_PID: 'p',
    KAON_PID: 'ka'
}

# Map between shape and allowed PID/primary labels
SHP_TO_PID = {
    SHOWR_SHP: np.array([PHOT_PID, ELEC_PID]),
    TRACK_SHP: np.array([MUON_PID, PION_PID, PROT_PID, KAON_PID]),
    DELTA_SHP: np.array([ELEC_PID]),
    MICHL_SHP: np.array([ELEC_PID])
}

SHP_TO_PRIMARY = {
    SHOWR_SHP: np.array([0, 1]),
    TRACK_SHP: np.array([0, 1]),
    DELTA_SHP: np.array([0]),
    MICHL_SHP: np.array([0])
}

# Particle masses
PHOT_MASS = 0.       # [MeV/c^2]
ELEC_MASS = 0.511998 # [MeV/c^2]
MUON_MASS = 105.658  # [MeV/c^2]
PION_MASS = 139.570  # [MeV/c^2]
PROT_MASS = 938.272  # [MeV/c^2]
KAON_MASS = 483.677  # [MeV/c^2]

PID_MASSES = {
    PHOT_PID: PHOT_MASS,
    ELEC_PID: ELEC_MASS,
    MUON_PID: MUON_MASS,
    PION_PID: PION_MASS,
    PROT_PID: PROT_MASS,
    KAON_PID: KAON_MASS
}

# Neutrino current type
NU_CURR_TYPE = {
    -1: 'UnknownCurrent',
    0:  'CC',
    1:  'NC'
}

# Neutrino interaction mode and type labels
# Source: https://internal.dunescience.org/doxygen/MCNeutrino_8h_source.html
NU_INT_TYPE = {
    -1:   'UnknownInteraction',
    1:    'QE',
    2:    'DIS',
    3:    'Coh',
    4:    'CohElastic',
    5:    'ElectronScattering',
    6:    'IMDAnnihilation',
    7:    'InverseBetaDecay',
    8:    'GlashowResonance',
    9:    'AMNuGamma',
    10:   'MEC',
    11:   'Diffractive',
    12:   'EM',
    13:   'WeakMix',
    1000: 'NuanceOffset',
    1001: 'CCQE',
    1002: 'NCQE',
    1003: 'ResCCNuProtonPiPlus',
    1004: 'ResCCNuNeutronPi0',
    1005: 'ResCCNuNeutronPiPlus',
    1006: 'ResNCNuProtonPi0',
    1007: 'ResNCNuProtonPiPlus',
    1008: 'ResNCNuNeutronPi0',
    1009: 'ResNCNuNeutronPiMinus',
    1010: 'ResCCNuBarNeutronPiMinus',
    1011: 'ResCCNuBarProtonPi0',
    1012: 'ResCCNuBarProtonPiMinus',
    1013: 'ResNCNuBarProtonPi0',
    1014: 'ResNCNuBarProtonPiPlus',
    1015: 'ResNCNuBarNeutronPi0',
    1016: 'ResNCNuBarNeutronPiMinus',
    1017: 'ResCCNuDeltaPlusPiPlus',
    1021: 'ResCCNuDelta2PlusPiMinus',
    1028: 'ResCCNuBarDelta0PiMinus',
    1032: 'ResCCNuBarDeltaMinusPiPlus',
    1039: 'ResCCNuProtonRhoPlus',
    1041: 'ResCCNuNeutronRhoPlus',
    1046: 'ResCCNuBarNeutronRhoMinus',
    1048: 'ResCCNuBarNeutronRho0',
    1053: 'ResCCNuSigmaPlusKaonPlus',
    1055: 'ResCCNuSigmaPlusKaon0',
    1060: 'ResCCNuBarSigmaMinusKaon0',
    1062: 'ResCCNuBarSigma0Kaon0',
    1067: 'ResCCNuProtonEta',
    1070: 'ResCCNuBarNeutronEta',
    1073: 'ResCCNuKaonPlusLambda0',
    1076: 'ResCCNuBarKaon0Lambda0',
    1079: 'ResCCNuProtonPiPlusPiMinus',
    1080: 'ResCCNuProtonPi0Pi0',
    1085: 'ResCCNuBarNeutronPiPlusPiMinus',
    1086: 'ResCCNuBarNeutronPi0Pi0',
    1090: 'ResCCNuBarProtonPi0Pi0',
    1091: 'CCDIS',
    1092: 'NCDIS',
    1093: 'UnUsed1',
    1094: 'UnUsed2',
    1095: 'CCQEHyperon',
    1096: 'NCCOH',
    1097: 'CCCOH',
    1098: 'NuElectronElastic',
    1099: 'InverseMuDecay',
    1100: 'MEC2p2h'
}

# Liquid argon properties
LAR_DENSITY = 1.396        # Density [g/cm^3]
LAR_Z       = 18           # Nuclear number
LAR_A       = 39.9481      # Nuclear mass [g/mol]
LAR_MEE     = 188.0 * 1e-6 # Mean excitation energy [MeV]
LAR_X0      = 14.0         # Radiation length [cm]
LAR_WION    = 23.6 * 1e-6  # Ionization work function [MeV]

# Sternheimer parametrization of density effects in liquid argon
LAR_a      = 0.19559
LAR_k      = 3.0000
LAR_x0     = 0.2000
LAR_x1     = 3.0000
LAR_Cbar   = 5.2146
LAR_delta0 = 0.00
