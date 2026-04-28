"""Tests for the canonical SPINE constants package."""

import pytest

import spine.constants as constants
from spine.constants.columns import (
    ANCST_COL,
    BATCH_COL,
    CLUST_COL,
    COORD_COLS,
    COORD_COLS_HI,
    COORD_COLS_LO,
    COORD_END_COLS,
    COORD_END_COLS_HI,
    COORD_END_COLS_LO,
    COORD_START_COLS,
    COORD_START_COLS_HI,
    COORD_START_COLS_LO,
    COORD_TIME_COL,
    GROUP_COL,
    INTER_COL,
    MOM_COL,
    NU_COL,
    PART_COL,
    PID_COL,
    PPN_CLASS_COLS,
    PPN_CLASS_COLS_HI,
    PPN_CLASS_COLS_LO,
    PPN_END_COLS,
    PPN_END_COLS_HI,
    PPN_END_COLS_LO,
    PPN_LENDP_COL,
    PPN_LPART_COL,
    PPN_LTYPE_COL,
    PPN_OCC_COL,
    PPN_ROFF_COLS,
    PPN_ROFF_COLS_HI,
    PPN_ROFF_COLS_LO,
    PPN_RPOS_COLS,
    PPN_RPOS_COLS_HI,
    PPN_RPOS_COLS_LO,
    PPN_RTYPE_COLS,
    PPN_RTYPE_COLS_HI,
    PPN_RTYPE_COLS_LO,
    PPN_SCORE_COLS,
    PPN_SCORE_COLS_HI,
    PPN_SCORE_COLS_LO,
    PPN_SHAPE_COL,
    PRGRP_COL,
    PRINT_COL,
    SHAPE_COL,
    VALUE_COL,
    VTX_COLS,
    VTX_COLS_HI,
    VTX_COLS_LO,
    ClusterLabelCol,
    CoordLabelCol,
    PPNPredCol,
    SparseTensorCol,
)
from spine.constants.enums import (
    DELTA_SHP,
    ELEC_PID,
    GHOST_SHP,
    KAON_PID,
    LOWES_SHP,
    MICHL_SHP,
    MUON_PID,
    PHOT_PID,
    PION_PID,
    PROT_PID,
    SHOWR_SHP,
    TRACK_SHP,
    UNKWN_SHP,
    GenieNuInteractionType,
    LArSoftNuInteractionType,
    NuCurrentType,
    NuInteractionScheme,
    NuInteractionType,
    ParticlePID,
    ParticleShape,
)
from spine.constants.factory import enum_factory
from spine.constants.labels import (
    NU_CURR_TYPE,
    NU_INT_TYPE,
    PDG_TO_PID,
    PID_LABELS,
    PID_TAGS,
    PID_TO_PDG,
    SHAPE_LABELS,
    SHAPE_PREC,
    SHP_TO_PID,
    SHP_TO_PRIMARY,
)
from spine.constants.physics import (
    ELEC_MASS,
    ELEC_MASS_MEV_C2,
    KAON_MASS,
    KAON_MASS_MEV_C2,
    LAR_A,
    LAR_A_G_MOL,
    LAR_CRITICAL_ENERGY_MEV,
    LAR_DENSITY,
    LAR_DENSITY_G_CM3,
    LAR_E_CRIT,
    LAR_MEAN_EXCITATION_ENERGY_MEV,
    LAR_MEE,
    LAR_MOLIERE_RADIUS_CM,
    LAR_RADIATION_LENGTH_CM,
    LAR_RM,
    LAR_WION,
    LAR_WION_MEV,
    LAR_X0,
    MUON_MASS,
    MUON_MASS_MEV_C2,
    PHOT_MASS,
    PHOT_MASS_MEV_C2,
    PID_MASSES,
    PION_MASS,
    PION_MASS_MEV_C2,
    PROT_MASS,
    PROT_MASS_MEV_C2,
)
from spine.constants.sentinels import (
    INVAL_ID,
    INVAL_IDX,
    INVAL_PDG,
    INVAL_TID,
    INVALID_PDG,
    LARCV_INVALID_INDEX,
    LARCV_INVALID_INSTANCE_ID,
    LARCV_INVALID_UINT,
)


def test_enum_factory_parses_scalar_and_sequence_values():
    """Enum factory should parse supported groups and uppercase values."""
    assert enum_factory("shape", "shower") == int(ParticleShape.SHOWER)
    assert enum_factory("pid", "MUON") == int(ParticlePID.MUON)
    assert enum_factory("interaction_scheme", "genie") == int(NuInteractionScheme.GENIE)
    assert enum_factory("cluster", "particle") == int(ClusterLabelCol.PARTICLE)
    assert enum_factory("pid", ["photon", "proton"]) == [
        int(ParticlePID.PHOTON),
        int(ParticlePID.PROTON),
    ]


def test_enum_factory_rejects_unknown_enum_group():
    """Enum factory should fail clearly on unsupported enum groups."""
    with pytest.raises(AssertionError, match="Enumerated type not recognized"):
        enum_factory("not_a_real_group", "value")


def test_enum_factory_rejects_unknown_enum_member():
    """Enum factory should fail clearly on unsupported member names."""
    with pytest.raises(ValueError, match="Enumerated object not recognized"):
        enum_factory("shape", "not_a_real_shape")


def test_package_root_reexports_common_symbols():
    """Package root should expose the documented convenience exports."""
    assert constants.ParticleShape is ParticleShape
    assert constants.ParticlePID is ParticlePID
    assert constants.NuCurrentType is NuCurrentType
    assert constants.NuInteractionScheme is NuInteractionScheme
    assert constants.LArSoftNuInteractionType is LArSoftNuInteractionType
    assert constants.GenieNuInteractionType is GenieNuInteractionType
    assert constants.enum_factory is enum_factory
    assert constants.COORD_COLS == COORD_COLS
    assert constants.LAR_DENSITY_G_CM3 == LAR_DENSITY_G_CM3
    assert constants.LARCV_INVALID_INDEX == LARCV_INVALID_INDEX


def test_enum_aliases_match_canonical_members():
    """Backward-compatible enum aliases should remain consistent."""
    assert SHOWR_SHP == int(ParticleShape.SHOWER)
    assert TRACK_SHP == int(ParticleShape.TRACK)
    assert MICHL_SHP == int(ParticleShape.MICHEL)
    assert DELTA_SHP == int(ParticleShape.DELTA)
    assert LOWES_SHP == int(ParticleShape.LOWE)
    assert GHOST_SHP == int(ParticleShape.GHOST)
    assert UNKWN_SHP == int(ParticleShape.LARCV_UNKNOWN)

    assert PHOT_PID == int(ParticlePID.PHOTON)
    assert ELEC_PID == int(ParticlePID.ELECTRON)
    assert MUON_PID == int(ParticlePID.MUON)
    assert PION_PID == int(ParticlePID.PION)
    assert PROT_PID == int(ParticlePID.PROTON)
    assert KAON_PID == int(ParticlePID.KAON)

    assert NuInteractionType is LArSoftNuInteractionType


def test_label_mappings_remain_consistent_with_enums():
    """Derived label maps should stay aligned with the canonical enums."""
    assert SHAPE_PREC == (
        int(ParticleShape.TRACK),
        int(ParticleShape.MICHEL),
        int(ParticleShape.SHOWER),
        int(ParticleShape.DELTA),
        int(ParticleShape.LOWE),
        int(ParticleShape.LARCV_UNKNOWN),
    )
    assert SHAPE_LABELS[int(ParticleShape.SHOWER)] == "Shower"
    assert SHAPE_LABELS[int(ParticleShape.UNKNOWN)] == "Unknown"

    assert PID_LABELS[int(ParticlePID.PROTON)] == "Proton"
    assert PID_TAGS[int(ParticlePID.MUON)] == "mu"

    assert NU_CURR_TYPE[int(NuCurrentType.CC)] == "CC"
    assert NU_INT_TYPE[int(LArSoftNuInteractionType.UNKNOWN)] == "UnknownInteraction"
    assert NU_INT_TYPE[int(LArSoftNuInteractionType.QE)] == "QE"

    for pdg, pid in PDG_TO_PID.items():
        assert PID_TO_PDG[pid] == abs(pdg)
    assert PID_TO_PDG[int(ParticlePID.UNKNOWN)] == -1

    assert SHP_TO_PID[int(ParticleShape.SHOWER)] == [
        int(ParticlePID.PHOTON),
        int(ParticlePID.ELECTRON),
    ]
    assert SHP_TO_PID[int(ParticleShape.TRACK)] == [
        int(ParticlePID.MUON),
        int(ParticlePID.PION),
        int(ParticlePID.PROTON),
        int(ParticlePID.KAON),
    ]
    assert SHP_TO_PRIMARY[int(ParticleShape.DELTA)] == (0,)
    assert SHP_TO_PRIMARY[int(ParticleShape.TRACK)] == (0, 1)


def test_column_groups_and_legacy_aliases_remain_consistent():
    """Grouped selectors and compatibility aliases should stay in sync."""
    assert COORD_COLS == (
        int(SparseTensorCol.X),
        int(SparseTensorCol.Y),
        int(SparseTensorCol.Z),
    )
    assert COORD_COLS_LO == int(SparseTensorCol.X)
    assert COORD_COLS_HI == int(SparseTensorCol.Z) + 1

    assert VTX_COLS == (
        int(ClusterLabelCol.VTX_X),
        int(ClusterLabelCol.VTX_Y),
        int(ClusterLabelCol.VTX_Z),
    )
    assert VTX_COLS_LO == int(ClusterLabelCol.VTX_X)
    assert VTX_COLS_HI == int(ClusterLabelCol.VTX_Z) + 1

    assert COORD_START_COLS == (
        int(CoordLabelCol.START_X),
        int(CoordLabelCol.START_Y),
        int(CoordLabelCol.START_Z),
    )
    assert COORD_START_COLS_LO == int(CoordLabelCol.START_X)
    assert COORD_START_COLS_HI == int(CoordLabelCol.START_Z) + 1

    assert COORD_END_COLS == (
        int(CoordLabelCol.END_X),
        int(CoordLabelCol.END_Y),
        int(CoordLabelCol.END_Z),
    )
    assert COORD_END_COLS_LO == int(CoordLabelCol.END_X)
    assert COORD_END_COLS_HI == int(CoordLabelCol.END_Z) + 1

    assert PPN_ROFF_COLS == (
        int(PPNPredCol.RAW_DX),
        int(PPNPredCol.RAW_DY),
        int(PPNPredCol.RAW_DZ),
    )
    assert PPN_ROFF_COLS_LO == int(PPNPredCol.RAW_DX)
    assert PPN_ROFF_COLS_HI == int(PPNPredCol.RAW_DZ) + 1

    assert PPN_RTYPE_COLS == (
        int(PPNPredCol.RAW_TYPE_0),
        int(PPNPredCol.RAW_TYPE_1),
        int(PPNPredCol.RAW_TYPE_2),
        int(PPNPredCol.RAW_TYPE_3),
        int(PPNPredCol.RAW_TYPE_4),
    )
    assert PPN_RTYPE_COLS_LO == int(PPNPredCol.RAW_TYPE_0)
    assert PPN_RTYPE_COLS_HI == int(PPNPredCol.RAW_TYPE_4) + 1

    assert PPN_RPOS_COLS == (int(PPNPredCol.RAW_POS_0), int(PPNPredCol.RAW_POS_1))
    assert PPN_RPOS_COLS_LO == int(PPNPredCol.RAW_POS_0)
    assert PPN_RPOS_COLS_HI == int(PPNPredCol.RAW_POS_1) + 1

    assert PPN_SCORE_COLS == (int(PPNPredCol.SCORE_NEG), int(PPNPredCol.SCORE_POS))
    assert PPN_SCORE_COLS_LO == int(PPNPredCol.SCORE_NEG)
    assert PPN_SCORE_COLS_HI == int(PPNPredCol.SCORE_POS) + 1

    assert PPN_CLASS_COLS == (
        int(PPNPredCol.CLASS_0),
        int(PPNPredCol.CLASS_1),
        int(PPNPredCol.CLASS_2),
        int(PPNPredCol.CLASS_3),
        int(PPNPredCol.CLASS_4),
    )
    assert PPN_CLASS_COLS_LO == int(PPNPredCol.CLASS_0)
    assert PPN_CLASS_COLS_HI == int(PPNPredCol.CLASS_4) + 1

    assert PPN_END_COLS == (int(PPNPredCol.END_NEG), int(PPNPredCol.END_POS))
    assert PPN_END_COLS_LO == int(PPNPredCol.END_NEG)
    assert PPN_END_COLS_HI == int(PPNPredCol.END_POS) + 1

    assert BATCH_COL == int(SparseTensorCol.BATCH)
    assert VALUE_COL == int(SparseTensorCol.VALUE)
    assert CLUST_COL == int(ClusterLabelCol.CLUSTER)
    assert PART_COL == int(ClusterLabelCol.PARTICLE)
    assert GROUP_COL == int(ClusterLabelCol.GROUP)
    assert ANCST_COL == int(ClusterLabelCol.ANCESTOR)
    assert INTER_COL == int(ClusterLabelCol.INTERACTION)
    assert NU_COL == int(ClusterLabelCol.NU)
    assert PID_COL == int(ClusterLabelCol.PID)
    assert PRGRP_COL == int(ClusterLabelCol.GROUP_PRIMARY)
    assert PRINT_COL == int(ClusterLabelCol.INTER_PRIMARY)
    assert MOM_COL == int(ClusterLabelCol.MOMENTUM)
    assert SHAPE_COL == -1
    assert COORD_TIME_COL == int(CoordLabelCol.TIME)
    assert PPN_LTYPE_COL == 4
    assert PPN_LPART_COL == 5
    assert PPN_LENDP_COL == 6
    assert PPN_OCC_COL == int(PPNPredCol.OCCUPANCY)
    assert PPN_SHAPE_COL == int(PPNPredCol.SHAPE)


def test_physics_constants_and_aliases_remain_consistent():
    """Physics aliases and PID mass lookup should stay aligned."""
    assert PID_MASSES[int(ParticlePID.PHOTON)] == PHOT_MASS_MEV_C2
    assert PID_MASSES[int(ParticlePID.ELECTRON)] == ELEC_MASS_MEV_C2
    assert PID_MASSES[int(ParticlePID.MUON)] == MUON_MASS_MEV_C2
    assert PID_MASSES[int(ParticlePID.PION)] == PION_MASS_MEV_C2
    assert PID_MASSES[int(ParticlePID.PROTON)] == PROT_MASS_MEV_C2
    assert PID_MASSES[int(ParticlePID.KAON)] == KAON_MASS_MEV_C2

    assert PHOT_MASS == PHOT_MASS_MEV_C2
    assert ELEC_MASS == ELEC_MASS_MEV_C2
    assert MUON_MASS == MUON_MASS_MEV_C2
    assert PION_MASS == PION_MASS_MEV_C2
    assert PROT_MASS == PROT_MASS_MEV_C2
    assert KAON_MASS == KAON_MASS_MEV_C2

    assert LAR_DENSITY == LAR_DENSITY_G_CM3
    assert LAR_A == LAR_A_G_MOL
    assert LAR_MEE == LAR_MEAN_EXCITATION_ENERGY_MEV
    assert LAR_X0 == LAR_RADIATION_LENGTH_CM
    assert LAR_WION == LAR_WION_MEV
    assert LAR_E_CRIT == LAR_CRITICAL_ENERGY_MEV
    assert LAR_RM == LAR_MOLIERE_RADIUS_CM


def test_sentinel_aliases_remain_consistent():
    """Legacy sentinel aliases should still match canonical values."""
    assert INVAL_IDX == LARCV_INVALID_INDEX
    assert INVAL_ID == LARCV_INVALID_INSTANCE_ID
    assert INVAL_TID == LARCV_INVALID_UINT
    assert INVAL_PDG == INVALID_PDG
