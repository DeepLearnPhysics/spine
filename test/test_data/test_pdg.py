"""Tests for optional PDG particle-name resolution."""

import numpy as np

from spine.data import pdg as pdg_module
from spine.data import pdg_name


def test_pdg_name_resolves_particles_antiparticles_and_nuclei(pdg_lookup):
    """Known PDG identifiers should receive canonical display names."""
    assert pdg_name(np.int64(13)) == "mu-"
    assert pdg_name(-13) == "mu+"
    assert pdg_name(1000010020) == "D2"


def test_pdg_name_preserves_unknown_and_missing_codes(pdg_lookup):
    """Unknown identifiers and SPINE sentinels should not receive labels."""
    assert pdg_name(-1) == "d~"
    assert pdg_name(0) == "UNKNOWN"
    assert pdg_name(99999999) is None
    assert pdg_name(13.5) is None
    assert pdg_name(None) is None


def test_pdg_name_without_optional_dependency(monkeypatch):
    """Core-only installations should retain numeric fallback behavior."""
    monkeypatch.setattr(pdg_module, "PARTICLE_AVAILABLE", False)
    assert pdg_name(13) is None
    assert pdg_name(0) == "UNKNOWN"
