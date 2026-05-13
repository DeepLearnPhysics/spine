"""Tests for output object visualization."""

import numpy as np

from spine.data.out import RecoParticle
from spine.vis import Drawer


def test_drawer_accepts_derived_hover_attr():
    """Drawer validation should accept derived DataBase attributes."""
    data = {
        "points": np.empty((0, 3), dtype=np.float32),
        "reco_particles": [RecoParticle()],
    }

    figure = Drawer(data, draw_mode="reco").get("particles", attr=["ke"])

    assert figure is not None


def test_drawer_accepts_skipped_hover_attr():
    """Drawer validation should still accept skipped DataBase attributes."""
    data = {
        "points": np.empty((0, 3), dtype=np.float32),
        "reco_particles": [RecoParticle()],
    }

    figure = Drawer(data, draw_mode="reco").get("particles", attr=["depositions"])

    assert figure is not None
