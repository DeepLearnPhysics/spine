"""Tests for particle visualization helpers."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from spine.constants import PART_COL
from spine.data import Particle
from spine.vis.drawer.particle import scatter_particles

particle_module = import_module("spine.vis.drawer.particle")

POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def test_scatter_particles_uses_truth_particle_metadata():
    particle = Particle(id=0, group_id=1, interaction_id=2, pid=3, shape=4)
    labels = np.zeros((2, PART_COL + 1), dtype=np.float32)
    labels[:, :3] = POINTS
    labels[:, PART_COL] = 0

    traces = scatter_particles(labels, [particle])

    assert len(traces) == 1
    assert traces[0].name == "Particle 0"
    assert "Particle ID" in traces[0].hovertemplate


def test_scatter_particles_skips_empty_entries_and_casts_external_labels(monkeypatch):
    labels = np.zeros((1, PART_COL + 1), dtype=np.float32)
    labels[0, :3] = POINTS[0]
    labels[0, PART_COL] = 1

    monkeypatch.setattr(
        particle_module.Particle,
        "from_larcv",
        lambda obj: Particle(id=obj.id, group_id=0, interaction_id=0, pid=0, shape=0),
    )
    traces = scatter_particles(labels, [Particle(id=0), SimpleNamespace(id=7)])

    assert len(traces) == 1
    assert traces[0].name == "Particle 7"
