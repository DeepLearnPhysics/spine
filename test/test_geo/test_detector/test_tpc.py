"""Tests for TPC detector geometry components."""

import numpy as np
import pytest

from spine.geo.detector.tpc import TPCChamber, TPCDetector, TPCModule


def test_tpc_chamber_and_module_properties():
    """TPC chambers and modules should expose drift and cathode geometry."""
    chamber = TPCChamber(
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 20.0, 30.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    assert chamber.drift_axis == 0
    assert chamber.drift_sign == 1
    assert chamber.anode_side == 1
    assert chamber.cathode_side == 0
    assert chamber.anode_pos == 5.0
    assert chamber.cathode_pos == -5.0

    module = TPCModule(
        np.array([[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
    )
    assert len(module) == 2
    assert list(module) == module.chambers
    assert module[0] is module.chambers[0]
    assert module.drift_axis == 0
    assert module.cathode_pos == 0.0
    assert module.cathode_thickness == 2.0


def test_tpc_module_uses_explicit_drift_directions():
    """Explicit drift directions should be used directly when provided."""
    module = TPCModule(
        positions=np.array([[0.0, 0.0, 0.0]]),
        dimensions=np.array([[10.0, 20.0, 30.0]]),
        drift_dirs=np.array([[0.0, 1.0, 0.0]]),
    )

    np.testing.assert_array_equal(module[0].drift_dir, [0.0, 1.0, 0.0])
    assert module[0].drift_axis == 1


def test_tpc_invalid_drift_direction():
    """TPC drift vectors must be axis aligned."""
    with pytest.raises(AssertionError, match="aligned with a base axis"):
        TPCChamber(
            np.zeros(3),
            np.ones(3),
            np.array([1.0, 1.0, 0.0]),
        )


def test_tpc_detector_with_limits_and_detector_ids():
    """TPCDetector should build modules, limits and logical detector IDs."""
    detector = TPCDetector(
        dimensions=[10.0, 20.0, 30.0],
        positions=[[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        module_ids=[0, 0],
        det_ids=[0, 1],
        limits={"intercepts": [[0.0, 0.0, 0.0]], "norms": [[1.0, 0.0, 0.0]]},
    )

    assert len(detector) == 1
    assert detector[0] is detector.modules[0]
    assert list(detector) == detector.modules
    assert detector.num_modules == 1
    assert detector.num_chambers == 2
    assert detector.num_chambers_per_module == 2
    assert len(detector.limits) == 1
    np.testing.assert_array_equal(detector.det_ids, [0, 1])


def test_tpc_detector_per_chamber_dimensions_and_drift_dirs():
    """Per-chamber dimensions and drift directions should be accepted."""
    detector = TPCDetector(
        dimensions=[[10.0, 20.0, 30.0], [8.0, 18.0, 28.0]],
        positions=[[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        module_ids=[0, 0],
        drift_dirs=[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    )

    np.testing.assert_allclose(detector[0][0].dimensions, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(detector[0][1].dimensions, [8.0, 18.0, 28.0])
    np.testing.assert_array_equal(detector[0][1].drift_dir, [-1.0, 0.0, 0.0])


def test_tpc_detector_rejects_bad_dimensions():
    """TPCDetector should reject malformed per-chamber dimensions."""
    with pytest.raises(AssertionError, match="one set of dimensions per TPC"):
        TPCDetector(
            dimensions=[[10.0, 20.0, 30.0]],
            positions=[[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            module_ids=[0, 0],
        )

    with pytest.raises(AssertionError, match="along 3 dimensions"):
        TPCDetector(
            dimensions=[[10.0, 20.0], [10.0, 20.0]],
            positions=[[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            module_ids=[0, 0],
        )


def test_tpc_detector_rejects_incomplete_detector_ids():
    """Physical TPC IDs should cover all chambers in a module."""
    with pytest.raises(AssertionError, match="All physical TPCs"):
        TPCDetector(
            dimensions=[10.0, 20.0, 30.0],
            positions=[[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            module_ids=[0, 0],
            det_ids=[0, 0],
        )
