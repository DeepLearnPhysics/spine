"""Tests for decomposition helpers."""

import numpy as np
import pytest

from spine.math.decomposition import PCA, principal_components


def test_principal_components_are_orthonormal():
    """Principal component vectors should form an orthonormal basis."""
    x = np.array(
        [[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    components = principal_components(x)

    np.testing.assert_allclose(components @ components.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.abs(components[0]), [1.0, 0.0, 0.0], atol=1e-6)


def test_pca_fit_returns_requested_components():
    """PCA jitclass should return the requested number of components."""
    x = np.array(
        [[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    pca = PCA(2)

    components, variance = pca.fit(x)

    assert components.shape == (2, 3)
    assert variance.shape == (2,)
    np.testing.assert_allclose(np.abs(components[0]), [1.0, 0.0, 0.0], atol=1e-6)
    assert variance[0] > variance[1]


def test_pca_rejects_invalid_component_count():
    """PCA should reject empty or overcomplete component requests."""
    with pytest.raises(AssertionError, match="one component"):
        PCA(0)

    pca = PCA(4)
    with pytest.raises(AssertionError, match="dimensionality"):
        pca.fit(np.ones((3, 3), dtype=np.float32))


def test_pca_rejects_undersampled_inputs():
    """PCA requires at least two samples to produce meaningful variance."""
    x = np.ones((1, 3), dtype=np.float32)

    with pytest.raises(AssertionError, match="two samples"):
        principal_components(x)

    with pytest.raises(AssertionError, match="two samples"):
        PCA(1).fit(x)
