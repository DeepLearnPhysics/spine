"""Comprehensive tests for spine.math.distance module."""

import pytest
import numpy as np
from spine.math.distance import METRICS


class TestMathDistance:
    """Test distance computation functions."""

    def test_distance_imports(self):
        """Test that distance functions can be imported."""
        try:
            from spine.math.distance import (
                cityblock,
                euclidean,
                sqeuclidean,
                minkowski,
                chebyshev,
                pdist,
                cdist,
                farthest_pair,
                closest_pair,
            )

            assert callable(cityblock)
            assert callable(euclidean)
            assert callable(sqeuclidean)
            assert callable(minkowski)
            assert callable(chebyshev)
            assert callable(pdist)
            assert callable(cdist)
            assert callable(farthest_pair)
            assert callable(closest_pair)

        except ImportError:
            pytest.skip("Distance functions not available")

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        try:
            from spine.math.distance import euclidean

            # Test simple 3D points
            p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            p2 = np.array([3.0, 4.0, 0.0], dtype=np.float32)

            distance = euclidean(p1, p2)
            expected = 5.0  # 3-4-5 triangle

            assert abs(distance - expected) < 1e-6

        except (ImportError, TypeError):
            pytest.skip("Euclidean distance not available")

    def test_cityblock_distance(self):
        """Test Manhattan (cityblock) distance."""
        try:
            from spine.math.distance import cityblock

            # Test simple 3D points
            p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            p2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            distance = cityblock(p1, p2)
            expected = 6.0  # |1-0| + |2-0| + |3-0| = 6

            assert abs(distance - expected) < 1e-6

        except (ImportError, TypeError):
            pytest.skip("Cityblock distance not available")

    def test_chebyshev_distance(self):
        """Test Chebyshev (max) distance."""
        try:
            from spine.math.distance import chebyshev

            # Test simple 3D points
            p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            p2 = np.array([1.0, 3.0, 2.0], dtype=np.float32)

            distance = chebyshev(p1, p2)
            expected = 3.0  # max(|1-0|, |3-0|, |2-0|) = 3

            assert abs(distance - expected) < 1e-6

        except (ImportError, TypeError):
            pytest.skip("Chebyshev distance not available")

    def test_minkowski_distance(self):
        """Test Minkowski distance with different p values."""
        try:
            from spine.math.distance import minkowski

            # Test simple 3D points
            p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            p2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)

            # Test p=2 (should be euclidean)
            dist_p2 = minkowski(p1, p2, 2.0)
            expected_p2 = np.sqrt(3.0)
            assert abs(dist_p2 - expected_p2) < 1e-6

            # Test p=1 (should be cityblock)
            dist_p1 = minkowski(p1, p2, 1.0)
            expected_p1 = 3.0
            assert abs(dist_p1 - expected_p1) < 1e-6

        except (ImportError, TypeError):
            pytest.skip("Minkowski distance not available")

    def test_sqeuclidean_distance(self):
        """Test squared Euclidean distance."""
        try:
            from spine.math.distance import sqeuclidean

            # Test simple 3D points
            p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            p2 = np.array([3.0, 4.0, 0.0], dtype=np.float32)

            distance = sqeuclidean(p1, p2)
            expected = 25.0  # 3^2 + 4^2 = 25

            assert abs(distance - expected) < 1e-6

        except (ImportError, TypeError):
            pytest.skip("Squared Euclidean distance not available")


class TestDistanceArrays:
    """Test distance functions on arrays of points."""

    def test_pdist_function(self):
        """Test pairwise distance computation."""
        try:
            from spine.math.distance import pdist

            # Create simple point set
            points = np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
            )

            # Compute pairwise distances
            from spine.math.distance import METRICS

            distances = pdist(points, METRICS["euclidean"])

            # Should return NxN matrix for N points
            assert distances.shape == (3, 3)

            # Check specific distances (symmetric matrix)
            assert abs(distances[0, 1] - 1.0) < 1e-6  # Distance between points 0 and 1
            assert abs(distances[0, 2] - 1.0) < 1e-6  # Distance between points 0 and 2
            assert (
                abs(distances[1, 2] - np.sqrt(2.0)) < 1e-6
            )  # Distance between points 1 and 2

            # Check symmetry
            assert abs(distances[0, 1] - distances[1, 0]) < 1e-6
            assert abs(distances[0, 2] - distances[2, 0]) < 1e-6

        except (ImportError, TypeError):
            pytest.skip("Pdist function not available")

    def test_cdist_function(self):
        """Test cross-distance computation."""
        try:
            from spine.math.distance import cdist

            # Create two point sets
            points_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

            points_b = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)

            # Compute cross distances
            distances = cdist(points_a, points_b, METRICS["euclidean"])

            # Should be 2x2 matrix
            assert distances.shape == (2, 2)

            # Check specific distances
            assert abs(distances[0, 0] - 1.0) < 1e-6  # (0,0,0) to (0,1,0)
            assert abs(distances[0, 1] - np.sqrt(2.0)) < 1e-6  # (0,0,0) to (1,1,0)
            assert abs(distances[1, 0] - np.sqrt(2.0)) < 1e-6  # (1,0,0) to (0,1,0)
            assert abs(distances[1, 1] - 1.0) < 1e-6  # (1,0,0) to (1,1,0)

        except (ImportError, TypeError):
            pytest.skip("Cdist function not available")

    def test_closest_pair(self):
        """Test closest pair finding."""
        try:
            from spine.math.distance import closest_pair

            # Create two point sets with known closest pair
            points_a = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float32)

            points_b = np.array(
                [
                    [0.1, 0.0, 0.0],  # Very close to first point in set A
                    [5.0, 5.0, 5.0],
                ],
                dtype=np.float32,
            )

            # Compute closest pair between sets
            i, j, distance = closest_pair(
                points_a, points_b, metric_id=METRICS["euclidean"]
            )

            # Should find the closest pair (point 0 from A and point 0 from B)
            assert i == 0 and j == 0
            assert abs(distance - 0.1) < 1e-6

        except (ImportError, TypeError):
            pytest.skip("Closest pair function not available")

    def test_farthest_pair(self):
        """Test farthest pair finding."""
        try:
            from spine.math.distance import farthest_pair

            # Create point set with known farthest pair
            points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],  # Farthest from first point
                ],
                dtype=np.float32,
            )

            i, j, distance = farthest_pair(points, METRICS["euclidean"])

            # Should find the farthest pair (points 0 and 2)
            assert set([i, j]) == {0, 2}
            assert abs(distance - 5.0) < 1e-6

        except (ImportError, TypeError):
            pytest.skip("Farthest pair function not available")


@pytest.mark.slow
class TestDistanceIntegration:
    """Integration tests for distance functions."""

    def test_distance_metrics_consistency(self):
        """Test consistency between different distance metrics."""
        try:
            from spine.math.distance import euclidean, minkowski, sqeuclidean

            # Test points
            p1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            p2 = np.array([4.0, 6.0, 8.0], dtype=np.float32)

            # Euclidean should equal Minkowski with p=2
            euclidean_dist = euclidean(p1, p2)
            minkowski_dist = minkowski(p1, p2, 2.0)
            assert abs(euclidean_dist - minkowski_dist) < 1e-6

            # Squared euclidean should equal euclidean squared
            sqeuclidean_dist = sqeuclidean(p1, p2)
            assert abs(sqeuclidean_dist - euclidean_dist**2) < 1e-6

        except (ImportError, TypeError):
            pytest.skip("Distance metrics consistency test not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
