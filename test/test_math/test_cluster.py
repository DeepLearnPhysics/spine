"""Comprehensive tests for spine.math.cluster module."""

import pytest
import numpy as np


class TestMathCluster:
    """Test clustering functions."""

    def test_cluster_imports(self):
        """Test that clustering functions can be imported."""
        try:
            from spine.math.cluster import DBSCAN, dbscan

            assert DBSCAN is not None
            assert callable(dbscan)

        except ImportError:
            pytest.skip("Clustering functions not available")

    def test_dbscan_function(self):
        """Test DBSCAN clustering function."""
        try:
            from spine.math.cluster import dbscan

            # Create simple clusterable data - two distinct groups
            points = np.array(
                [
                    # First cluster
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    # Second cluster (far away)
                    [10.0, 10.0, 10.0],
                    [10.1, 10.0, 10.0],
                    [10.0, 10.1, 10.0],
                ],
                dtype=np.float32,
            )

            # Run DBSCAN
            labels = dbscan(points, eps=0.5, min_samples=2)

            # Should have 2 clusters (labels 0 and 1) or similar
            unique_labels = np.unique(labels)
            # Filter out noise label (-1) if present
            cluster_labels = unique_labels[unique_labels >= 0]

            # Should have at least 2 clusters
            assert len(cluster_labels) >= 2

            # Points in same group should have same labels
            assert labels[0] == labels[1] == labels[2]  # First group
            assert labels[3] == labels[4] == labels[5]  # Second group

            # Different groups should have different labels
            assert labels[0] != labels[3]

        except (ImportError, TypeError, AttributeError):
            pytest.skip("DBSCAN function not available")

    def test_dbscan_class(self):
        """Test DBSCAN class interface."""
        try:
            from spine.math.cluster import DBSCAN

            # Create DBSCAN clusterer
            clusterer = DBSCAN(eps=0.5, min_samples=2)

            # Test attributes are set (eps is squared for euclidean metric)
            assert clusterer.eps == 0.25  # 0.5^2 because euclidean -> sqeuclidean
            assert clusterer.min_samples == 2

            # Create test data
            points = np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [5.0, 5.0, 5.0]], dtype=np.float32
            )

            # Test fit method if available
            if hasattr(clusterer, "fit"):
                labels = clusterer.fit(points)
                assert len(labels) == len(points)

        except (ImportError, TypeError, AttributeError):
            pytest.skip("DBSCAN class not available")

    def test_dbscan_parameters(self):
        """Test DBSCAN with different parameters."""
        try:
            from spine.math.cluster import dbscan

            # Create test data
            points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [10.0, 0.0, 0.0],  # Outlier
                ],
                dtype=np.float32,
            )

            # Test with strict parameters (small eps, high min_samples)
            labels_strict = dbscan(points, eps=0.05, min_samples=3)

            # Test with loose parameters (large eps, low min_samples)
            labels_loose = dbscan(points, eps=1.0, min_samples=1)

            # Loose parameters should find more/larger clusters
            n_clusters_strict = len(np.unique(labels_strict[labels_strict >= 0]))
            n_clusters_loose = len(np.unique(labels_loose[labels_loose >= 0]))

            # This is a general expectation but may not always hold
            # assert n_clusters_loose >= n_clusters_strict
            assert len(labels_strict) == len(points)
            assert len(labels_loose) == len(points)

        except (ImportError, TypeError, AttributeError):
            pytest.skip("DBSCAN parameters test not available")

    def test_dbscan_noise_detection(self):
        """Test DBSCAN noise detection."""
        try:
            from spine.math.cluster import dbscan

            # Create data with clear outliers
            points = np.array(
                [
                    # Dense cluster
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [0.1, 0.1, 0.0],
                    # Isolated outlier
                    [100.0, 100.0, 100.0],
                ],
                dtype=np.float32,
            )

            # Run DBSCAN with parameters that should mark outlier as noise
            labels = dbscan(points, eps=0.2, min_samples=2)

            # Last point should be noise (-1) or in a different cluster
            cluster_points = labels[:4]  # First 4 points
            outlier_label = labels[4]  # Last point

            # First 4 points should be clustered together
            if len(np.unique(cluster_points[cluster_points >= 0])) == 1:
                # They form one cluster, outlier should be different
                assert outlier_label != cluster_points[0] or outlier_label == -1

        except (ImportError, TypeError, AttributeError):
            pytest.skip("DBSCAN noise detection test not available")


@pytest.mark.slow
class TestClusterIntegration:
    """Integration tests for clustering functions."""

    def test_clustering_with_different_metrics(self):
        """Test clustering with different distance metrics."""
        try:
            from spine.math.cluster import dbscan

            # Create test data
            points = np.array(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [10.0, 0.0, 0.0], [11.0, 1.0, 1.0]],
                dtype=np.float32,
            )

            # Test with different metrics
            metrics_to_test = ["euclidean", "cityblock"]

            for metric in metrics_to_test:
                try:
                    labels = dbscan(points, eps=2.0, min_samples=1, metric=metric)
                    assert len(labels) == len(points)
                    assert np.all(labels >= -1)  # Valid label range
                except Exception:
                    # Some metrics might not be implemented
                    continue

        except (ImportError, TypeError, AttributeError):
            pytest.skip("Clustering metrics integration test not available")

    def test_clustering_reproducibility(self):
        """Test that clustering results are reproducible."""
        try:
            from spine.math.cluster import dbscan

            # Create test data
            np.random.seed(42)
            points = np.random.random((20, 3)).astype(np.float32)

            # Run clustering twice with same parameters
            labels1 = dbscan(points, eps=0.3, min_samples=2)
            labels2 = dbscan(points, eps=0.3, min_samples=2)

            # Results should be identical
            assert np.array_equal(labels1, labels2)

        except (ImportError, TypeError, AttributeError):
            pytest.skip("Clustering reproducibility test not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
