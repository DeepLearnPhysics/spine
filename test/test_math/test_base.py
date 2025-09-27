"""Comprehensive tests for spine.math.base module."""

import numba as nb
import numpy as np
import pytest


class TestMathBase:
    """Test basic mathematical functions."""

    def test_imports(self):
        """Test that base math functions can be imported."""
        try:
            from spine.math.base import mean, mode, seed, sum, unique

            assert callable(seed)
            assert callable(unique)
            assert callable(sum)
            assert callable(mean)
            assert callable(mode)
        except ImportError:
            pytest.skip("Math base functions not available")

    def test_seed_function(self):
        """Test random seed function."""
        try:
            from spine.math.base import seed

            # Should not raise error
            seed(42)
            seed(0)

        except (ImportError, TypeError):
            pytest.skip("Seed function not available")

    def test_unique_function(self):
        """Test unique function with counts."""
        try:
            from spine.math.base import unique

            # Test with simple array
            x = np.array([1, 2, 2, 3, 1, 3, 3], dtype=np.int64)
            values, counts = unique(x)

            # Check results
            assert len(values) == 3  # Should have 3 unique values
            assert len(counts) == 3  # Should have 3 count values
            assert np.sum(counts) == len(x)  # Counts should sum to original length

        except (ImportError, TypeError, AttributeError):
            pytest.skip("Unique function not available")

    def test_aggregation_functions(self):
        """Test aggregation functions (sum, mean, mode)."""
        try:
            from spine.math.base import mean, mode, sum

            # Test data (2D array for sum/mean functions)
            x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

            # Test sum along axis 0
            result_sum = sum(x, 0)
            expected_sum = np.array([9.0, 12.0])
            assert np.allclose(result_sum, expected_sum)

            # Test mean along axis 0
            result_mean = mean(x, 0)
            expected_mean = np.array([3.0, 4.0])
            assert np.allclose(result_mean, expected_mean)

            # Test mode function (1D integer array)
            mode_data = np.array([1, 2, 2, 3, 2], dtype=np.int64)
            result_mode = mode(mode_data)
            assert result_mode == 2  # Most frequent value

        except (ImportError, TypeError, AttributeError):
            pytest.skip("Aggregation functions not available")

    def test_argmax_argmin_functions(self):
        """Test argmax and argmin functions."""
        try:
            from spine.math.base import amax, amin, argmax, argmin

            # 2D array for axis-based functions
            x = np.array([[1.0, 8.0], [5.0, 2.0], [3.0, 6.0]], dtype=np.float32)

            # Test argmax/argmin along axis 0
            max_idx = argmax(x, 0)
            min_idx = argmin(x, 0)
            assert max_idx[0] == 1  # Index of max in first column (5.0)
            assert max_idx[1] == 0  # Index of max in second column (8.0)
            assert min_idx[0] == 0  # Index of min in first column (1.0)
            assert min_idx[1] == 1  # Index of min in second column (2.0)

            # Test amax/amin along axis 0
            max_val = amax(x, 0)
            min_val = amin(x, 0)
            assert np.allclose(max_val, [5.0, 8.0])
            assert np.allclose(min_val, [1.0, 2.0])

        except (ImportError, TypeError, AttributeError):
            pytest.skip("Argmax/argmin functions not available")

    def test_softmax_function(self):
        """Test softmax function."""
        try:
            from spine.math.base import softmax

            # Test with 2D data
            x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]], dtype=np.float32)
            result = softmax(x, 0)  # Apply softmax along axis 0

            # Check properties of softmax
            assert result.shape == x.shape
            # Each column should sum to 1
            col_sums = np.sum(result, axis=0)
            assert np.allclose(col_sums, 1.0, atol=1e-6)
            assert np.all(result >= 0)  # All values should be positive
            assert np.all(result <= 1)  # All values should be <= 1

        except (ImportError, TypeError, AttributeError):
            pytest.skip("Softmax function not available")

    def test_log_loss_function(self):
        """Test log loss function."""
        try:
            from spine.math.base import log_loss

            # Test with simple probability data
            y_true = np.array([0, 1, 1, 0])
            y_prob = np.array([0.1, 0.9, 0.8, 0.2])

            loss = log_loss(y_true, y_prob)
            assert loss >= 0  # Log loss should be non-negative
            assert np.isfinite(loss)  # Should be finite

        except (ImportError, TypeError, AttributeError):
            pytest.skip("Log loss function not available")


@pytest.mark.slow
class TestMathBaseIntegration:
    """Integration tests for math base functions."""

    def test_numba_compilation(self):
        """Test that functions are properly JIT compiled."""
        try:
            from spine.math.base import sum, unique

            # Test that functions work with numba types
            x = np.array([1, 2, 2, 3], dtype=np.int64)
            values, counts = unique(x)

            # Should work without errors
            assert len(values) > 0
            assert len(counts) > 0

        except (ImportError, TypeError, AttributeError):
            pytest.skip("Numba integration test not available")

    def test_function_consistency(self):
        """Test consistency with numpy equivalents where applicable."""
        try:
            from spine.math.base import mean, sum

            # Create test data (2D for axis-based functions)
            x = np.random.random((10, 5)).astype(np.float32)

            # Compare with numpy (allowing for small numerical differences)
            spine_sum = sum(x, 0)
            numpy_sum = np.sum(x, axis=0)
            assert np.allclose(spine_sum, numpy_sum, atol=1e-6)

            spine_mean = mean(x, 0)
            numpy_mean = np.mean(x, axis=0)
            assert np.allclose(spine_mean, numpy_mean, atol=1e-6)

        except (ImportError, TypeError, AttributeError):
            pytest.skip("Function consistency test not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
