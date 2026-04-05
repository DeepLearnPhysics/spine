"""Comprehensive test suite for spine.data.batch.base module."""

from dataclasses import dataclass

import numpy as np
import pytest

from spine.data.batch.base import BatchBase
from spine.utils.conditional import TORCH_AVAILABLE, torch


# Create a concrete implementation for testing the abstract BatchBase class
@dataclass(eq=False)
class ConcreteBatch(BatchBase):
    """Concrete implementation of BatchBase for testing purposes."""

    def __init__(self, data, counts, edges, batch_size, is_sparse=False, is_list=False):
        """Initialize with proper parent init."""
        # Detect if data is a list (when not explicitly specified)
        if not is_list and isinstance(data, (list, tuple)):
            is_list = True

        # For lists, pass the first element to parent (or empty array if empty list)
        if is_list:
            init_data = data[0] if len(data) > 0 else np.empty(0)
        else:
            init_data = data

        super().__init__(init_data, is_sparse=is_sparse, is_list=is_list)
        self.data = data
        self.counts = counts
        self.edges = edges
        self.batch_size = batch_size


class TestBatchBaseInitialization:
    """Test BatchBase initialization and type detection."""

    def test_numpy_initialization(self):
        """Test BatchBase detects numpy arrays correctly."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        batch = ConcreteBatch(
            data=data,
            counts=np.array([2, 1]),
            edges=np.array([0, 2, 3]),
            batch_size=2,
            is_sparse=False,
            is_list=False,
        )

        assert batch.is_numpy is True
        assert batch.is_sparse is False
        assert batch.is_list is False
        assert batch.dtype == np.int64
        assert batch.device is None

    def test_list_initialization(self):
        """Test BatchBase with list data."""
        data = [np.array([1, 2]), np.array([3, 4, 5])]

        # Create a mock list with dtype attribute
        class MockList(list):
            dtype = np.dtype(np.int64)

        mock_data = MockList(data)
        batch = ConcreteBatch(
            data=mock_data,
            counts=np.array([2, 3]),
            edges=np.array([0, 2, 5]),
            batch_size=2,
            is_sparse=False,
            is_list=True,
        )

        assert batch.is_list is True
        assert batch.is_numpy is True

    def test_dtype_storage(self):
        """Test BatchBase stores dtype correctly."""
        data_int = np.array([1, 2, 3], dtype=np.int32)
        batch_int = ConcreteBatch(
            data=data_int,
            counts=np.array([3]),
            edges=np.array([0, 3]),
            batch_size=1,
        )

        assert batch_int.dtype == np.int32

        data_float = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        batch_float = ConcreteBatch(
            data=data_float,
            counts=np.array([3]),
            edges=np.array([0, 3]),
            batch_size=1,
        )

        assert batch_float.dtype == np.float64


class TestBatchBaseLength:
    """Test BatchBase length property."""

    def test_len_single_entry(self):
        """Test length with single batch entry."""
        data = np.array([[1, 2, 3]])
        batch = ConcreteBatch(
            data=data, counts=np.array([1]), edges=np.array([0, 1]), batch_size=1
        )

        assert len(batch) == 1

    def test_len_multiple_entries(self):
        """Test length with multiple batch entries."""
        data = np.array([[1], [2], [3], [4], [5]])
        batch = ConcreteBatch(
            data=data,
            counts=np.array([2, 1, 2]),
            edges=np.array([0, 2, 3, 5]),
            batch_size=3,
        )

        assert len(batch) == 3

    def test_len_empty_batch(self):
        """Test length with empty batch."""
        data = np.array([]).reshape(0, 3)
        batch = ConcreteBatch(
            data=data, counts=np.array([]), edges=np.array([0]), batch_size=0
        )

        assert len(batch) == 0


class TestBatchBaseEquality:
    """Test BatchBase equality method."""

    def test_equality_same_objects(self):
        """Test equality with identical batches."""
        data = np.array([[1, 2], [3, 4]])
        batch1 = ConcreteBatch(
            data=data, counts=np.array([2]), edges=np.array([0, 2]), batch_size=1
        )
        batch2 = ConcreteBatch(
            data=data, counts=np.array([2]), edges=np.array([0, 2]), batch_size=1
        )

        assert batch1 == batch2

    def test_equality_different_data(self):
        """Test inequality with different data."""
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[1, 2], [5, 6]])
        batch1 = ConcreteBatch(
            data=data1, counts=np.array([2]), edges=np.array([0, 2]), batch_size=1
        )
        batch2 = ConcreteBatch(
            data=data2, counts=np.array([2]), edges=np.array([0, 2]), batch_size=1
        )

        assert batch1 != batch2

    def test_equality_different_classes(self):
        """Test inequality with different class types."""

        @dataclass(eq=False)
        class OtherBatch(BatchBase):
            pass

        data = np.array([[1, 2], [3, 4]])
        batch1 = ConcreteBatch(
            data=data, counts=np.array([2]), edges=np.array([0, 2]), batch_size=1
        )
        batch2 = OtherBatch(
            data=data, counts=np.array([2]), edges=np.array([0, 2]), batch_size=1
        )

        assert batch1 != batch2

    def test_equality_with_nan_values(self):
        """Test equality handles NaN values correctly."""
        data1 = np.array([[1.0, np.nan], [3.0, 4.0]])
        data2 = np.array([[1.0, np.nan], [3.0, 4.0]])
        batch1 = ConcreteBatch(
            data=data1, counts=np.array([2]), edges=np.array([0, 2]), batch_size=1
        )
        batch2 = ConcreteBatch(
            data=data2, counts=np.array([2]), edges=np.array([0, 2]), batch_size=1
        )

        assert batch1 == batch2

    def test_equality_with_none_attributes(self):
        """Test equality with None attributes."""
        data = np.array([[1, 2]])
        batch1 = ConcreteBatch(
            data=data, counts=np.array([1]), edges=np.array([0, 1]), batch_size=1
        )
        batch2 = ConcreteBatch(
            data=data, counts=np.array([1]), edges=np.array([0, 1]), batch_size=1
        )

        # Manually set an attribute to None
        batch1.device = None
        batch2.device = None

        assert batch1 == batch2

    def test_inequality_one_none_one_not(self):
        """Test inequality when one attribute is None and other is not."""
        data = np.array([[1, 2]])
        batch1 = ConcreteBatch(
            data=data, counts=np.array([1]), edges=np.array([0, 1]), batch_size=1
        )
        batch2 = ConcreteBatch(
            data=data, counts=np.array([1]), edges=np.array([0, 1]), batch_size=1
        )

        # Manually set attributes
        batch1.device = None
        batch2.device = "cuda:0"

        assert batch1 != batch2

    def test_equality_scalar_attributes(self):
        """Test equality compares scalar attributes correctly."""
        data = np.array([[1, 2]])
        batch1 = ConcreteBatch(
            data=data, counts=np.array([1]), edges=np.array([0, 1]), batch_size=1
        )
        batch2 = ConcreteBatch(
            data=data, counts=np.array([1]), edges=np.array([0, 1]), batch_size=1
        )

        assert batch1.batch_size == batch2.batch_size
        assert batch1 == batch2


class TestBatchBaseProperties:
    """Test BatchBase properties."""

    def test_shape_property_array(self):
        """Test shape property for array data."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        batch = ConcreteBatch(
            data=data, counts=np.array([2, 1]), edges=np.array([0, 2, 3]), batch_size=2
        )

        assert batch.shape == (3, 3)

    def test_shape_property_list(self):
        """Test shape property for list data."""
        data = [np.array([1, 2]), np.array([3, 4, 5])]
        batch = ConcreteBatch(
            data=data, counts=np.array([2, 3]), edges=np.array([0, 2, 5]), batch_size=2
        )

        class MockList(list):
            dtype = np.dtype(np.int64)

        mock_data = MockList(data)
        batch2 = ConcreteBatch(
            data=mock_data,
            counts=np.array([2, 3]),
            edges=np.array([0, 2, 5]),
            batch_size=2,
            is_list=True,
        )

        assert batch.shape == (2,)

    def test_splits_property(self):
        """Test splits property returns internal edges."""
        data = np.array([1, 2, 3, 4, 5, 6])
        batch = ConcreteBatch(
            data=data,
            counts=np.array([2, 1, 3]),
            edges=np.array([0, 2, 3, 6]),
            batch_size=3,
        )

        splits = batch.splits
        np.testing.assert_array_equal(splits, [2, 3])

    def test_splits_property_single_entry(self):
        """Test splits property with single entry (no splits)."""
        data = np.array([1, 2, 3])
        batch = ConcreteBatch(
            data=data, counts=np.array([3]), edges=np.array([0, 3]), batch_size=1
        )

        splits = batch.splits
        assert len(splits) == 0


class TestBatchBaseGetCounts:
    """Test BatchBase get_counts method."""

    def test_get_counts_basic(self):
        """Test get_counts with basic batch IDs."""
        data = np.array([1, 2, 3, 4, 5])
        batch = ConcreteBatch(
            data=data,
            counts=np.array([2, 3]),
            edges=np.array([0, 2, 5]),
            batch_size=2,
        )

        batch_ids = np.array([0, 0, 1, 1, 1])
        counts = batch.get_counts(batch_ids, batch_size=2)

        np.testing.assert_array_equal(counts, [2, 3])

    def test_get_counts_with_empty_batches(self):
        """Test get_counts with some empty batches."""
        data = np.array([1, 2, 3])
        batch = ConcreteBatch(
            data=data,
            counts=np.array([0, 2, 0, 1]),
            edges=np.array([0, 0, 2, 2, 3]),
            batch_size=4,
        )

        batch_ids = np.array([1, 1, 3])
        counts = batch.get_counts(batch_ids, batch_size=4)

        np.testing.assert_array_equal(counts, [0, 2, 0, 1])

    def test_get_counts_empty_batch_ids(self):
        """Test get_counts with empty batch IDs."""
        data = np.array([])
        batch = ConcreteBatch(
            data=data, counts=np.array([0, 0]), edges=np.array([0, 0, 0]), batch_size=2
        )

        batch_ids = np.array([])
        counts = batch.get_counts(batch_ids, batch_size=2)

        np.testing.assert_array_equal(counts, [0, 0])


class TestBatchBaseGetEdges:
    """Test BatchBase get_edges method."""

    def test_get_edges_basic(self):
        """Test get_edges computes cumulative edges."""
        data = np.array([1, 2, 3, 4, 5, 6])
        batch = ConcreteBatch(
            data=data,
            counts=np.array([2, 1, 3]),
            edges=np.array([0, 2, 3, 6]),
            batch_size=3,
        )

        counts = np.array([2, 1, 3])
        edges = batch.get_edges(counts)

        np.testing.assert_array_equal(edges, [0, 2, 3, 6])

    def test_get_edges_all_same_size(self):
        """Test get_edges with uniform batch sizes."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        batch = ConcreteBatch(
            data=data,
            counts=np.array([3, 3, 3]),
            edges=np.array([0, 3, 6, 9]),
            batch_size=3,
        )

        counts = np.array([3, 3, 3])
        edges = batch.get_edges(counts)

        np.testing.assert_array_equal(edges, [0, 3, 6, 9])

    def test_get_edges_with_zeros(self):
        """Test get_edges with zero-sized batches."""
        data = np.array([1, 2])
        batch = ConcreteBatch(
            data=data,
            counts=np.array([0, 2, 0]),
            edges=np.array([0, 0, 2, 2]),
            batch_size=3,
        )

        counts = np.array([0, 2, 0])
        edges = batch.get_edges(counts)

        np.testing.assert_array_equal(edges, [0, 0, 2, 2])


class TestBatchBaseHelperMethods:
    """Test BatchBase helper methods (_zeros, _ones, etc.)."""

    def test_zeros_numpy(self):
        """Test _zeros creates numpy array."""
        data = np.array([1, 2, 3])
        batch = ConcreteBatch(
            data=data, counts=np.array([3]), edges=np.array([0, 3]), batch_size=1
        )

        zeros = batch._zeros(5)
        assert isinstance(zeros, np.ndarray)
        assert zeros.dtype == np.int64
        np.testing.assert_array_equal(zeros, [0, 0, 0, 0, 0])

    def test_ones_numpy(self):
        """Test _ones creates numpy array."""
        data = np.array([1, 2, 3])
        batch = ConcreteBatch(
            data=data, counts=np.array([3]), edges=np.array([0, 3]), batch_size=1
        )

        ones = batch._ones(4)
        assert isinstance(ones, np.ndarray)
        assert ones.dtype == np.int64
        np.testing.assert_array_equal(ones, [1, 1, 1, 1])

    def test_as_long_numpy(self):
        """Test _as_long converts to int64."""
        data = np.array([1, 2, 3])
        batch = ConcreteBatch(
            data=data, counts=np.array([3]), edges=np.array([0, 3]), batch_size=1
        )

        float_array = np.array([1.5, 2.7, 3.9])
        long_array = batch._as_long(float_array)

        assert long_array.dtype == np.int64
        np.testing.assert_array_equal(long_array, [1, 2, 3])

    def test_unique_numpy(self):
        """Test _unique returns unique values and counts."""
        data = np.array([1, 2, 3])
        batch = ConcreteBatch(
            data=data, counts=np.array([3]), edges=np.array([0, 3]), batch_size=1
        )

        values = np.array([0, 0, 1, 1, 1, 2])
        unique, counts = batch._unique(values)

        np.testing.assert_array_equal(unique, [0, 1, 2])
        np.testing.assert_array_equal(counts, [2, 3, 1])

    def test_transpose_numpy(self):
        """Test _transpose transposes array."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        batch = ConcreteBatch(
            data=data, counts=np.array([2]), edges=np.array([0, 2]), batch_size=1
        )

        transposed = batch._transpose(data)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(transposed, expected)

    def test_sum_numpy(self):
        """Test _sum computes sum."""
        data = np.array([1, 2, 3, 4, 5])
        batch = ConcreteBatch(
            data=data, counts=np.array([5]), edges=np.array([0, 5]), batch_size=1
        )

        total = batch._sum(data)
        assert total == 15

    def test_cumsum_numpy(self):
        """Test _cumsum computes cumulative sum."""
        data = np.array([1, 2, 3])
        batch = ConcreteBatch(
            data=data, counts=np.array([3]), edges=np.array([0, 3]), batch_size=1
        )

        cumsum = batch._cumsum(data)
        np.testing.assert_array_equal(cumsum, [1, 3, 6])

    def test_arange_numpy(self):
        """Test _arange creates range array."""
        data = np.array([1])
        batch = ConcreteBatch(
            data=data, counts=np.array([1]), edges=np.array([0, 1]), batch_size=1
        )

        arange = batch._arange(5)
        np.testing.assert_array_equal(arange, [0, 1, 2, 3, 4])

    def test_cat_numpy(self):
        """Test _cat concatenates arrays."""
        data = np.array([1])
        batch = ConcreteBatch(
            data=data, counts=np.array([1]), edges=np.array([0, 1]), batch_size=1
        )

        arrays = [np.array([1, 2]), np.array([3, 4]), np.array([5])]
        concatenated = batch._cat(arrays)
        np.testing.assert_array_equal(concatenated, [1, 2, 3, 4, 5])

    def test_split_numpy(self):
        """Test _split splits array."""
        data = np.array([1, 2, 3, 4, 5, 6])
        batch = ConcreteBatch(
            data=data, counts=np.array([6]), edges=np.array([0, 6]), batch_size=1
        )

        splits = batch._split(data, [2, 4])
        assert len(splits) == 3
        np.testing.assert_array_equal(splits[0], [1, 2])
        np.testing.assert_array_equal(splits[1], [3, 4])
        np.testing.assert_array_equal(splits[2], [5, 6])

    def test_stack_numpy(self):
        """Test _stack stacks arrays."""
        data = np.array([1])
        batch = ConcreteBatch(
            data=data, counts=np.array([1]), edges=np.array([0, 1]), batch_size=1
        )

        arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        stacked = batch._stack(arrays)
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(stacked, expected)

    def test_repeat_numpy(self):
        """Test _repeat repeats array elements."""
        data = np.array([1, 2, 3])
        batch = ConcreteBatch(
            data=data, counts=np.array([3]), edges=np.array([0, 3]), batch_size=1
        )

        repeated = batch._repeat(data, 2)
        np.testing.assert_array_equal(repeated, [1, 1, 2, 2, 3, 3])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestBatchBaseWithTorch:
    """Test BatchBase with PyTorch tensors."""

    def test_torch_initialization(self):
        """Test BatchBase detects torch tensors correctly."""
        data = torch.tensor([[1, 2], [3, 4], [5, 6]])
        batch = ConcreteBatch(
            data=data,
            counts=torch.tensor([3]),
            edges=torch.tensor([0, 3]),
            batch_size=1,
        )

        assert batch.is_numpy is False
        assert batch.is_sparse is False
        assert batch.device == data.device

    def test_torch_helper_methods(self):
        """Test helper methods work with torch tensors."""
        data = torch.tensor([1, 2, 3])
        batch = ConcreteBatch(
            data=data,
            counts=torch.tensor([3]),
            edges=torch.tensor([0, 3]),
            batch_size=1,
        )

        # Test _zeros
        zeros = batch._zeros(5)
        assert torch.equal(zeros, torch.zeros(5, dtype=torch.long))

        # Test _ones
        ones = batch._ones(3)
        assert torch.equal(ones, torch.ones(3, dtype=torch.long))

        # Test _arange
        arange = batch._arange(5)
        assert torch.equal(arange, torch.arange(5))

        # Test _unique
        data_dup = torch.tensor([1, 2, 2, 3, 3, 3])
        unique, counts = batch._unique(data_dup)
        assert torch.equal(unique, torch.tensor([1, 2, 3]))
        assert torch.equal(counts, torch.tensor([1, 2, 3]))

        # Test _sum
        assert batch._sum(data) == 6

        # Test _cumsum
        cumsum = batch._cumsum(data)
        assert torch.equal(cumsum, torch.tensor([1, 3, 6]))

    def test_torch_cat_and_split(self):
        """Test _cat and _split with torch tensors."""
        data = torch.tensor([1, 2, 3])
        batch = ConcreteBatch(
            data=data,
            counts=torch.tensor([3]),
            edges=torch.tensor([0, 3]),
            batch_size=1,
        )

        # Test _cat
        arrays = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        concatenated = batch._cat(arrays)
        assert torch.equal(concatenated, torch.tensor([1, 2, 3, 4, 5]))

        # Test _split
        splits = batch._split(concatenated, [2, 4])
        assert len(splits) == 3
        assert torch.equal(splits[0], torch.tensor([1, 2]))
        assert torch.equal(splits[1], torch.tensor([3, 4]))
        assert torch.equal(splits[2], torch.tensor([5]))

    def test_torch_stack_and_repeat(self):
        """Test _stack and _repeat with torch tensors."""
        data = torch.tensor([1, 2, 3])
        batch = ConcreteBatch(
            data=data,
            counts=torch.tensor([3]),
            edges=torch.tensor([0, 3]),
            batch_size=1,
        )

        # Test _stack
        arrays = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        stacked = batch._stack(arrays)
        expected = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert torch.equal(stacked, expected)

        # Test _repeat
        repeated = batch._repeat(data, 2)
        assert torch.equal(repeated, torch.tensor([1, 1, 2, 2, 3, 3]))

    def test_to_numpy_from_torch(self):
        """Test _to_numpy converts torch tensor to numpy."""
        data = torch.tensor([[1, 2], [3, 4]])
        batch = ConcreteBatch(
            data=data,
            counts=torch.tensor([2]),
            edges=torch.tensor([0, 2]),
            batch_size=1,
        )

        numpy_array = batch._to_numpy(data)
        assert isinstance(numpy_array, np.ndarray)
        np.testing.assert_array_equal(numpy_array, [[1, 2], [3, 4]])

    def test_to_tensor(self):
        """Test _to_tensor converts numpy to torch."""
        data = np.array([1, 2, 3])
        batch = ConcreteBatch(
            data=data,
            counts=np.array([3]),
            edges=np.array([0, 3]),
            batch_size=1,
        )

        tensor = batch._to_tensor(data)
        assert isinstance(tensor, torch.Tensor)
        assert torch.equal(tensor, torch.tensor([1, 2, 3]))

    def test_transpose_torch(self):
        """Test _transpose with torch tensors."""
        data = torch.tensor([[1, 2], [3, 4], [5, 6]])
        batch = ConcreteBatch(
            data=data,
            counts=torch.tensor([3]),
            edges=torch.tensor([0, 3]),
            batch_size=1,
        )

        transposed = batch._transpose(data)
        expected = torch.tensor([[1, 3, 5], [2, 4, 6]])
        assert torch.equal(transposed, expected)
