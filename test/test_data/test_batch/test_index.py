"""Comprehensive test suite for spine.data.batch.index module."""

import numpy as np
import pytest

from spine.data.batch.index import IndexBatch
from spine.utils.conditional import TORCH_AVAILABLE, torch


class TestIndexBatchInitialization:
    """Test IndexBatch initialization patterns."""

    def test_initialization_single_index_with_counts(self):
        """Test initialization with single index and counts."""
        data = np.array([0, 1, 10, 11, 12])
        offsets = np.array([0, 10])
        counts = [2, 3]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        assert batch.batch_size == 2
        assert batch.is_list is False
        np.testing.assert_array_equal(batch.counts, [2, 3])
        np.testing.assert_array_equal(batch.single_counts, [2, 3])
        np.testing.assert_array_equal(batch.offsets, [0, 10])

    def test_initialization_with_batch_ids(self):
        """Test initialization using batch_ids instead of counts."""
        data = np.array([5, 6, 7, 15, 16])
        offsets = np.array([0, 10])
        batch_ids = np.array([0, 0, 0, 1, 1])
        batch_size = 2

        batch = IndexBatch(
            data, offsets=offsets, batch_ids=batch_ids, batch_size=batch_size
        )

        assert batch.batch_size == 2
        np.testing.assert_array_equal(batch.counts, [3, 2])

    def test_initialization_index_list_with_single_counts(self):
        """Test initialization with list of indexes."""
        data = [np.array([0, 1]), np.array([2, 3, 4]), np.array([10, 11])]
        offsets = np.array([0, 10])  # Fixed: must match counts length
        counts = [2, 1]
        single_counts = [2, 3, 2]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        assert batch.batch_size == 2
        assert batch.is_list is True
        np.testing.assert_array_equal(batch.counts, [2, 1])
        np.testing.assert_array_equal(batch.single_counts, [2, 3, 2])

    def test_initialization_batch_ids_and_batch_size_required(self):
        """Test that batch_ids requires batch_size."""
        data = np.array([1, 2, 3])
        offsets = np.array([0, 10])
        batch_ids = np.array([0, 0, 1])

        # Missing batch_size
        with pytest.raises(ValueError, match="batch_size"):
            IndexBatch(data, offsets=offsets, batch_ids=batch_ids)

    def test_initialization_index_list_requires_single_counts(self):
        """Test that index list requires single_counts."""
        data = [np.array([0, 1]), np.array([2, 3])]
        offsets = np.array([0, 10])
        counts = [1, 1]

        with pytest.raises(ValueError, match="provide `single_counts`"):
            IndexBatch(data, offsets=offsets, counts=counts)

    def test_initialization_counts_sum_validation(self):
        """Test that counts must sum to data length."""
        data = np.array([1, 2, 3, 4, 5])
        offsets = np.array([0, 10])
        counts = [2, 2]  # Sum is 4, but data has 5 elements

        with pytest.raises(ValueError, match="add up"):
            IndexBatch(data, offsets=offsets, counts=counts)

    def test_initialization_counts_offsets_length_match(self):
        """Test that counts and offsets must have same length."""
        data = np.array([1, 2, 3])
        offsets = np.array([0, 10])
        counts = [3]  # Only 1 count, but 2 offsets

        with pytest.raises(ValueError, match="match the number"):
            IndexBatch(data, offsets=offsets, counts=counts)

    def test_initialization_single_counts_length_validation(self):
        """Test that single_counts length must match data length."""
        data = [np.array([0, 1]), np.array([2, 3])]
        offsets = np.array([0, 10])
        counts = [2]
        single_counts = [2]  # Only 1, but data has 2 indexes

        with pytest.raises(ValueError, match="one single count per index"):
            IndexBatch(
                data, offsets=offsets, counts=counts, single_counts=single_counts
            )

    def test_initialization_empty_list_with_default(self):
        """Test initialization with empty list and default."""
        data = []
        offsets = np.array([0, 0])
        counts = [0, 0]
        single_counts = []
        default = np.empty(0, dtype=np.int64)

        batch = IndexBatch(
            data,
            offsets=offsets,
            counts=counts,
            single_counts=single_counts,
            default=default,
        )

        assert batch.batch_size == 2
        assert batch.is_list is True

    def test_initialization_empty_list_without_default_warns(self):
        """Test initialization with empty list without default warns."""
        data = []
        offsets = np.array([0])
        counts = [0]
        single_counts = []

        with pytest.warns(UserWarning, match="empty list without a default"):
            batch = IndexBatch(
                data, offsets=offsets, counts=counts, single_counts=single_counts
            )

        assert batch.is_list is True


class TestIndexBatchIndexing:
    """Test IndexBatch __getitem__ method."""

    def test_getitem_single_index(self):
        """Test indexing with single index."""
        data = np.array([0, 1, 10, 11, 12])
        offsets = np.array([0, 10])
        counts = [2, 3]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        batch_0 = batch[0]
        batch_1 = batch[1]

        # Offsets are subtracted
        np.testing.assert_array_equal(batch_0, [0, 1])
        np.testing.assert_array_equal(batch_1, [0, 1, 2])  # 10-10, 11-10, 12-10

    def test_getitem_index_list(self):
        """Test indexing with index list."""
        data = [np.array([0, 1]), np.array([2, 3]), np.array([10, 11])]
        offsets = np.array([0, 10])  # Fixed: must match counts length
        counts = [2, 1]
        single_counts = [2, 2, 2]  # Fixed: 3 indexes

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        batch_0 = batch[0]
        batch_1 = batch[1]

        assert len(batch_0) == 2
        assert len(batch_1) == 1

        # Check offset subtraction
        np.testing.assert_array_equal(batch_0[0], [0, 1])
        np.testing.assert_array_equal(batch_0[1], [2, 3])
        np.testing.assert_array_equal(batch_1[0], [0, 1])  # 10-10, 11-10

    def test_getitem_out_of_bounds(self):
        """Test indexing beyond batch_size raises IndexError."""
        data = np.array([1, 2, 3])
        offsets = np.array([0])
        counts = [3]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        with pytest.raises(IndexError, match="out of bound"):
            _ = batch[1]

    def test_getitem_empty_batch_entry(self):
        """Test indexing empty batch entry."""
        data = np.array([1, 2])
        offsets = np.array([0, 10, 20])
        counts = [0, 2, 0]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        batch_0 = batch[0]
        batch_1 = batch[1]
        batch_2 = batch[2]

        assert len(batch_0) == 0
        assert len(batch_1) == 2
        assert len(batch_2) == 0


class TestIndexBatchProperties:
    """Test IndexBatch properties."""

    def test_index_property_single_index(self):
        """Test index property for single index."""
        data = np.array([0, 1, 2, 3])
        offsets = np.array([0, 10])
        counts = [2, 2]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        assert batch.index is batch.data
        np.testing.assert_array_equal(batch.index, data)

    def test_index_property_fails_for_list(self):
        """Test index property raises error for index list."""
        data = [np.array([0, 1]), np.array([2, 3])]
        offsets = np.array([0, 0])  # Fixed: 2 batches
        counts = [1, 1]  # Fixed: 2 batches
        single_counts = [2, 2]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        with pytest.raises(ValueError, match="not a single index"):
            _ = batch.index

    def test_index_list_property(self):
        """Test index_list property for index list."""
        data = [np.array([0, 1]), np.array([2, 3])]
        offsets = np.array([0, 0])  # Fixed: 2 batches
        counts = [1, 1]  # Fixed: 2 batches
        single_counts = [2, 2]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        assert batch.index_list is batch.data

    def test_index_list_property_fails_for_single(self):
        """Test index_list property raises error for single index."""
        data = np.array([0, 1, 2])
        offsets = np.array([0])
        counts = [3]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        with pytest.raises(ValueError, match="single index"):
            _ = batch.index_list

    def test_full_index_single(self):
        """Test full_index for single index."""
        data = np.array([0, 1, 2, 3])
        offsets = np.array([0, 10])
        counts = [2, 2]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        np.testing.assert_array_equal(batch.full_index, data)

    def test_full_index_list(self):
        """Test full_index concatenates index list."""
        data = [np.array([0, 1]), np.array([2, 3, 4]), np.array([5])]
        offsets = np.array([0, 0, 0])  # Fixed: 3 batches
        counts = [1, 1, 1]  # Fixed: 3 batches
        single_counts = [2, 3, 1]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        full = batch.full_index
        np.testing.assert_array_equal(full, [0, 1, 2, 3, 4, 5])

    def test_full_index_empty_list(self):
        """Test full_index for empty list."""
        data = []
        offsets = np.array([0])
        counts = [0]
        single_counts = []
        default = np.empty(0, dtype=np.int64)

        batch = IndexBatch(
            data,
            offsets=offsets,
            counts=counts,
            single_counts=single_counts,
            default=default,
        )

        full = batch.full_index
        assert len(full) == 0

    def test_index_ids_property(self):
        """Test index_ids returns ID for each element."""
        data = [np.array([0, 1]), np.array([2, 3, 4]), np.array([5])]
        offsets = np.array([0])
        counts = [3]
        single_counts = [2, 3, 1]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        index_ids = batch.index_ids
        # 2 elements from index 0, 3 from index 1, 1 from index 2
        np.testing.assert_array_equal(index_ids, [0, 0, 1, 1, 1, 2])

    def test_index_ids_fails_for_single_index(self):
        """Test index_ids raises error for single index."""
        data = np.array([0, 1, 2])
        offsets = np.array([0])
        counts = [3]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        with pytest.raises(ValueError, match="list of index"):
            _ = batch.index_ids

    def test_full_counts_single_index(self):
        """Test full_counts for single index equals counts."""
        data = np.array([0, 1, 2, 3, 4])
        offsets = np.array([0, 10])
        counts = [2, 3]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        np.testing.assert_array_equal(batch.full_counts, batch.counts)

    def test_full_counts_index_list(self):
        """Test full_counts sums single_counts per batch."""
        data = [np.array([0, 1]), np.array([2, 3, 4]), np.array([5, 6])]
        offsets = np.array([0, 0])
        counts = [2, 1]
        single_counts = [2, 3, 2]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        full_counts = batch.full_counts
        # Batch 0 has indexes 0,1: 2+3=5 elements
        # Batch 1 has index 2: 2 elements
        np.testing.assert_array_equal(full_counts, [5, 2])

    def test_batch_ids_property(self):
        """Test batch_ids returns batch ID per index."""
        data = [np.array([0]), np.array([1]), np.array([2]), np.array([3])]
        offsets = np.array([0, 0])
        counts = [2, 2]
        single_counts = [1, 1, 1, 1]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        batch_ids = batch.batch_ids
        np.testing.assert_array_equal(batch_ids, [0, 0, 1, 1])

    def test_full_batch_ids_property(self):
        """Test full_batch_ids returns batch ID per element."""
        data = [np.array([0, 1]), np.array([2, 3, 4]), np.array([5, 6])]
        offsets = np.array([0, 0])
        counts = [2, 1]
        single_counts = [2, 3, 2]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        full_batch_ids = batch.full_batch_ids
        # Batch 0: 2+3=5 elements, Batch 1: 2 elements
        np.testing.assert_array_equal(full_batch_ids, [0, 0, 0, 0, 0, 1, 1])


class TestIndexBatchSplit:
    """Test IndexBatch split method."""

    def test_split_single_index(self):
        """Test split with single index."""
        data = np.array([0, 1, 10, 11, 12])
        offsets = np.array([0, 10])
        counts = [2, 3]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        result = batch.split()

        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [0, 1])
        np.testing.assert_array_equal(result[1], [0, 1, 2])

    def test_split_index_list(self):
        """Test split with index list."""
        data = [np.array([0, 1]), np.array([2, 3]), np.array([10, 11])]
        offsets = np.array([0, 10])  # Fixed: 2 batches
        counts = [2, 1]
        single_counts = [2, 2, 2]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        result = batch.split()

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1


class TestIndexBatchMerge:
    """Test IndexBatch merge method."""

    def test_merge_single_index(self):
        """Test merge with single indexes."""
        data1 = np.array([0, 1, 10, 11])
        data2 = np.array([2, 3, 12])
        offsets = np.array([0, 10])
        counts1 = [2, 2]
        counts2 = [2, 1]

        batch1 = IndexBatch(data1, offsets=offsets, counts=counts1)
        batch2 = IndexBatch(data2, offsets=offsets, counts=counts2)

        merged = batch1.merge(batch2)

        assert merged.batch_size == 2
        np.testing.assert_array_equal(merged.counts, [4, 3])

    def test_merge_index_list(self):
        """Test merge with index lists."""
        data1 = [np.array([0]), np.array([10])]
        data2 = [np.array([1]), np.array([11])]
        offsets = np.array([0, 10])
        counts1 = [1, 1]
        counts2 = [1, 1]
        single_counts = [1, 1]

        batch1 = IndexBatch(
            data1, offsets=offsets, counts=counts1, single_counts=single_counts
        )
        batch2 = IndexBatch(
            data2, offsets=offsets, counts=counts2, single_counts=single_counts
        )

        merged = batch1.merge(batch2)

        assert merged.batch_size == 2
        np.testing.assert_array_equal(merged.counts, [2, 2])

    def test_merge_mismatched_offsets_fails(self):
        """Test merge fails with mismatched offsets."""
        data1 = np.array([0, 1])
        data2 = np.array([2, 3])
        offsets1 = np.array([0, 10])
        offsets2 = np.array([0, 20])  # Different!
        counts = [1, 1]

        batch1 = IndexBatch(data1, offsets=offsets1, counts=counts)
        batch2 = IndexBatch(data2, offsets=offsets2, counts=counts)

        with pytest.raises(ValueError, match="same tensor"):
            batch1.merge(batch2)


class TestIndexBatchTypeConversions:
    """Test IndexBatch to_numpy and to_tensor methods."""

    def test_to_numpy_already_numpy_single_index(self):
        """Test to_numpy on already numpy single index is idempotent."""
        data = np.array([0, 1, 2])
        offsets = np.array([0])
        counts = [3]

        batch = IndexBatch(data, offsets=offsets, counts=counts)
        result = batch.to_numpy()

        assert result is batch

    def test_to_numpy_already_numpy_index_list(self):
        """Test to_numpy on already numpy index list is idempotent."""
        data = [np.array([0, 1]), np.array([2, 3])]
        offsets = np.array([0])
        counts = [2]
        single_counts = [2, 2]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )
        result = batch.to_numpy()

        assert result is batch

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_to_tensor_single_index(self):
        """Test to_tensor converts single index to torch tensor."""
        data = np.array([0, 1, 2])
        offsets = np.array([0])
        counts = [3]

        batch = IndexBatch(data, offsets=offsets, counts=counts)
        result = batch.to_tensor()

        assert isinstance(result.data, torch.Tensor)
        np.testing.assert_array_equal(result.data.cpu().numpy(), data)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_to_tensor_index_list(self):
        """Test to_tensor converts index list to list of torch tensors."""
        data = [np.array([0, 1]), np.array([2, 3])]
        offsets = np.array([0])
        counts = [2]
        single_counts = [2, 2]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )
        result = batch.to_tensor()

        assert isinstance(result.data, list)
        assert all(isinstance(x, torch.Tensor) for x in result.data)
        np.testing.assert_array_equal(result.data[0].cpu().numpy(), [0, 1])
        np.testing.assert_array_equal(result.data[1].cpu().numpy(), [2, 3])


class TestIndexBatchEdgeCases:
    """Test IndexBatch edge cases."""

    def test_empty_batch_entries(self):
        """Test batch with empty entries."""
        data = np.array([1, 2])
        offsets = np.array([0, 10, 20])
        counts = [0, 2, 0]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        assert batch.batch_size == 3
        np.testing.assert_array_equal(batch.counts, [0, 2, 0])

    def test_single_element_indexes(self):
        """Test with single element per index."""
        data = np.array([0, 10, 20])
        offsets = np.array([0, 10, 20])
        counts = [1, 1, 1]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        assert batch.batch_size == 3
        for i in range(3):
            assert len(batch[i]) == 1

    def test_large_offsets(self):
        """Test with large offset values."""
        data = np.array([100000, 100001, 200000, 200001])
        offsets = np.array([100000, 200000])
        counts = [2, 2]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        batch_0 = batch[0]
        batch_1 = batch[1]

        np.testing.assert_array_equal(batch_0, [0, 1])
        np.testing.assert_array_equal(batch_1, [0, 1])

    def test_zero_offsets(self):
        """Test with all zero offsets."""
        data = np.array([0, 1, 2, 3, 4])
        offsets = np.array([0, 0, 0])
        counts = [2, 2, 1]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        assert batch.batch_size == 3
        # All should have same values since offset is 0
        batch_0 = batch[0]
        np.testing.assert_array_equal(batch_0, [0, 1])

    def test_list_with_variable_index_sizes(self):
        """Test index list with varying sizes per index."""
        data = [
            np.array([0]),
            np.array([1, 2, 3, 4]),
            np.array([5, 6]),
            np.array([7]),
        ]
        offsets = np.array([0, 0])
        counts = [3, 1]
        single_counts = [1, 4, 2, 1]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        # Batch 0 has indexes 0,1,2: 1+4+2=7 total elements
        # Batch 1 has index 3: 1 element
        full_counts = batch.full_counts
        np.testing.assert_array_equal(full_counts, [7, 1])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestIndexBatchWithTorch:
    """Test IndexBatch with PyTorch tensors."""

    def test_torch_single_index_initialization(self):
        """Test IndexBatch with torch tensor for single index."""
        data = torch.tensor([0, 1, 2, 3, 4])
        offsets = torch.tensor([0, 10])
        counts = [3, 2]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        assert batch.is_numpy is False
        assert batch.is_list is False
        assert isinstance(batch.data, torch.Tensor)
        assert batch.batch_size == 2

    def test_torch_index_list(self):
        """Test IndexBatch with list of torch tensors."""
        data = [
            torch.tensor([0, 1, 2]),
            torch.tensor([3, 4]),
            torch.tensor([5]),
        ]
        offsets = torch.tensor([0, 10])
        counts = [2, 1]
        single_counts = [3, 2, 1]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        assert batch.is_list is True
        assert batch.batch_size == 2
        assert torch.equal(batch.single_counts, torch.tensor([3, 2, 1]))

    def test_torch_indexing(self):
        """Test indexing with torch tensors."""
        data = torch.tensor([0, 1, 2, 3, 4])
        offsets = torch.tensor([0, 10])
        counts = [3, 2]

        batch = IndexBatch(data, offsets=offsets, counts=counts)

        # Test single index
        batch_0 = batch[0]
        assert torch.equal(batch_0, torch.tensor([0, 1, 2]))

        batch_1 = batch[1]
        assert torch.equal(batch_1, torch.tensor([-7, -6]))  # With offset

    def test_torch_split(self):
        """Test split method with torch tensors."""
        data = torch.tensor([0, 1, 2, 3, 4])
        offsets = torch.tensor([0, 10])
        counts = [3, 2]

        batch = IndexBatch(data, offsets=offsets, counts=counts)
        split_data = batch.split()

        assert len(split_data) == 2
        assert torch.equal(split_data[0], torch.tensor([0, 1, 2]))
        assert torch.equal(split_data[1], torch.tensor([-7, -6]))

    def test_torch_batch_ids(self):
        """Test batch_ids with torch tensors."""
        data = torch.tensor([0, 1, 2, 3, 4])
        offsets = torch.tensor([0, 10])
        counts = [3, 2]

        batch = IndexBatch(data, offsets=offsets, counts=counts)
        batch_ids = batch.batch_ids

        assert torch.equal(batch_ids, torch.tensor([0, 0, 0, 1, 1]))

    def test_torch_full_index(self):
        """Test full_index property with torch index list."""
        data = [
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
        ]
        offsets = torch.tensor([0, 10])
        counts = [1, 1]
        single_counts = [2, 2]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )

        full_index = batch.full_index
        assert torch.equal(full_index, torch.tensor([0, 1, 2, 3]))

    def test_to_tensor_idempotent(self):
        """Test to_tensor on already torch data is idempotent."""
        data = torch.tensor([0, 1, 2])
        offsets = torch.tensor([0])
        counts = [3]

        batch = IndexBatch(data, offsets=offsets, counts=counts)
        result = batch.to_tensor()

        assert result is batch
        assert result.is_numpy is False

    def test_to_numpy_from_torch(self):
        """Test to_numpy converts torch to numpy."""
        data = torch.tensor([0, 1, 2, 3])
        offsets = torch.tensor([0, 10])
        counts = [2, 2]

        batch = IndexBatch(data, offsets=offsets, counts=counts)
        result = batch.to_numpy()

        assert result.is_numpy is True
        assert isinstance(result.data, np.ndarray)
        np.testing.assert_array_equal(result.data, [0, 1, 2, 3])

    def test_to_numpy_from_torch_index_list(self):
        """Test to_numpy converts torch index list to numpy."""
        data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        offsets = torch.tensor([0])
        counts = [2]
        single_counts = [2, 2]

        batch = IndexBatch(
            data, offsets=offsets, counts=counts, single_counts=single_counts
        )
        result = batch.to_numpy()

        assert result.is_numpy is True
        assert isinstance(result.data, list)
        assert all(isinstance(x, np.ndarray) for x in result.data)
        np.testing.assert_array_equal(result.data[0], [0, 1])
        np.testing.assert_array_equal(result.data[1], [2, 3])

    def test_torch_merge_index_list(self):
        """Test merge with torch index lists."""
        data1 = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        data2 = [torch.tensor([4, 5]), torch.tensor([6, 7])]
        offsets1 = torch.tensor([0, 10])
        offsets2 = torch.tensor([0, 10])
        counts1 = [1, 1]
        counts2 = [1, 1]
        single_counts1 = [2, 2]
        single_counts2 = [2, 2]

        batch1 = IndexBatch(
            data1, offsets=offsets1, counts=counts1, single_counts=single_counts1
        )
        batch2 = IndexBatch(
            data2, offsets=offsets2, counts=counts2, single_counts=single_counts2
        )

        merged = batch1.merge(batch2)

        assert merged.is_list is True
        assert merged.batch_size == 2
        # Merged interleaves: index 0 from batch1, index 0 from batch2, index 1 from batch1, index 1 from batch2
        assert len(merged.data) == 4
