"""Comprehensive test suite for spine.data.batch.tensor module."""

from unittest.mock import Mock

import numpy as np
import pytest

from spine.data.batch.tensor import TensorBatch
from spine.utils.conditional import ME, ME_AVAILABLE, TORCH_AVAILABLE, torch


class TestTensorBatchInitialization:
    """Test TensorBatch initialization patterns."""

    def test_initialization_with_counts(self):
        """Test basic initialization with counts."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        counts = [2, 1]

        batch = TensorBatch(data, counts=counts)

        assert batch.batch_size == 2
        assert len(batch) == 2
        np.testing.assert_array_equal(batch.counts, [2, 1])
        np.testing.assert_array_equal(batch.edges, [0, 2, 3])
        assert batch.has_batch_col is False
        assert batch.coord_cols is None

    def test_initialization_with_batch_size_and_batch_col(self):
        """Test initialization with batch_size and has_batch_col."""
        data = np.array(
            [
                [0, 1, 2, 3],  # batch 0
                [0, 4, 5, 6],  # batch 0
                [1, 7, 8, 9],  # batch 1
            ]
        )

        batch = TensorBatch(data, batch_size=2, has_batch_col=True)

        assert batch.batch_size == 2
        np.testing.assert_array_equal(batch.counts, [2, 1])
        assert batch.has_batch_col is True

    def test_initialization_xor_counts_batch_size(self):
        """Test that either counts OR batch_size must be provided, not both."""
        data = np.array([[1, 2], [3, 4]])

        # Providing both should fail
        with pytest.raises(AssertionError, match="either `counts` or `batch_size`"):
            TensorBatch(data, counts=[2], batch_size=1)

        # Providing neither should fail
        with pytest.raises(AssertionError, match="either `counts` or `batch_size`"):
            TensorBatch(data)

    def test_initialization_counts_sum_validation(self):
        """Test that counts must sum to data length."""
        data = np.array([[1, 2], [3, 4], [5, 6]])

        # Invalid counts (sum = 5, but data has 3 rows)
        with pytest.raises(AssertionError, match="do not add up"):
            TensorBatch(data, counts=[2, 3])

    def test_initialization_batch_size_without_batch_col_fails(self):
        """Test that batch_size without has_batch_col fails."""
        data = np.array([[1, 2], [3, 4]])

        with pytest.raises(AssertionError, match="without a batch column"):
            TensorBatch(data, batch_size=2, has_batch_col=False)

    def test_initialization_with_coord_cols(self):
        """Test initialization with coordinate columns specified."""
        data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        counts = [2]
        coord_cols = [1, 2, 3]

        batch = TensorBatch(data, counts=counts, coord_cols=coord_cols)

        assert batch.coord_cols == [1, 2, 3]

    def test_initialization_single_entry(self):
        """Test initialization with single batch entry."""
        data = np.array([[1, 2], [3, 4]])
        counts = [2]

        batch = TensorBatch(data, counts=counts)

        assert batch.batch_size == 1
        assert len(batch) == 1

    def test_initialization_empty_tensor(self):
        """Test initialization with empty tensor."""
        data = np.array([]).reshape(0, 3)
        counts = []

        batch = TensorBatch(data, counts=counts)

        assert batch.batch_size == 0
        assert len(batch) == 0


class TestTensorBatchIndexing:
    """Test TensorBatch __getitem__ method."""

    def test_getitem_basic(self):
        """Test basic indexing into batch."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        counts = [2, 1]

        batch = TensorBatch(data, counts=counts)

        batch_0 = batch[0]
        batch_1 = batch[1]

        np.testing.assert_array_equal(batch_0, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(batch_1, [[5, 6]])

    def test_getitem_all_entries(self):
        """Test indexing all entries in sequence."""
        data = np.array([[1], [2], [3], [4], [5]])
        counts = [1, 2, 1, 1]

        batch = TensorBatch(data, counts=counts)

        assert len(batch[0]) == 1
        assert len(batch[1]) == 2
        assert len(batch[2]) == 1
        assert len(batch[3]) == 1

    def test_getitem_out_of_bounds(self):
        """Test indexing beyond batch_size raises IndexError."""
        data = np.array([[1, 2], [3, 4]])
        counts = [2]

        batch = TensorBatch(data, counts=counts)

        with pytest.raises(IndexError, match="Index 1 out of bound"):
            _ = batch[1]

        with pytest.raises(IndexError, match="Index 5 out of bound"):
            _ = batch[5]

    def test_getitem_single_element_batches(self):
        """Test indexing batches with single elements."""
        data = np.array([[1], [2], [3]])
        counts = [1, 1, 1]

        batch = TensorBatch(data, counts=counts)

        for i in range(3):
            batch_i = batch[i]
            assert batch_i.shape == (1, 1)


class TestTensorBatchProperties:
    """Test TensorBatch properties."""

    def test_tensor_property(self):
        """Test tensor property is alias for data."""
        data = np.array([[1, 2], [3, 4]])
        counts = [2]

        batch = TensorBatch(data, counts=counts)

        assert batch.tensor is batch.data
        np.testing.assert_array_equal(batch.tensor, data)

    def test_batch_ids_property(self):
        """Test batch_ids returns correct batch ID for each element."""
        data = np.array([[1], [2], [3], [4], [5]])
        counts = [2, 1, 2]

        batch = TensorBatch(data, counts=counts)

        batch_ids = batch.batch_ids
        np.testing.assert_array_equal(batch_ids, [0, 0, 1, 2, 2])

    def test_batch_ids_uniform_batches(self):
        """Test batch_ids with uniform batch sizes."""
        data = np.array([[1], [2], [3], [4], [5], [6]])
        counts = [2, 2, 2]

        batch = TensorBatch(data, counts=counts)

        batch_ids = batch.batch_ids
        np.testing.assert_array_equal(batch_ids, [0, 0, 1, 1, 2, 2])

    def test_batch_ids_single_batch(self):
        """Test batch_ids with single batch entry."""
        data = np.array([[1], [2], [3]])
        counts = [3]

        batch = TensorBatch(data, counts=counts)

        batch_ids = batch.batch_ids
        np.testing.assert_array_equal(batch_ids, [0, 0, 0])


class TestTensorBatchSplit:
    """Test TensorBatch split method."""

    def test_split_basic(self):
        """Test split breaks batch into list of tensors."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        counts = [2, 1]

        batch = TensorBatch(data, counts=counts)

        result = batch.split()

        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(result[1], [[5, 6]])

    def test_split_multiple_entries(self):
        """Test split with multiple entries."""
        data = np.array([[1], [2], [3], [4], [5], [6]])
        counts = [1, 2, 1, 2]

        batch = TensorBatch(data, counts=counts)

        result = batch.split()

        assert len(result) == 4
        assert len(result[0]) == 1
        assert len(result[1]) == 2
        assert len(result[2]) == 1
        assert len(result[3]) == 2

    def test_split_single_entry(self):
        """Test split with single batch entry."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        counts = [2]

        batch = TensorBatch(data, counts=counts)

        result = batch.split()

        assert len(result) == 1
        np.testing.assert_array_equal(result[0], data)


class TestTensorBatchApplyMask:
    """Test TensorBatch apply_mask method."""

    def test_apply_mask_basic(self):
        """Test apply_mask filters data and updates batching."""
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        counts = [2, 2]

        batch = TensorBatch(data, counts=counts)

        # Keep first and last element
        mask = np.array([True, False, False, True])
        batch.apply_mask(mask)

        assert len(batch.data) == 2
        np.testing.assert_array_equal(batch.data, [[1, 2], [7, 8]])
        np.testing.assert_array_equal(batch.counts, [1, 1])
        np.testing.assert_array_equal(batch.edges, [0, 1, 2])

    def test_apply_mask_all_true(self):
        """Test apply_mask with all True (no filtering)."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        counts = [2, 1]

        batch = TensorBatch(data, counts=counts)
        original_data = data.copy()

        mask = np.array([True, True, True])
        batch.apply_mask(mask)

        np.testing.assert_array_equal(batch.data, original_data)
        np.testing.assert_array_equal(batch.counts, [2, 1])

    def test_apply_mask_all_false(self):
        """Test apply_mask with all False (filter everything)."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        counts = [2, 1]

        batch = TensorBatch(data, counts=counts)

        mask = np.array([False, False, False])
        batch.apply_mask(mask)

        assert len(batch.data) == 0
        np.testing.assert_array_equal(batch.counts, [0, 0])
        np.testing.assert_array_equal(batch.edges, [0, 0, 0])

    def test_apply_mask_filter_entire_batch(self):
        """Test apply_mask filtering entire batch entries."""
        data = np.array([[1], [2], [3], [4]])
        counts = [2, 2]

        batch = TensorBatch(data, counts=counts)

        # Filter out first batch entirely
        mask = np.array([False, False, True, True])
        batch.apply_mask(mask)

        np.testing.assert_array_equal(batch.counts, [0, 2])


class TestTensorBatchMerge:
    """Test TensorBatch merge method."""

    def test_merge_basic(self):
        """Test merge interleaves two batches."""
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[5, 6], [7, 8]])
        counts1 = [1, 1]
        counts2 = [1, 1]

        batch1 = TensorBatch(data1, counts=counts1)
        batch2 = TensorBatch(data2, counts=counts2)

        merged = batch1.merge(batch2)

        # Should interleave: batch1[0], batch2[0], batch1[1], batch2[1]
        expected_data = np.array([[1, 2], [5, 6], [3, 4], [7, 8]])
        np.testing.assert_array_equal(merged.data, expected_data)
        np.testing.assert_array_equal(merged.counts, [2, 2])

    def test_merge_different_sizes(self):
        """Test merge with different sized batches."""
        data1 = np.array([[1], [2], [3]])
        data2 = np.array([[4], [5]])
        counts1 = [2, 1]
        counts2 = [1, 1]

        batch1 = TensorBatch(data1, counts=counts1)
        batch2 = TensorBatch(data2, counts=counts2)

        merged = batch1.merge(batch2)

        # batch1[0] has 2, batch2[0] has 1, batch1[1] has 1, batch2[1] has 1
        np.testing.assert_array_equal(merged.counts, [3, 2])
        assert merged.batch_size == 2

    def test_merge_preserves_order(self):
        """Test merge preserves entry-wise order."""
        data1 = np.array([[10], [20]])
        data2 = np.array([[100], [200]])
        counts1 = [1, 1]
        counts2 = [1, 1]

        batch1 = TensorBatch(data1, counts=counts1)
        batch2 = TensorBatch(data2, counts=counts2)

        merged = batch1.merge(batch2)

        # Should be: [10], [100], [20], [200]
        expected = np.array([[10], [100], [20], [200]])
        np.testing.assert_array_equal(merged.data, expected)


class TestTensorBatchTypeConversions:
    """Test TensorBatch to_numpy and to_tensor methods."""

    def test_to_numpy_already_numpy(self):
        """Test to_numpy on already numpy data is idempotent."""
        data = np.array([[1, 2], [3, 4]])
        counts = [2]

        batch = TensorBatch(data, counts=counts)
        result = batch.to_numpy()

        assert result is batch
        assert result.is_numpy is True

    def test_to_numpy_preserves_attributes(self):
        """Test to_numpy preserves has_batch_col and coord_cols."""
        data = np.array([[0, 1, 2, 3], [0, 4, 5, 6]])
        counts = [2]

        batch = TensorBatch(data, counts=counts, has_batch_col=True, coord_cols=[1, 2])
        result = batch.to_numpy()

        assert result.has_batch_col is True
        assert result.coord_cols == [1, 2]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_to_tensor_already_torch(self):
        """Test to_tensor on already torch data is idempotent."""
        data = torch.tensor([[1, 2], [3, 4]])
        counts = [2]

        batch = TensorBatch(data, counts=counts)
        result = batch.to_tensor()

        assert result is batch
        assert result.is_numpy is False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_to_tensor_from_numpy(self):
        """Test to_tensor converts numpy to torch."""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        counts = [2]

        batch = TensorBatch(data, counts=counts)
        result = batch.to_tensor()

        assert result.is_numpy is False
        assert isinstance(result.data, torch.Tensor)
        np.testing.assert_array_equal(result.data.numpy(), [[1, 2], [3, 4]])


class TestTensorBatchUnitConversions:
    """Test TensorBatch to_cm and to_px methods."""

    def test_to_cm_modifies_coord_cols(self):
        """Test to_cm converts coordinate columns."""
        data = np.array([[0, 10, 20, 30, 1.5], [0, 40, 50, 60, 2.5]], dtype=np.float64)
        counts = [2]

        # Mock meta object
        meta = Mock()
        meta.to_cm = Mock(
            return_value=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )  # type: ignore

        batch = TensorBatch(data, counts=counts)
        batch.to_cm(meta)

        # Verify to_cm was called
        meta.to_cm.assert_called_once()
        call_args = meta.to_cm.call_args
        # Check that coordinate columns were passed
        np.testing.assert_array_equal(call_args[0][0], [[10, 20, 30], [40, 50, 60]])

    def test_to_px_modifies_coord_cols(self):
        """Test to_px converts coordinate columns."""
        data = np.array(
            [[0, 1.5, 2.5, 3.5, 1.0], [0, 4.5, 5.5, 6.5, 2.0]], dtype=np.float64
        )
        counts = [2]

        # Mock meta object
        meta = Mock()
        meta.to_px = Mock(
            return_value=np.array([[10, 20, 30], [40, 50, 60]])
        )  # type: ignore

        batch = TensorBatch(data, counts=counts)
        batch.to_px(meta)

        # Verify to_px was called
        meta.to_px.assert_called_once()
        call_args = meta.to_px.call_args
        # Check that coordinate columns were passed
        np.testing.assert_array_equal(
            call_args[0][0], [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]
        )


class TestTensorBatchFromList:
    """Test TensorBatch from_list class method."""

    def test_from_list_basic(self):
        """Test from_list builds batch from list of tensors."""
        data_list = [np.array([[1, 2], [3, 4]]), np.array([[5, 6]])]

        batch = TensorBatch.from_list(data_list)

        assert batch.batch_size == 2
        np.testing.assert_array_equal(batch.counts, [2, 1])
        np.testing.assert_array_equal(batch.data, [[1, 2], [3, 4], [5, 6]])

    def test_from_list_single_tensor(self):
        """Test from_list with single tensor in list."""
        data_list = [np.array([[1], [2], [3]])]

        batch = TensorBatch.from_list(data_list)

        assert batch.batch_size == 1
        np.testing.assert_array_equal(batch.counts, [3])

    def test_from_list_empty_raises(self):
        """Test from_list with empty list raises error."""
        data_list = []  # type: ignore

        with pytest.raises(AssertionError, match="at least one tensor"):
            TensorBatch.from_list(data_list)

    def test_from_list_various_sizes(self):
        """Test from_list with tensors of various sizes."""
        data_list = [
            np.array([[1]]),
            np.array([[2], [3], [4]]),
            np.array([[5], [6]]),
        ]

        batch = TensorBatch.from_list(data_list)

        assert batch.batch_size == 3
        np.testing.assert_array_equal(batch.counts, [1, 3, 2])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_from_list_with_torch(self):
        """Test from_list with torch tensors."""
        data_list = [
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6]]),
        ]

        batch = TensorBatch.from_list(data_list)

        assert batch.batch_size == 2
        np.testing.assert_array_equal(batch.counts, [2, 1])
        assert isinstance(batch.data, torch.Tensor)
        np.testing.assert_array_equal(batch.data.numpy(), [[1, 2], [3, 4], [5, 6]])


class TestTensorBatchEdgeCases:
    """Test TensorBatch edge cases."""

    def test_empty_batches_in_middle(self):
        """Test batch with empty entries in the middle."""
        data = np.array([[1, 2], [3, 4]])
        counts = [1, 0, 1]

        batch = TensorBatch(data, counts=counts)

        assert batch.batch_size == 3
        assert len(batch[0]) == 1
        assert len(batch[1]) == 0  # Empty batch
        assert len(batch[2]) == 1

    def test_large_batch(self):
        """Test with large batch size."""
        data = np.random.rand(1000, 5).astype(np.float32)
        counts = [10] * 100  # 100 batches of 10 elements

        batch = TensorBatch(data, counts=counts)

        assert batch.batch_size == 100
        assert len(batch.data) == 1000

        # Test random access
        batch_50 = batch[50]
        assert len(batch_50) == 10

    def test_single_element_batches(self):
        """Test batch where each entry is single element."""
        data = np.array([[1], [2], [3], [4], [5]])
        counts = [1, 1, 1, 1, 1]

        batch = TensorBatch(data, counts=counts)

        assert batch.batch_size == 5
        for i in range(5):
            assert len(batch[i]) == 1

    def test_datatype_preservation(self):
        """Test that data types are preserved through operations."""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        counts = [2]

        batch = TensorBatch(data, counts=counts)

        assert batch.dtype == np.float32
        assert batch.data.dtype == np.float32

        # After split
        split_data = batch.split()
        assert split_data[0].dtype == np.float32


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestTensorBatchWithTorch:
    """Test TensorBatch with PyTorch tensors."""

    def test_torch_initialization(self):
        """Test TensorBatch with torch tensors."""
        data = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
        counts = [3]

        batch = TensorBatch(data, counts=counts)

        assert batch.is_numpy is False
        assert isinstance(batch.data, torch.Tensor)
        assert batch.batch_size == 1

    def test_torch_with_batch_col(self):
        """Test TensorBatch with torch tensor and batch column."""
        data = torch.tensor([[0, 1, 2], [0, 3, 4], [1, 5, 6], [1, 7, 8]])

        batch = TensorBatch(data, batch_size=2, has_batch_col=True)

        assert batch.batch_size == 2
        assert batch.has_batch_col is True
        assert torch.equal(batch.counts, torch.tensor([2, 2]))

    def test_torch_indexing(self):
        """Test indexing with torch tensors."""
        data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        counts = [2, 2]

        batch = TensorBatch(data, counts=counts)

        # Test single index
        batch_0 = batch[0]
        assert torch.equal(batch_0.data, torch.tensor([[1, 2], [3, 4]]))

        batch_1 = batch[1]
        assert torch.equal(batch_1.data, torch.tensor([[5, 6], [7, 8]]))

    def test_torch_split(self):
        """Test split method with torch tensors."""
        data = torch.tensor([[1, 2], [3, 4], [5, 6]])
        counts = [1, 2]

        batch = TensorBatch(data, counts=counts)
        split_data = batch.split()

        assert len(split_data) == 2
        assert torch.equal(split_data[0], torch.tensor([[1, 2]]))
        assert torch.equal(split_data[1], torch.tensor([[3, 4], [5, 6]]))

    def test_torch_apply_mask(self):
        """Test apply_mask with torch tensors."""
        data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        counts = [2, 2]

        batch = TensorBatch(data, counts=counts)
        mask = torch.tensor([True, False, True, False])

        batch.apply_mask(mask)

        assert torch.equal(batch.data, torch.tensor([[1, 2], [5, 6]]))
        assert torch.equal(batch.counts, torch.tensor([1, 1]))

    def test_torch_merge(self):
        """Test merge method with torch tensors."""
        data1 = torch.tensor([[1, 2], [3, 4]])
        data2 = torch.tensor([[5, 6], [7, 8]])
        counts1 = [1, 1]
        counts2 = [1, 1]

        batch1 = TensorBatch(data1, counts=counts1)
        batch2 = TensorBatch(data2, counts=counts2)

        merged = batch1.merge(batch2)

        assert torch.equal(merged.data, torch.tensor([[1, 2], [5, 6], [3, 4], [7, 8]]))
        assert torch.equal(merged.counts, torch.tensor([2, 2]))

    def test_torch_batch_ids(self):
        """Test batch_ids property with torch tensors."""
        data = torch.tensor([[1, 2], [3, 4], [5, 6]])
        counts = [2, 1]

        batch = TensorBatch(data, counts=counts)
        batch_ids = batch.batch_ids

        assert torch.equal(batch_ids, torch.tensor([0, 0, 1]))

    def test_to_tensor_idempotent(self):
        """Test to_tensor on already torch data is idempotent."""
        data = torch.tensor([[1, 2], [3, 4]])
        counts = [2]

        batch = TensorBatch(data, counts=counts)
        result = batch.to_tensor()

        assert result is batch
        assert result.is_numpy is False

    def test_to_numpy_from_torch(self):
        """Test to_numpy converts torch to numpy."""
        data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        counts = [2]

        batch = TensorBatch(data, counts=counts)
        result = batch.to_numpy()

        assert result.is_numpy is True
        assert isinstance(result.data, np.ndarray)
        np.testing.assert_array_equal(result.data, [[1, 2], [3, 4]])

    def test_unit_conversions_with_torch(self):
        """Test to_cm and to_px with torch tensors."""
        data = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.float32)
        counts = [2]

        batch = TensorBatch(data, counts=counts, has_batch_col=True, coord_cols=[1, 2])

        # Mock Meta object
        meta = Mock()
        meta.size = Mock(return_value=0.5)  # 0.5 cm per pixel

        with pytest.raises(AssertionError, match="numpy arrays"):
            batch.to_cm(meta)

        with pytest.raises(AssertionError, match="numpy arrays"):
            batch.to_px(meta)


@pytest.mark.skipif(not ME_AVAILABLE, reason="ME not available")
class TestTensorBatchWithME:
    """Test TensorBatch with MinkowskiEngine tensors."""

    def test_me_initialization(self):
        """Test TensorBatch with ME tensors."""
        data = ME.SparseTensor(
            features=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32),
            coordinates=torch.tensor(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.int32
            ),
        )
        counts = [3]

        batch = TensorBatch(data, counts=counts, is_sparse=True)

        assert batch.is_numpy is False
        assert isinstance(batch.data, ME.SparseTensor)
        assert batch.batch_size == 1
        assert batch.device == torch.device("cpu")

    def test_to_numpy_from_me(self):
        """Test to_numpy converts ME SparseTensor to numpy."""
        data = ME.SparseTensor(
            features=torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
            coordinates=torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.int32),
        )
        counts = [2]

        batch = TensorBatch(data, counts=counts, is_sparse=True)
        result = batch.to_numpy()

        assert result.is_numpy is True
        assert isinstance(result.data, np.ndarray)
        # The data should be concatenation of coordinates and features
        expected = np.array([[0, 0, 0, 1, 2], [0, 1, 0, 3, 4]])
        np.testing.assert_array_equal(result.data, expected)

    def test_getitem_with_me(self):
        """Test indexing with ME SparseTensor."""
        data = ME.SparseTensor(
            features=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32),
            coordinates=torch.tensor(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.int32
            ),
        )
        counts = [2, 1]

        batch = TensorBatch(data, counts=counts, is_sparse=True)

        batch_0 = batch[0]
        batch_1 = batch[1]

        # batch_0 should contain the first two entries
        expected_0 = np.array([[0, 0, 0, 1, 2], [0, 1, 0, 3, 4]])
        np.testing.assert_array_equal(batch_0.C.numpy(), expected_0[:, :3])
        np.testing.assert_array_equal(batch_0.F.numpy(), expected_0[:, 3:])

        # batch_1 should contain the last entry
        expected_1 = np.array([[1, 0, 0, 5, 6]])
        np.testing.assert_array_equal(batch_1.C.numpy(), expected_1[:, :3])
        np.testing.assert_array_equal(batch_1.F.numpy(), expected_1[:, 3:])

    def test_split_with_me(self):
        """Test split method with ME SparseTensor."""
        data = ME.SparseTensor(
            features=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32),
            coordinates=torch.tensor(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=torch.int32
            ),
        )
        counts = [1, 2]

        batch = TensorBatch(data, counts=counts, is_sparse=True)
        split_data = batch.split()

        assert len(split_data) == 2

        expected_0 = np.array([[0, 0, 0, 1, 2]])
        np.testing.assert_array_equal(split_data[0].C.numpy(), expected_0[:, :3])
        np.testing.assert_array_equal(split_data[0].F.numpy(), expected_0[:, 3:])

        expected_1 = np.array([[0, 1, 0, 3, 4], [1, 0, 0, 5, 6]])
        np.testing.assert_array_equal(split_data[1].C.numpy(), expected_1[:, :3])
        np.testing.assert_array_equal(split_data[1].F.numpy(), expected_1[:, 3:])
