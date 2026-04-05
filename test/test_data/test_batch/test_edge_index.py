"""Comprehensive test suite for spine.data.batch.edge_index module."""

import numpy as np
import pytest

from spine.data.batch.edge_index import EdgeIndexBatch
from spine.utils.conditional import TORCH_AVAILABLE, torch


class TestEdgeIndexBatchInitialization:
    """Test EdgeIndexBatch initialization patterns."""

    def test_initialization_directed_graph(self):
        """Test initialization with directed graph."""
        edges = np.array(
            [
                [0, 1, 2],  # Source nodes
                [1, 2, 0],  # Target nodes
            ]
        )
        counts = [3]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        assert batch.batch_size == 1
        assert batch.directed is True
        np.testing.assert_array_equal(batch.counts, [3])
        np.testing.assert_array_equal(batch.offsets, [0])

    def test_initialization_undirected_graph(self):
        """Test initialization with undirected graph."""
        # Undirected: each edge appears twice
        edges = np.array(
            [
                [0, 1, 1, 0],  # Edge 0-1 as both 0->1 and 1->0
                [1, 0, 0, 1],
            ]
        )
        counts = [4]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=False)

        assert batch.batch_size == 1
        assert batch.directed is False
        np.testing.assert_array_equal(batch.counts, [4])

    def test_initialization_multiple_graphs(self):
        """Test initialization with multiple graphs."""
        edges = np.array(
            [
                [0, 1, 0, 1],  # Two graphs
                [1, 0, 1, 0],
            ]
        )
        counts = [2, 2]
        offsets = np.array([0, 2])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        assert batch.batch_size == 2
        np.testing.assert_array_equal(batch.counts, [2, 2])

    def test_initialization_counts_sum_validation(self):
        """Test that counts must sum to number of edges."""
        edges = np.array(
            [
                [0, 1, 2],
                [1, 2, 0],
            ]
        )
        counts = [2]  # Only 2, but data has 3 edges

        with pytest.raises(AssertionError, match="do not add up"):
            EdgeIndexBatch(edges, counts=counts, offsets=np.array([0]), directed=True)

    def test_initialization_counts_offsets_length_match(self):
        """Test that counts and offsets must have same length."""
        edges = np.array(
            [
                [0, 1],
                [1, 0],
            ]
        )
        counts = [2]
        offsets = np.array([0, 10])  # Two offsets but only one count

        with pytest.raises(
            AssertionError, match="es not match"
        ):  # Typo in source: "es not match"
            EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

    def test_initialization_undirected_even_edges_validation(self):
        """Test that undirected graphs must have even number of edges."""
        edges = np.array(
            [
                [0, 1, 2],  # Odd number of edges
                [1, 2, 0],
            ]
        )
        counts = [3]
        offsets = np.array([0])

        with pytest.raises(AssertionError, match="even number"):
            EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=False)

    def test_initialization_empty_graphs(self):
        """Test initialization with empty graphs."""
        edges = np.array([[], []]).reshape(2, 0)
        counts = [0, 0]
        offsets = np.array([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        assert batch.batch_size == 2
        np.testing.assert_array_equal(batch.counts, [0, 0])


class TestEdgeIndexBatchIndexing:
    """Test EdgeIndexBatch __getitem__ method."""

    def test_getitem_basic(self):
        """Test basic indexing into batch."""
        edges = np.array(
            [
                [0, 1, 10, 11],  # Two graphs
                [1, 0, 11, 10],
            ]
        )
        counts = [2, 2]
        offsets = np.array([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        batch_0 = batch[0]
        batch_1 = batch[1]

        # Returns (N, 2) transposed array with offsets subtracted
        assert batch_0.shape == (2, 2)
        assert batch_1.shape == (2, 2)
        np.testing.assert_array_equal(batch_0, [[0, 1], [1, 0]])
        np.testing.assert_array_equal(batch_1, [[0, 1], [1, 0]])  # offset=10

    def test_getitem_single_edge(self):
        """Test indexing with single edge graphs."""
        edges = np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
            ]
        )
        counts = [1, 1, 1]
        offsets = np.array([0, 1, 2])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        batch_0 = batch[0]
        batch_1 = batch[1]
        batch_2 = batch[2]

        assert batch_0.shape == (1, 2)
        assert batch_1.shape == (1, 2)
        assert batch_2.shape == (1, 2)

    def test_getitem_out_of_bounds(self):
        """Test indexing beyond batch_size raises IndexError."""
        edges = np.array(
            [
                [0, 1],
                [1, 0],
            ]
        )
        counts = [2]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        with pytest.raises(IndexError, match="out of bound"):
            _ = batch[1]

    def test_getitem_empty_graph(self):
        """Test indexing empty graph."""
        edges = np.array(
            [
                [0, 1],
                [1, 0],
            ]
        )
        counts = [0, 2, 0]
        offsets = np.array([0, 0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        batch_0 = batch[0]
        batch_1 = batch[1]
        batch_2 = batch[2]

        assert batch_0.shape == (0, 2)
        assert batch_1.shape == (2, 2)
        assert batch_2.shape == (0, 2)


class TestEdgeIndexBatchProperties:
    """Test EdgeIndexBatch properties."""

    def test_index_property(self):
        """Test index property returns (2, E) data."""
        edges = np.array(
            [
                [0, 1, 2],
                [1, 2, 0],
            ]
        )
        counts = [3]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        assert batch.index is batch.data
        assert batch.index.shape == (2, 3)
        np.testing.assert_array_equal(batch.index, edges)

    def test_index_t_property(self):
        """Test index_t property returns transposed (E, 2) data."""
        edges = np.array(
            [
                [0, 1, 2],
                [1, 2, 0],
            ]
        )
        counts = [3]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        index_t = batch.index_t
        assert index_t.shape == (3, 2)
        np.testing.assert_array_equal(index_t, [[0, 1], [1, 2], [2, 0]])

    def test_batch_ids_property(self):
        """Test batch_ids returns batch ID per edge."""
        edges = np.array(
            [
                [0, 1, 2, 10, 11],
                [1, 2, 0, 11, 10],
            ]
        )
        counts = [3, 2]
        offsets = np.array([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        batch_ids = batch.batch_ids
        np.testing.assert_array_equal(batch_ids, [0, 0, 0, 1, 1])

    def test_directed_index_directed_graph(self):
        """Test directed_index for directed graph returns all edges."""
        edges = np.array(
            [
                [0, 1, 2],
                [1, 2, 0],
            ]
        )
        counts = [3]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        directed_idx = batch.directed_index
        np.testing.assert_array_equal(directed_idx, edges)

    def test_directed_index_undirected_graph(self):
        """Test directed_index for undirected graph returns every other edge."""
        edges = np.array(
            [
                [0, 1, 1, 0, 2, 3, 3, 2],  # Reciprocal pairs
                [1, 0, 0, 1, 3, 2, 2, 3],
            ]
        )
        counts = [8]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=False)

        directed_idx = batch.directed_index
        # Should skip every second edge
        expected = np.array(
            [
                [0, 1, 2, 3],
                [1, 0, 3, 2],
            ]
        )
        np.testing.assert_array_equal(directed_idx, expected)

    def test_directed_index_t_property(self):
        """Test directed_index_t returns transposed directed index."""
        edges = np.array(
            [
                [0, 1, 1, 0],  # Undirected
                [1, 0, 0, 1],
            ]
        )
        counts = [4]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=False)

        directed_idx_t = batch.directed_index_t
        # Should be (2, 2) after skipping every other edge
        assert directed_idx_t.shape == (2, 2)
        np.testing.assert_array_equal(directed_idx_t, [[0, 1], [1, 0]])

    def test_directed_counts_directed_graph(self):
        """Test directed_counts for directed graph equals counts."""
        edges = np.array(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 0],
            ]
        )
        counts = [2, 2]
        offsets = np.array([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        directed_counts = batch.directed_counts
        np.testing.assert_array_equal(directed_counts, counts)

    def test_directed_counts_undirected_graph(self):
        """Test directed_counts for undirected graph is half of counts."""
        edges = np.array(
            [
                [0, 1, 1, 0, 2, 3, 3, 2],
                [1, 0, 0, 1, 3, 2, 2, 3],
            ]
        )
        counts = [4, 4]
        offsets = np.array([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=False)

        directed_counts = batch.directed_counts
        np.testing.assert_array_equal(directed_counts, [2, 2])

    def test_directed_batch_ids_directed_graph(self):
        """Test directed_batch_ids for directed graph."""
        edges = np.array(
            [
                [0, 1, 2, 10, 11],
                [1, 2, 0, 11, 10],
            ]
        )
        counts = [3, 2]
        offsets = np.array([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        directed_batch_ids = batch.directed_batch_ids
        np.testing.assert_array_equal(directed_batch_ids, [0, 0, 0, 1, 1])

    def test_directed_batch_ids_undirected_graph(self):
        """Test directed_batch_ids for undirected graph."""
        edges = np.array(
            [
                [0, 1, 1, 0, 2, 3, 3, 2],
                [1, 0, 0, 1, 3, 2, 2, 3],
            ]
        )
        counts = [4, 4]
        offsets = np.array([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=False)

        directed_batch_ids = batch.directed_batch_ids
        # 4 edges -> 2 directed, 4 edges -> 2 directed
        np.testing.assert_array_equal(directed_batch_ids, [0, 0, 1, 1])


class TestEdgeIndexBatchSplit:
    """Test EdgeIndexBatch split method."""

    def test_split_basic(self):
        """Test split breaks batch into list of edge indexes."""
        edges = np.array(
            [
                [0, 1, 10, 11],
                [1, 0, 11, 10],
            ]
        )
        counts = [2, 2]
        offsets = np.array([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        result = batch.split()

        assert len(result) == 2
        assert result[0].shape == (2, 2)
        assert result[1].shape == (2, 2)

        # Check offset subtraction
        np.testing.assert_array_equal(result[0], [[0, 1], [1, 0]])
        np.testing.assert_array_equal(result[1], [[0, 1], [1, 0]])

    def test_split_multiple_entries(self):
        """Test split with multiple entries."""
        edges = np.array(
            [
                [0, 1, 10, 20, 21],
                [1, 0, 11, 21, 20],
            ]
        )
        counts = [2, 1, 2]
        offsets = np.array([0, 10, 20])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        result = batch.split()

        assert len(result) == 3
        assert result[0].shape == (2, 2)
        assert result[1].shape == (1, 2)
        assert result[2].shape == (2, 2)

    def test_split_single_entry(self):
        """Test split with single batch entry."""
        edges = np.array(
            [
                [0, 1, 2],
                [1, 2, 0],
            ]
        )
        counts = [3]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        result = batch.split()

        assert len(result) == 1
        assert result[0].shape == (3, 2)


class TestEdgeIndexBatchTypeConversions:
    """Test EdgeIndexBatch to_numpy and to_tensor methods."""

    def test_to_numpy_already_numpy(self):
        """Test to_numpy on already numpy data is idempotent."""
        edges = np.array(
            [
                [0, 1],
                [1, 0],
            ]
        )
        counts = [2]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)
        result = batch.to_numpy()

        assert result is batch

    def test_to_numpy_preserves_attributes(self):
        """Test to_numpy preserves directed attribute."""
        edges = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ]
        )
        counts = [4]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=False)
        result = batch.to_numpy()

        assert result.directed is False


class TestEdgeIndexBatchEdgeCases:
    """Test EdgeIndexBatch edge cases."""

    def test_empty_graph_in_middle(self):
        """Test batch with empty graph in the middle."""
        edges = np.array(
            [
                [0, 1, 10, 11],
                [1, 0, 11, 10],
            ]
        )
        counts = [2, 0, 2]
        offsets = np.array([0, 5, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        assert batch.batch_size == 3
        assert len(batch[0]) == 2
        assert len(batch[1]) == 0
        assert len(batch[2]) == 2

    def test_single_edge_graphs(self):
        """Test with single edge per graph."""
        edges = np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
            ]
        )
        counts = [1, 1, 1]
        offsets = np.array([0, 1, 2])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        assert batch.batch_size == 3
        for i in range(3):
            assert len(batch[i]) == 1

    def test_large_graph(self):
        """Test with large number of edges."""
        n_edges = 1000
        edges = np.array([np.arange(n_edges), np.arange(n_edges)[::-1]])
        counts = [n_edges]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        assert batch.batch_size == 1
        assert batch.data.shape == (2, n_edges)

    def test_large_offsets(self):
        """Test with large offset values."""
        edges = np.array(
            [
                [100000, 100001, 200000, 200001],
                [100001, 100000, 200001, 200000],
            ]
        )
        counts = [2, 2]
        offsets = np.array([100000, 200000])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        batch_0 = batch[0]
        batch_1 = batch[1]

        # Offsets should be subtracted correctly
        np.testing.assert_array_equal(batch_0, [[0, 1], [1, 0]])
        np.testing.assert_array_equal(batch_1, [[0, 1], [1, 0]])

    def test_undirected_reciprocal_edges(self):
        """Test undirected graph with proper reciprocal edges."""
        # Each edge (i,j) should have reciprocal (j,i) immediately after
        edges = np.array(
            [
                [0, 1, 1, 2, 2, 0],  # Edges: 0-1, 1-2, 2-0 (undirected)
                [1, 0, 2, 1, 0, 2],
            ]
        )
        counts = [6]
        offsets = np.array([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=False)

        # directed_index should have 3 edges (half of 6)
        directed_idx = batch.directed_index
        assert directed_idx.shape == (2, 3)

        # directed_counts should be 3
        np.testing.assert_array_equal(batch.directed_counts, [3])

    def test_zero_offsets_all_batches(self):
        """Test with zero offsets for all batches."""
        edges = np.array(
            [
                [0, 1, 2, 3, 4, 5],
                [1, 0, 3, 2, 5, 4],
            ]
        )
        counts = [2, 2, 2]
        offsets = np.array([0, 0, 0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        # All batches should have same values since offset is 0
        batch_0 = batch[0]
        np.testing.assert_array_equal(batch_0, [[0, 1], [1, 0]])

    def test_directed_vs_undirected_properties(self):
        """Test that directed and undirected differ in properties."""
        edges = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ]
        )
        counts = [4]
        offsets = np.array([0])

        directed_batch = EdgeIndexBatch(
            edges, counts=counts, offsets=offsets, directed=True
        )
        undirected_batch = EdgeIndexBatch(
            edges, counts=counts, offsets=offsets, directed=False
        )

        # Directed has all 4 edges
        assert directed_batch.directed_index.shape == (2, 4)
        np.testing.assert_array_equal(directed_batch.directed_counts, [4])

        # Undirected has 2 directed edges
        assert undirected_batch.directed_index.shape == (2, 2)
        np.testing.assert_array_equal(undirected_batch.directed_counts, [2])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestEdgeIndexBatchWithTorch:
    """Test EdgeIndexBatch with PyTorch tensors."""

    def test_torch_directed_graph_initialization(self):
        """Test EdgeIndexBatch with torch tensor for directed graph."""
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])  # Shape (2, 3)
        counts = [3]
        offsets = torch.tensor([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        assert batch.is_numpy is False
        assert isinstance(batch.data, torch.Tensor)
        assert batch.directed is True
        assert batch.batch_size == 1

    def test_torch_undirected_graph(self):
        """Test EdgeIndexBatch with torch tensor for undirected graph."""
        # Reciprocal edges for undirected graph
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # Shape (2, 4)
        counts = [4]
        offsets = torch.tensor([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=False)

        assert batch.directed is False
        # Verify even number of edges
        assert torch.equal(batch.counts, torch.tensor([4]))

    def test_torch_indexing(self):
        """Test indexing with torch tensors."""
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # Shape (2, 4)
        counts = [2, 2]
        offsets = torch.tensor([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        # Test single index - returns (N, 2) transposed format
        batch_0 = batch[0]
        assert batch_0.shape == (2, 2)
        # First edge [0, 1] becomes [[0, 1], [1, 2]] after transpose
        assert torch.equal(batch_0, torch.tensor([[0, 1], [1, 2]]))

        batch_1 = batch[1]
        assert batch_1.shape == (2, 2)
        # Offset subtraction: [[2, 3], [3, 4]] - 10 = [[-8, -7], [-7, -6]]
        assert torch.equal(batch_1, torch.tensor([[-8, -7], [-7, -6]]))

    def test_torch_split(self):
        """Test split method with torch tensors."""
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])  # Shape (2, 3)
        counts = [2, 1]
        offsets = torch.tensor([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)
        split_data = batch.split()

        assert len(split_data) == 2
        # First batch: 2 edges, returned as (N, 2) format
        assert split_data[0].shape == (2, 2)
        # Second batch: 1 edge, returned as (N, 2) format, with offset subtracted
        assert split_data[1].shape == (1, 2)
        assert torch.equal(split_data[1], torch.tensor([[-8, -7]]))

    def test_torch_index_property(self):
        """Test index property with torch tensors."""
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])  # Shape (2, 3)
        counts = [3]
        offsets = torch.tensor([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        index = batch.index
        # Should be transposed to (2, N)
        assert index.shape == (2, 3)
        assert torch.equal(index, torch.tensor([[0, 1, 2], [1, 2, 3]]))

    def test_torch_index_t_property(self):
        """Test index_t property with torch tensors."""
        edges = torch.tensor([[0, 1], [1, 2]])  # Shape (2, 2)
        counts = [2]
        offsets = torch.tensor([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        index_t = batch.index_t
        # Transposed (2, 2) format
        assert index_t.shape == (2, 2)
        assert torch.equal(index_t, torch.tensor([[0, 1], [1, 2]]))

    def test_torch_batch_ids(self):
        """Test batch_ids with torch tensors."""
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # Shape (2, 4)
        counts = [2, 2]
        offsets = torch.tensor([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)
        batch_ids = batch.batch_ids

        assert torch.equal(batch_ids, torch.tensor([0, 0, 1, 1]))

    def test_torch_directed_index_undirected(self):
        """Test directed_index for undirected graph with torch."""
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # Shape (2, 4)
        counts = [4]
        offsets = torch.tensor([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=False)

        # directed_index should filter every other edge
        directed_index = batch.directed_index
        assert directed_index.shape == (2, 2)
        assert torch.equal(directed_index, torch.tensor([[0, 1], [1, 2]]))

    def test_torch_directed_counts(self):
        """Test directed_counts with torch tensors."""
        edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])  # Shape (2, 6)
        counts = [4, 2]
        offsets = torch.tensor([0, 10])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=False)

        directed_counts = batch.directed_counts
        # Each undirected edge is stored as pair, so half the counts
        assert torch.equal(directed_counts, torch.tensor([2, 1]))

    def test_to_tensor_idempotent(self):
        """Test to_tensor on already torch data is idempotent."""
        edges = torch.tensor([[0, 1], [1, 2]])  # Shape (2, 2)
        counts = [2]
        offsets = torch.tensor([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)
        result = batch.to_tensor()

        assert result is batch
        assert result.is_numpy is False

    def test_to_numpy_from_torch(self):
        """Test to_numpy converts torch to numpy."""
        edges = torch.tensor([[0, 1], [1, 2]])  # Shape (2, 2)
        counts = [2]
        offsets = torch.tensor([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)
        result = batch.to_numpy()

        assert result.is_numpy is True
        assert isinstance(result.data, np.ndarray)
        np.testing.assert_array_equal(result.data, [[0, 1], [1, 2]])

    def test_torch_empty_graph(self):
        """Test with empty torch tensor graph."""
        edges = torch.empty((2, 0), dtype=torch.long)  # Shape (2, 0)
        counts = [0]
        offsets = torch.tensor([0])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        assert batch.batch_size == 1
        assert batch.index.shape == (2, 0)

    def test_torch_large_offsets(self):
        """Test with large offsets using torch."""
        edges = torch.tensor([[0, 1], [1, 2]])  # Shape (2, 2)
        counts = [1, 1]
        offsets = torch.tensor([0, 1000000])

        batch = EdgeIndexBatch(edges, counts=counts, offsets=offsets, directed=True)

        # Indexing should handle large offsets
        batch_1 = batch[1]
        # Second edge [1, 2] with offset 1000000 subtracted, returned as (N, 2) format
        assert torch.equal(batch_1, torch.tensor([[-999999, -999998]]))
