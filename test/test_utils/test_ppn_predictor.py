"""Direct tests for the shared PPN prediction utilities."""

import numpy as np
import pytest
import torch

from spine.constants import SHOWR_SHP, TRACK_SHP
from spine.data import IndexBatch, TensorBatch, TensorData
from spine.utils.ppn import (
    ParticlePointPredictor,
    PPNPredictor,
    ppn_prediction_schema,
    ppn_raw_schema,
)


def raw_prediction(array_type=np.asarray):
    """Build four raw PPN proposals, two of which pass the score mask."""
    features = np.zeros((4, 10), dtype=np.float32)
    features[:, :3] = [[0, 0, 0], [0.1, 0, 0], [0, 0, 0], [0, 0, 0]]
    features[:, 3:8] = [
        [8, 0, 0, 0, 0],
        [7, 0, 0, 0, 0],
        [0, 8, 0, 0, 0],
        [0, 8, 0, 0, 0],
    ]
    features[:, 8:10] = [[0, 8], [0, 7], [8, 0], [8, 0]]
    if array_type is torch.as_tensor:
        features = torch.as_tensor(features)
    return TensorData(features=features, schema=ppn_raw_schema())


@pytest.mark.parametrize("pool", ["max", "mean"])
def test_ppn_process_single_numpy_pooling_empty_and_selection(pool):
    """NumPy post-processing should filter, pool, classify endpoints, and empty."""
    predictor = PPNPredictor(enforce_type=False, pool_score_fn=pool, pool_dist=1.0)
    raw = raw_prediction()
    coords = np.array([[0, 0, 0], [0.2, 0, 0], [5, 0, 0], [6, 0, 0]], dtype=np.float32)
    mask = np.ones(4, dtype=bool)
    endpoints = np.array([[0, 4], [0, 3], [2, 0], [2, 0]], dtype=np.float32)

    result = predictor.process_single(raw, coords, mask, ppn_ends=endpoints)
    assert result.coords.shape == (1, 3)
    assert result.features.shape == (1, 11)
    assert result.feature("occupancy")[0, 0] == 2
    assert result.feature("shape")[0, 0] == SHOWR_SHP

    selected = predictor.process_single(raw, coords, mask, selection=np.array([1]))
    assert selected.coords.shape == (1, 3)
    empty = predictor.process_single(raw, coords, np.zeros(4, dtype=bool))
    assert empty.coords.shape == (0, 3)
    assert empty.features.shape == (0, 9)


def test_ppn_process_single_type_enforcement_and_deghosting():
    """Semantic enforcement should require logits and honor the ghost mask."""
    raw = raw_prediction()
    coords = np.array([[0, 0, 0], [0.2, 0, 0], [5, 0, 0], [6, 0, 0]], dtype=np.float32)
    mask = np.ones(4, dtype=bool)
    predictor = PPNPredictor(classes=(SHOWR_SHP,), apply_deghosting=True)
    with pytest.raises(ValueError, match="segmentation"):
        predictor.process_single(raw, coords, mask)

    segmentation = np.zeros((4, 5), dtype=np.float32)
    segmentation[:, SHOWR_SHP] = 5.0
    ghost = np.array([[5, 0], [5, 0], [0, 5], [0, 5]], dtype=np.float32)
    result = predictor.process_single(
        raw, coords, mask, segmentation=segmentation, ghost=ghost
    )
    assert result.coords.shape == (1, 3)


def test_ppn_process_single_torch_and_schema_helpers():
    """Torch post-processing should preserve its device-backed representation."""
    predictor = PPNPredictor(
        classes=(SHOWR_SHP,), apply_deghosting=True, pool_score_fn="max"
    )
    raw = raw_prediction(torch.as_tensor)
    coords = torch.tensor([[0, 0, 0], [0.2, 0, 0], [5, 0, 0], [6, 0, 0]])
    segmentation = torch.zeros((4, 5))
    segmentation[:, SHOWR_SHP] = 5.0
    result = predictor.process_single(
        raw,
        coords,
        torch.ones(4, dtype=torch.bool),
        ppn_ends=torch.tensor([[0.0, 4.0]] * 4),
        segmentation=segmentation,
        ghost=torch.tensor([[5.0, 0.0]] * 4),
        selection=torch.tensor([0, 1]),
    )
    assert torch.is_tensor(result.features)
    assert result.coords.shape == (1, 3)
    assert "endpoint_scores" not in ppn_prediction_schema(False).feature_fields
    assert "endpoint_scores" in ppn_prediction_schema(True).feature_fields


def test_ppn_batch_and_unwrapped_dispatch():
    """The public predictor should preserve event containers and entry selection."""
    predictor = PPNPredictor(enforce_type=False)
    coords = np.array([[0, 0, 0], [0.2, 0, 0], [5, 0, 0], [6, 0, 0]], dtype=np.float32)
    coord_data = TensorData(coords=coords, features=np.empty((4, 0), dtype=np.float32))
    mask_data = TensorData(features=np.ones((4, 1), dtype=bool))
    endpoints = TensorData(features=np.array([[0.0, 4.0]] * 4, dtype=np.float32))
    segmentation = TensorData(features=np.zeros((4, 5), dtype=np.float32))
    ghost = TensorData(features=np.zeros((4, 2), dtype=np.float32))

    kwargs = {
        "ppn_points": [raw_prediction()],
        "ppn_coords": [[coord_data]],
        "ppn_masks": [[mask_data]],
        "ppn_classify_endpoints": [endpoints],
        "segmentation": [segmentation],
        "ghost": [ghost],
        "selection": [np.array([0, 1])],
    }
    output = predictor(**kwargs)
    assert isinstance(output, list)
    assert output[0].features.shape[1] == 11
    assert isinstance(predictor(entry=0, **kwargs), TensorData)
    with pytest.raises(TypeError, match="integer"):
        predictor(entry="0", **kwargs)

    batched = {
        "ppn_points": TensorBatch.from_data_list([raw_prediction()]),
        "ppn_coords": [TensorBatch.from_data_list([coord_data])],
        "ppn_masks": [TensorBatch.from_data_list([mask_data])],
        "ppn_classify_endpoints": TensorBatch.from_data_list([endpoints]),
        "segmentation": TensorBatch.from_data_list([segmentation]),
        "ghost": TensorBatch.from_data_list([ghost]),
        "selection": IndexBatch(np.array([0, 1]), spans=[4], counts=[2]),
    }
    output = predictor(**batched)
    assert isinstance(output, TensorBatch)
    assert output.counts.tolist() == [1]


def particle_inputs(array_type=np.asarray):
    """Return a track and shower with offsets and point logits."""
    points = np.array(
        [[0, 0, 0], [2, 0, 0], [4, 0, 0], [0, 5, 0], [1, 5, 0]],
        dtype=np.float32,
    )
    clusts = [np.array([0, 1, 2]), np.array([3, 4])]
    shapes = np.array([TRACK_SHP, SHOWR_SHP], dtype=np.int64)
    offsets = np.array([[0.2, 0, 0]] * 5, dtype=np.float32)
    logits = np.array([[0, 5], [5, 0], [0, 5], [0, 1], [0, 4]], dtype=np.float32)
    if array_type is torch.as_tensor:
        points = torch.as_tensor(points)
        shapes = torch.as_tensor(shapes)
        offsets = torch.as_tensor(offsets)
        logits = torch.as_tensor(logits)
    return points, clusts, shapes, offsets, logits


@pytest.mark.parametrize(
    "contained,anchor,enhance,approx",
    [(True, True, True, True), (False, False, False, False)],
)
def test_particle_point_predictor_numpy_branches(contained, anchor, enhance, approx):
    """NumPy particle points should cover track enhancement and shower choices."""
    predictor = ParticlePointPredictor(
        contained_first=contained,
        anchor_points=anchor,
        enhance_track_points=enhance,
        approx_farthest_points=approx,
    )
    output = predictor.get_end_points_numpy(*particle_inputs())
    assert output.shape == (2, 6)
    np.testing.assert_allclose(output[1, :3], output[1, 3:])
    assert predictor.get_end_points_numpy(
        np.empty((0, 3), dtype=np.float32),
        [],
        np.empty(0, dtype=np.int64),
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
    ).shape == (0, 6)


@pytest.mark.parametrize(
    "contained,anchor,enhance", [(True, True, True), (False, False, False)]
)
def test_particle_point_predictor_torch_branches(contained, anchor, enhance):
    """Torch particle points should mirror NumPy track and shower behavior."""
    predictor = ParticlePointPredictor(
        use_numpy=False,
        contained_first=contained,
        anchor_points=anchor,
        enhance_track_points=enhance,
    )
    output = predictor.get_end_points_torch(*particle_inputs(torch.as_tensor))
    assert output.shape == (2, 6)
    assert torch.allclose(output[1, :3], output[1, 3:])


def test_particle_point_public_dispatch_and_contained_fallback():
    """The public API should validate storage and cover no-contained-point fallback."""
    points, clusts, shapes, offsets, logits = particle_inputs()
    offsets[3:] = 2.0
    predictor = ParticlePointPredictor(contained_first=True, anchor_points=False)
    output = predictor.get_end_points_numpy(points, clusts, shapes, offsets, logits)
    assert output.shape == (2, 6)

    torch_inputs = particle_inputs(torch.as_tensor)
    torch_inputs[3][3:] = 2.0
    torch_predictor = ParticlePointPredictor(
        use_numpy=False, contained_first=True, anchor_points=False
    )
    assert torch_predictor.get_end_points_torch(*torch_inputs).shape == (2, 6)

    data = TensorBatch(points, counts=[5], coord_cols=np.arange(3))
    clusters = IndexBatch(clusts, spans=[5], counts=[2], single_counts=[3, 2])
    shape_batch = TensorBatch(shapes, counts=[2])
    ppn_batch = TensorBatch.from_data_list([raw_prediction()])
    # Pad the four-proposal fixture to the five input voxels.
    ppn_batch = TensorBatch(
        np.vstack([ppn_batch.tensor, ppn_batch.tensor[-1]]),
        counts=[5],
        schema=ppn_raw_schema(),
    )
    result = predictor(data, clusters, shape_batch, ppn_batch)
    assert result.shape == (2, 6)

    with pytest.raises(TypeError, match="list-backed"):
        predictor(
            data,
            IndexBatch(np.arange(5), spans=[5], counts=[5]),
            shape_batch,
            ppn_batch,
        )
    with pytest.raises(TypeError, match="torch-backed"):
        ParticlePointPredictor(use_numpy=False)(data, clusters, shape_batch, ppn_batch)

    torch_points, torch_clusts, torch_shapes, _, _ = particle_inputs(torch.as_tensor)
    torch_data = TensorBatch(torch_points, counts=[5], coord_cols=np.arange(3))
    torch_clusters = IndexBatch(
        torch_clusts, spans=[5], counts=[2], single_counts=[3, 2]
    )
    torch_shape_batch = TensorBatch(torch_shapes, counts=[2])
    torch_raw = raw_prediction(torch.as_tensor)
    torch_ppn_batch = TensorBatch(
        torch.vstack([torch_raw.features, torch_raw.features[-1]]),
        counts=[5],
        schema=ppn_raw_schema(),
    )
    result = ParticlePointPredictor(use_numpy=False)(
        torch_data, torch_clusters, torch_shape_batch, torch_ppn_batch
    )
    assert result.shape == (2, 6)
