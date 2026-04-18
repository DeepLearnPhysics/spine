"""Test that the cluster data parsers work as intended."""

import numpy as np
import pytest

from spine.data.larcv.meta import ImageMeta2D, ImageMeta3D
from spine.io.core.parse.cluster import *
from spine.io.core.parse.data import ParserTensor
from spine.utils.conditional import LARCV_AVAILABLE, larcv

pytestmark = pytest.mark.skipif(
    not LARCV_AVAILABLE, reason="LArCV is required to generate parser fixtures."
)


@pytest.mark.parametrize("projection_id", [0, 1, 2])
@pytest.mark.parametrize("cluster2d_event", [0, 1, 20], indirect=True)
def test_parse_cluster2d(cluster2d_event, projection_id):
    """Tests the parsing of LArCV 2D sparse data organized in a group."""
    # Initialize the parser
    parser = Cluster2DParser(
        dtype="float32", cluster_event=cluster2d_event, projection_id=projection_id
    )

    # Parse the data
    result = parser.process(cluster_event=cluster2d_event)

    # There should be 3 components of the output
    # - The first has both coordinates for each point
    # - The second has the feature tensor (value + cluster ID)
    # - The third has the metadata
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 2
    assert result.features.shape[1] == 2
    assert isinstance(result.meta, ImageMeta2D)


@pytest.mark.parametrize(
    "cluster3d_event, particle_event", [(0, 0), (1, 1), (20, 20)], indirect=True
)
@pytest.mark.parametrize("neutrino_event", [0, 1], indirect=True)
@pytest.mark.parametrize(
    "add_particle_info, clean_data",
    [(False, False), (False, False), (True, False), (True, True)],
)
@pytest.mark.parametrize("break_clusters", [False, True])
def test_parse_cluster3d(
    cluster3d_event,
    particle_event,
    neutrino_event,
    add_particle_info,
    break_clusters,
    clean_data,
):
    """Tests the parsing of LArCV 3D sparse data organized in a group."""
    # Generate the sparse value/sparse semantic labels based on the cluster3d
    sparse3d_event, sparse3d_seg_event = None, None
    if clean_data:
        sparse3d_event = cluster3d_to_sparse3d(cluster3d_event)
        sparse3d_seg_event = cluster3d_to_sparse3d(cluster3d_event, True)

    # Initialize the parser
    parser = Cluster3DParser(
        dtype="float32",
        cluster_event=cluster3d_event,
        particle_event=particle_event,
        neutrino_event=neutrino_event,
        sparse_value_event=sparse3d_event,
        sparse_semantics_event=sparse3d_seg_event,
        clean_data=clean_data,
        add_particle_info=add_particle_info,
        break_clusters=break_clusters,
    )

    # Parse the data
    result = parser.process(
        cluster_event=cluster3d_event,
        particle_event=particle_event,
        neutrino_event=neutrino_event,
        sparse_value_event=sparse3d_event,
        sparse_semantics_event=sparse3d_seg_event,
    )

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the feature tensor (value + cluster ID)
    # - The third has the metadata
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 3
    assert result.features.shape[1] == (15 if add_particle_info else 2)
    assert isinstance(result.meta, ImageMeta3D)


@pytest.mark.parametrize("cluster3d_event, particle_event", [(20, 20)], indirect=True)
@pytest.mark.parametrize("neutrino_event", [1], indirect=True)
@pytest.mark.parametrize("add_particle_info, clean_data", [(True, True)])
@pytest.mark.parametrize("break_clusters", [True])
def test_parse_cluster3d_rescale(
    cluster3d_event,
    particle_event,
    neutrino_event,
    add_particle_info,
    break_clusters,
    clean_data,
):
    """Tests the parsing of LArCV 3D sparse data organized in a group."""
    # Generate the sparse value/sparse semantic labels based on the cluster3d
    sparse3d_event, sparse3d_seg_event = None, None
    if clean_data:
        sparse3d_seg_event = cluster3d_to_sparse3d(cluster3d_event, True, False)
        sparse3d_event_list = [cluster3d_to_sparse3d(cluster3d_event)] * 6
        sparse3d_event_list += [sparse3d_seg_event]

    # Initialize the parser
    parser = Cluster3DChargeRescaledParser(
        dtype="float32",
        cluster_event=cluster3d_event,
        particle_event=particle_event,
        neutrino_event=neutrino_event,
        sparse_value_event_list=sparse3d_event_list,
        sparse_semantics_event=sparse3d_seg_event,
        clean_data=clean_data,
        add_particle_info=add_particle_info,
        break_clusters=break_clusters,
    )

    # Parse the data
    result = parser.process_rescale(
        cluster_event=cluster3d_event,
        particle_event=particle_event,
        neutrino_event=neutrino_event,
        sparse_value_event_list=sparse3d_event_list,
        sparse_semantics_event=sparse3d_seg_event,
    )

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the feature tensor (value + cluster ID)
    # - The third has the metadata
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 3
    assert result.features.shape[1] == (15 if add_particle_info else 2)
    assert isinstance(result.meta, ImageMeta3D)


@pytest.mark.parametrize("cluster3d_event, particle_event", [(20, 20)], indirect=True)
@pytest.mark.parametrize("neutrino_event", [1], indirect=True)
@pytest.mark.parametrize("add_particle_info, clean_data", [(True, True)])
@pytest.mark.parametrize("break_clusters", [True])
def test_parse_cluster3d_aggregate(
    cluster3d_event,
    particle_event,
    neutrino_event,
    add_particle_info,
    break_clusters,
    clean_data,
):
    """Tests the parsing of LArCV 3D sparse data organized in a group."""
    # Generate the sparse value/sparse semantic labels based on the cluster3d
    sparse3d_event_list, sparse3d_seg_event = None, None
    if clean_data:
        sparse3d_event_list = [cluster3d_to_sparse3d(cluster3d_event)] * 2
        sparse3d_seg_event = cluster3d_to_sparse3d(cluster3d_event, True, False)

    # Initialize the parser
    parser = Cluster3DAggregateParser(
        dtype="float32",
        value_aggr="max",
        cluster_event=cluster3d_event,
        particle_event=particle_event,
        neutrino_event=neutrino_event,
        sparse_value_event_list=sparse3d_event_list,
        sparse_semantics_event=sparse3d_seg_event,
        clean_data=clean_data,
        add_particle_info=add_particle_info,
        break_clusters=break_clusters,
    )

    # Parse the data
    result = parser.process_aggr(
        cluster_event=cluster3d_event,
        particle_event=particle_event,
        neutrino_event=neutrino_event,
        sparse_value_event_list=sparse3d_event_list,
        sparse_semantics_event=sparse3d_seg_event,
    )

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the feature tensor (value + cluster ID)
    # - The third has the metadata
    assert isinstance(result, ParserTensor)
    assert result.coords.shape[1] == 3
    assert result.features.shape[1] == (15 if add_particle_info else 2)
    assert isinstance(result.meta, ImageMeta3D)


def cluster3d_to_sparse3d(cluster3d_event, segmentation=False, ghost=True):
    """Merge all clusters in a cluster3d object into a single sparse object.

    Parameters
    ----------
    larcv.EventClusterVoxel3D
        Cluster of 3D sparse tensor
    segmentation : bool, default True
        If `True`, create dummy segmentation labels for the output tensor
    ghost : bool, default True
        If `True`, include ghost labels in the dummy segmentation labels

    Returns
    -------
    larcv.EventSparseTensor3D
        Event containing one 3D larcv sparse tensor
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Loop over the clusters, append the data needed to build a sparse tensor
    meta = cluster3d_event.meta()
    voxels, values = [], []
    for cluster in cluster3d_event.as_vector():
        num_points = cluster.size()
        if num_points:
            # Load data from this cluster
            x = np.empty(num_points, dtype=np.int32)
            y = np.empty(num_points, dtype=np.int32)
            z = np.empty(num_points, dtype=np.int32)
            value = np.empty(num_points, dtype=np.float32)
            larcv.as_flat_arrays(cluster, meta, x, y, z, value)

            voxels.append(np.vstack((x, y, z)).T)
            if not segmentation:
                values.append(value)
            else:
                values.append(
                    np.random.randint(0, 5 + int(ghost), size=num_points).astype(
                        np.float32
                    )
                )

    # Generate tensor
    if len(voxels):
        voxels = np.vstack(voxels)
        values = np.concatenate(values)
    else:
        voxels = np.empty((0, 3), dtype=np.int32)
        values = np.empty(0, dtype=np.float32)

    # Build a SparseTensor3D, set it
    voxel_set = larcv.as_tensor3d(voxels, values, meta, -0.01)
    event = larcv.EventSparseTensor3D()
    event.set(voxel_set, meta)

    return event
