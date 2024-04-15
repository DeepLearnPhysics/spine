"""Test that the cluster data parsers work as intended."""

import pytest

from mlreco import Meta
from mlreco.iotools.parsers.cluster import *

from test.test_iotools.test_parsers.fixtures import (
        fixture_cluster2d_event, fixture_cluster3d_event,
        fixture_particle_event, fixture_neutrino_event,
        cluster3d_to_sparse3d)


@pytest.mark.parametrize('projection_id', [0, 1, 2])
@pytest.mark.parametrize('cluster2d_event', [0, 1, 20], indirect=True)
def test_parse_cluster2d(cluster2d_event, projection_id):
    """Tests the parsing of LArCV 2D sparse data organized in a group."""
    # Initialize the parser
    parser = Cluster2DParser(
            cluster_event=cluster2d_event, projection_id=projection_id)

    # Parse the data
    result = parser.process(cluster_event=cluster2d_event)

    # There should be 3 components of the output
    # - The first has both coordinates for each point
    # - The second has the feature tensor (value + cluster ID)
    # - The third has the metadata
    assert len(result) == 3
    assert result[0].shape[1] == 2
    assert result[1].shape[1] == 2
    assert isinstance(result[2], Meta)


@pytest.mark.parametrize(
        'cluster3d_event, particle_event',
        [(0, 0), (1, 1), (20, 20)], indirect=True)
@pytest.mark.parametrize('neutrino_event', [0, 1], indirect=True)
@pytest.mark.parametrize(
        'add_particle_info, clean_data',
        [(False, False), (False, False), (True, False), (True, True)])
@pytest.mark.parametrize('break_clusters', [False, True])
def test_parse_cluster3d(cluster3d_event, particle_event, neutrino_event,
                         add_particle_info, break_clusters, clean_data):
    """Tests the parsing of LArCV 3D sparse data organized in a group."""
    # Generate the sparse value/sparse semantic labels based on the cluster3d
    sparse3d_event, sparse3d_seg_event = None, None
    if clean_data:
        sparse3d_event = cluster3d_to_sparse3d(cluster3d_event)
        sparse3d_seg_event = cluster3d_to_sparse3d(cluster3d_event, True)

    # Initialize the parser
    parser = Cluster3DParser(
            cluster_event=cluster3d_event, particle_event=particle_event,
            neutrino_event=neutrino_event, sparse_value_event=sparse3d_event,
            sparse_semantics_event=sparse3d_seg_event, clean_data=clean_data,
            add_particle_info=add_particle_info, break_clusters=break_clusters)

    # Parse the data
    result = parser.process(
            cluster_event=cluster3d_event, particle_event=particle_event,
            neutrino_event=neutrino_event, sparse_value_event=sparse3d_event,
            sparse_semantics_event=sparse3d_seg_event)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the feature tensor (value + cluster ID)
    # - The third has the metadata
    assert len(result) == 3
    assert result[0].shape[1] == 3
    assert result[1].shape[1] == (14 if add_particle_info else 2)
    assert isinstance(result[2], Meta)


@pytest.mark.parametrize(
        'cluster3d_event, particle_event', [(20, 20)], indirect=True)
@pytest.mark.parametrize('neutrino_event', [1], indirect=True)
@pytest.mark.parametrize('add_particle_info, clean_data', [(True, True)])
@pytest.mark.parametrize('break_clusters', [True])
def test_parse_cluster3d_rescale(cluster3d_event, particle_event,
                                 neutrino_event, add_particle_info,
                                 break_clusters, clean_data):
    """Tests the parsing of LArCV 3D sparse data organized in a group."""
    # Generate the sparse value/sparse semantic labels based on the cluster3d
    sparse3d_event, sparse3d_seg_event = None, None
    if clean_data:
        sparse3d_seg_event = cluster3d_to_sparse3d(cluster3d_event, True, False)
        sparse3d_event_list = [cluster3d_to_sparse3d(cluster3d_event)] * 6
        sparse3d_event_list += [sparse3d_seg_event]

    # Initialize the parser
    parser = Cluster3DChargeRescaledParser(
            cluster_event=cluster3d_event, particle_event=particle_event,
            neutrino_event=neutrino_event,
            sparse_value_event_list=sparse3d_event_list,
            sparse_semantics_event=sparse3d_seg_event, clean_data=clean_data,
            add_particle_info=add_particle_info, break_clusters=break_clusters)

    # Parse the data
    result = parser.process(
            cluster_event=cluster3d_event, particle_event=particle_event,
            neutrino_event=neutrino_event,
            sparse_value_event_list=sparse3d_event_list,
            sparse_semantics_event=sparse3d_seg_event)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the feature tensor (value + cluster ID)
    # - The third has the metadata
    assert len(result) == 3
    assert result[0].shape[1] == 3
    assert result[1].shape[1] == (14 if add_particle_info else 2)
    assert isinstance(result[2], Meta)


@pytest.mark.parametrize(
        'cluster3d_event, particle_event', [(20, 20)], indirect=True)
@pytest.mark.parametrize('neutrino_event', [1], indirect=True)
@pytest.mark.parametrize('add_particle_info, clean_data', [(True, True)])
@pytest.mark.parametrize('break_clusters', [True])
def test_parse_cluster3d_multi(cluster3d_event, particle_event,
                               neutrino_event, add_particle_info,
                               break_clusters, clean_data):
    """Tests the parsing of LArCV 3D sparse data organized in a group."""
    # Generate the sparse value/sparse semantic labels based on the cluster3d
    sparse3d_event, sparse3d_seg_event = None, None
    if clean_data:
        sparse3d_event_list = [cluster3d_to_sparse3d(cluster3d_event)] * 2
        sparse3d_seg_event = cluster3d_to_sparse3d(cluster3d_event, True, False)

    # Initialize the parser
    parser = Cluster3DMultiModuleParser(
            cluster_event=cluster3d_event, particle_event=particle_event,
            neutrino_event=neutrino_event,
            sparse_value_event_list=sparse3d_event_list,
            sparse_semantics_event=sparse3d_seg_event, clean_data=clean_data,
            add_particle_info=add_particle_info, break_clusters=break_clusters)

    # Parse the data
    result = parser.process(
            cluster_event=cluster3d_event, particle_event=particle_event,
            neutrino_event=neutrino_event,
            sparse_value_event_list=sparse3d_event_list,
            sparse_semantics_event=sparse3d_seg_event)

    # There should be 3 components of the output
    # - The first has all 3 coordinates for each point
    # - The second has the feature tensor (value + cluster ID)
    # - The third has the metadata
    assert len(result) == 3
    assert result[0].shape[1] == 3
    assert result[1].shape[1] == (14 if add_particle_info else 2)
    assert isinstance(result[2], Meta)
