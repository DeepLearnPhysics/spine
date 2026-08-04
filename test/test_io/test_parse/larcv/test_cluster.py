"""Test that the cluster data parsers work as intended."""

import numpy as np
import pytest

from spine.constants import LOWES_SHP, PART_COL, VALUE_COL
from spine.data import ClusterLabelData, TensorData
from spine.data.larcv.meta import ImageMeta2D, ImageMeta3D
from spine.io.parse.larcv.cluster import *
from spine.utils.conditional import LARCV_AVAILABLE, larcv

pytestmark = pytest.mark.skipif(
    not LARCV_AVAILABLE, reason="LArCV is required to generate parser fixtures."
)


@pytest.mark.parametrize("projection_id", [0, 1, 2])
@pytest.mark.parametrize("cluster2d_event", [0, 1, 20], indirect=True)
def test_parse_cluster2d(cluster2d_event, projection_id):
    """Tests the parsing of LArCV 2D sparse data organized in a group."""
    # Initialize the parser
    parser = LArCVCluster2DParser(
        dtype="float32", cluster_event=cluster2d_event, projection_id=projection_id
    )

    # Parse the data
    result = parser.process(cluster_event=cluster2d_event)

    # There should be 3 components of the output
    # - The first has both coordinates for each point
    # - The second has the feature tensor (value + cluster ID)
    # - The third has the metadata
    assert isinstance(result, TensorData)
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
@pytest.mark.filterwarnings(
    "ignore:Neutrino IDs are being produced on the basis of floating point agreement.*:UserWarning"
)
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
    parser = LArCVCluster3DParser(
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
    assert isinstance(result, ClusterLabelData)
    assert result.coords.shape[1] == 3
    assert result.features.shape[1] == (3 if add_particle_info else 2)
    assert (result.particles is not None) == add_particle_info
    assert isinstance(result.meta, ImageMeta3D)
    if add_particle_info:
        expected_shapes = np.asarray(
            [particle.shape() for particle in particle_event.as_vector()]
        )
        np.testing.assert_array_equal(result.particles["shape"], expected_shapes)


@pytest.mark.parametrize("cluster3d_event, particle_event", [(20, 20)], indirect=True)
def test_parse_cluster3d_nested_particle_configuration(cluster3d_event, particle_event):
    """The canonical nested configuration should build a particle table."""
    parser = LArCVCluster3DParser(
        dtype="float32",
        cluster_event="cluster",
        particle_info={
            "particle_event": "particles",
            "type_include_secondary": False,
            "type_include_mpr": False,
            "primary_include_mpr": False,
        },
    )

    result = parser({"cluster": cluster3d_event, "particles": particle_event})

    assert isinstance(result, ClusterLabelData)
    assert result.features.shape[1] == 3
    assert result.particles is not None
    assert parser.type_include_secondary is False
    assert parser.type_include_mpr is False
    assert parser.primary_include_mpr is False


@pytest.mark.parametrize("cluster3d_event", [20], indirect=True)
def test_parse_cluster3d_without_particle_table(cluster3d_event):
    """An explicit null particle configuration should retain compact labels."""
    parser = LArCVCluster3DParser(
        dtype="float32",
        cluster_event="cluster",
        particle_info=None,
    )

    result = parser({"cluster": cluster3d_event})

    assert isinstance(result, ClusterLabelData)
    assert result.features.shape[1] == 2
    assert result.particles is None


@pytest.mark.parametrize(
    ("particle_info", "kwargs", "message"),
    [
        ({"particle_event": "nested"}, {"particle_event": "legacy"}, "specified twice"),
        (
            {"particle_mpv_event": "nested"},
            {"particle_mpv_event": "legacy", "particle_event": "particle"},
            "specified twice",
        ),
        (
            {"neutrino_event": "nested"},
            {"neutrino_event": "legacy", "particle_event": "particle"},
            "specified twice",
        ),
        (
            {"type_include_secondary": False},
            {"type_include_secondary": True, "particle_event": "particle"},
            "specified twice",
        ),
        (
            {"label_le": False},
            {"label_le": True, "particle_event": "particle"},
            "specified twice",
        ),
        (
            {"unexpected": True},
            {"particle_event": "particle"},
            "Unknown particle information option",
        ),
    ],
)
def test_cluster3d_rejects_ambiguous_particle_configuration(
    particle_info, kwargs, message
):
    """Nested particle configuration should reject duplicates and unknowns."""
    with pytest.raises(ValueError, match=message):
        LArCVCluster3DParser(
            dtype="float32",
            cluster_event="cluster",
            particle_info=particle_info,
            **kwargs,
        )


def test_cluster3d_accepts_legacy_boolean_particle_configuration():
    """The legacy boolean particle-info switch should remain supported."""
    parser = LArCVCluster3DParser(
        dtype="float32",
        cluster_event="cluster",
        particle_event="particle",
        particle_info=True,
    )
    assert parser.include_particle_info is True

    with pytest.raises(ValueError, match="particle_event"):
        LArCVCluster3DParser(
            dtype="float32",
            cluster_event="cluster",
            particle_info=True,
        )


def test_cluster3d_registers_nested_optional_particle_inputs():
    """Nested MPV and neutrino products should become parser inputs."""
    parser = LArCVCluster3DParser(
        dtype="float32",
        cluster_event="cluster",
        particle_info={
            "particle_event": "particle",
            "particle_mpv_event": "particle_mpv",
            "neutrino_event": "neutrino",
        },
    )

    assert parser.data_map["particle_mpv_event"] == "particle_mpv"
    assert parser.data_map["neutrino_event"] == "neutrino"


@pytest.mark.parametrize("cluster3d_event, particle_event", [(1, 20)], indirect=True)
def test_cluster3d_rejects_misaligned_particle_count(cluster3d_event, particle_event):
    """Particle and cluster collections must remain row-aligned."""
    parser = LArCVCluster3DParser(
        dtype="float32",
        cluster_event="cluster",
        particle_event="particle",
        particle_info=True,
    )
    with pytest.raises(ValueError, match="aligned with the number of clusters"):
        parser.process(cluster_event=cluster3d_event, particle_event=particle_event)


@pytest.mark.parametrize("cluster3d_event", [20], indirect=True)
def test_cluster3d_cleaning_requires_semantics(cluster3d_event):
    """Value-based cleaning should require a semantic reference tensor."""
    sparse_value = cluster3d_to_sparse3d(cluster3d_event)
    parser = LArCVCluster3DParser(
        dtype="float32",
        cluster_event="cluster",
        sparse_value_event="value",
        clean_data=True,
    )
    with pytest.raises(ValueError, match="semantics tensor is required"):
        parser.process(
            cluster_event=cluster3d_event,
            sparse_value_event=sparse_value,
        )


@pytest.mark.parametrize("cluster3d_event, particle_event", [(20, 20)], indirect=True)
@pytest.mark.parametrize("neutrino_event", [1], indirect=True)
@pytest.mark.parametrize("add_particle_info, clean_data", [(True, True)])
@pytest.mark.parametrize("break_clusters", [True])
@pytest.mark.filterwarnings(
    "ignore:Neutrino IDs are being produced on the basis of floating point agreement.*:UserWarning"
)
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
    parser = LArCVCluster3DChargeRescaledParser(
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
    assert isinstance(result, ClusterLabelData)
    assert result.coords.shape[1] == 3
    assert result.features.shape[1] == (3 if add_particle_info else 2)
    assert result.particles is not None
    assert isinstance(result.meta, ImageMeta3D)


@pytest.mark.parametrize("cluster3d_event, particle_event", [(20, 20)], indirect=True)
@pytest.mark.parametrize("neutrino_event", [1], indirect=True)
@pytest.mark.parametrize("add_particle_info, clean_data", [(True, True)])
@pytest.mark.parametrize("break_clusters", [True])
@pytest.mark.filterwarnings(
    "ignore:Neutrino IDs are being produced on the basis of floating point agreement.*:UserWarning"
)
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
    parser = LArCVCluster3DAggregateParser(
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
    assert isinstance(result, ClusterLabelData)
    assert result.coords.shape[1] == 3
    assert result.features.shape[1] == (3 if add_particle_info else 2)
    assert result.particles is not None
    assert isinstance(result.meta, ImageMeta3D)


def cluster3d_to_sparse3d(
    cluster3d_event, segmentation=False, ghost=True, segmentation_value=None
):
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
            elif segmentation_value is not None:
                values.append(np.full(num_points, segmentation_value, dtype=np.float32))
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


@pytest.mark.parametrize("cluster3d_event, particle_event", [(20, 20)], indirect=True)
def test_cluster3d_clean_data_requires_semantics(cluster3d_event, particle_event):
    """Cleaning cluster labels should require a semantic reference tensor."""
    parser = LArCVCluster3DParser(
        dtype="float32",
        cluster_event=cluster3d_event,
        particle_event=particle_event,
        clean_data=True,
        add_particle_info=True,
    )

    with pytest.raises(ValueError, match="semantics tensor"):
        parser.process(cluster_event=cluster3d_event, particle_event=particle_event)


@pytest.mark.parametrize("cluster3d_event, particle_event", [(20, 20)], indirect=True)
def test_cluster3d_label_le_controls_raw_cluster_labels(
    cluster3d_event, particle_event
):
    """Low-energy clusters should only retain labels when explicitly enabled."""
    for particle in particle_event.as_vector():
        particle.shape(LOWES_SHP)

    results = []
    for label_le in (False, True):
        parser = LArCVCluster3DParser(
            dtype="float32",
            cluster_event=cluster3d_event,
            particle_event=particle_event,
            particle_info={"label_le": label_le},
        )
        results.append(
            parser.process(cluster_event=cluster3d_event, particle_event=particle_event)
        )

    part_col = PART_COL - VALUE_COL
    assert np.all(results[0].features[:, part_col] == -1)
    assert np.any(results[1].features[:, part_col] > -1)


@pytest.mark.parametrize("cluster3d_event, particle_event", [(20, 20)], indirect=True)
def test_cluster3d_label_le_controls_cleaned_labels(cluster3d_event, particle_event):
    """Semantic LE voxels should only retain labels when explicitly enabled."""
    for particle in particle_event.as_vector():
        particle.shape(0)
    semantics = cluster3d_to_sparse3d(
        cluster3d_event, segmentation=True, ghost=False, segmentation_value=LOWES_SHP
    )
    results = []
    for label_le in (False, True):
        parser = LArCVCluster3DParser(
            dtype="float32",
            cluster_event=cluster3d_event,
            particle_event=particle_event,
            sparse_semantics_event=semantics,
            clean_data=True,
            particle_info={"label_le": label_le},
        )
        results.append(
            parser.process(
                cluster_event=cluster3d_event,
                particle_event=particle_event,
                sparse_semantics_event=semantics,
            )
        )

    part_col = PART_COL - VALUE_COL
    assert np.all(results[0].features[:, part_col] == -1)
    assert np.any(results[1].features[:, part_col] > -1)


@pytest.mark.parametrize(
    "cluster2d_event, cluster3d_event, particle_event, neutrino_event",
    [(1, 20, 20, 1)],
    indirect=True,
)
@pytest.mark.filterwarnings(
    "ignore:Neutrino IDs are being produced on the basis of floating point agreement.*:UserWarning"
)
def test_cluster_parser_call_paths(
    cluster2d_event, cluster3d_event, particle_event, neutrino_event
):
    """Wrapper calls should route named inputs through the cluster parsers."""
    sparse3d_seg_event = cluster3d_to_sparse3d(cluster3d_event, True, False)
    sparse3d_event_list = [cluster3d_to_sparse3d(cluster3d_event)] * 2
    sparse3d_rescale_list = [cluster3d_to_sparse3d(cluster3d_event)] * 6 + [
        sparse3d_seg_event
    ]

    cluster2d_parser = LArCVCluster2DParser(
        dtype="float32", cluster_event="cluster2d", projection_id=0
    )
    assert isinstance(cluster2d_parser({"cluster2d": cluster2d_event}), TensorData)

    aggregate_parser = LArCVCluster3DAggregateParser(
        dtype="float32",
        value_aggr="max",
        cluster_event="cluster3d",
        particle_event="particle",
        neutrino_event="neutrino",
        sparse_value_event_list=["value_0", "value_1"],
        sparse_semantics_event="semantics",
        clean_data=True,
        add_particle_info=True,
    )
    assert isinstance(
        aggregate_parser(
            {
                "cluster3d": cluster3d_event,
                "particle": particle_event,
                "neutrino": neutrino_event,
                "value_0": sparse3d_event_list[0],
                "value_1": sparse3d_event_list[1],
                "semantics": sparse3d_seg_event,
            }
        ),
        ClusterLabelData,
    )

    rescale_parser = LArCVCluster3DChargeRescaledParser(
        dtype="float32",
        cluster_event="cluster3d",
        particle_event="particle",
        neutrino_event="neutrino",
        sparse_value_event_list=[f"value_{i}" for i in range(7)],
        sparse_semantics_event="semantics",
        clean_data=True,
        add_particle_info=True,
    )
    assert isinstance(
        rescale_parser(
            {
                "cluster3d": cluster3d_event,
                "particle": particle_event,
                "neutrino": neutrino_event,
                "semantics": sparse3d_seg_event,
                **{f"value_{i}": sparse3d_rescale_list[i] for i in range(7)},
            }
        ),
        ClusterLabelData,
    )


@pytest.mark.parametrize("cluster3d_event, particle_event", [(20, 20)], indirect=True)
def test_cluster3d_add_particle_info_special_cases(cluster3d_event, particle_event):
    """Cluster parsing should cover secondary/MPR masking, auto-clean, and inferred neutrino count."""
    sparse3d_seg_event = cluster3d_to_sparse3d(cluster3d_event, True, False)
    sparse3d_event = cluster3d_to_sparse3d(cluster3d_event)

    parser = LArCVCluster3DParser(
        dtype="float32",
        cluster_event=cluster3d_event,
        particle_event=particle_event,
        sparse_value_event=sparse3d_event,
        sparse_semantics_event=sparse3d_seg_event,
        clean_data=False,
        add_particle_info=True,
        type_include_secondary=False,
        type_include_mpr=False,
        primary_include_mpr=False,
    )
    with pytest.warns(UserWarning) as caught:
        result = parser.process(
            cluster_event=cluster3d_event,
            particle_event=particle_event,
            sparse_value_event=sparse3d_event,
            sparse_semantics_event=sparse3d_seg_event,
        )

    assert isinstance(result, ClusterLabelData)
    assert parser.clean_data is True
    warning_messages = [str(w.message) for w in caught]
    assert any("interaction multiplicity" in message for message in warning_messages)
    assert any(
        "You must set `clean_data` to `True`" in message for message in warning_messages
    )


@pytest.mark.parametrize("cluster3d_event, particle_event", [(20, 20)], indirect=True)
def test_cluster3d_resolves_ancestor_targets(cluster3d_event, particle_event):
    """Ancestor PID and momentum should resolve through the particle table."""
    particles = list(particle_event.as_vector())
    root = particles[0]
    root.shape(1)
    root.pdg_code(13)
    root.momentum(3.0, 4.0, 0.0)
    for particle in particles:
        particle.ancestor_track_id(root.track_id())

    parser = LArCVCluster3DParser(
        dtype="float32",
        cluster_event=cluster3d_event,
        particle_event=particle_event,
        add_particle_info=True,
    )
    result = parser.process(
        cluster_event=cluster3d_event,
        particle_event=particle_event,
    )

    data = np.concatenate((result.coords, result.features), axis=1)
    labels = ClusterLabelData(data, result.particles, result.meta)
    valid = labels.voxel_field("particle") >= 0
    ancestor_pids = labels.voxel_field("ancestor_pid")[valid]
    ancestor_momenta = labels.voxel_field("ancestor_momentum")[valid]
    assert len(ancestor_pids) > 0
    assert np.all(ancestor_pids == 2)
    assert np.allclose(ancestor_momenta, 5.0)
