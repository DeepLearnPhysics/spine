"""Physical truth round-trip assertions shared by integration tests."""

import copy

import numpy as np

from spine.construct import BuildManager
from spine.io.collate import CollateAll
from spine.io.read import HDF5Reader
from spine.io.unwrap import Unwrapper
from spine.io.write import HDF5Writer


def check_truth_round_trip(events, path):
    """Build all truth levels in cm and retain their associations through HDF5.

    Parameters
    ----------
    events : list[dict]
        Parsed events with the complete native schema and source provenance.
    path : pathlib.Path
        Temporary output HDF5 file.

    Returns
    -------
    dict
        Unwrapped truth products used for physical validation by callers.
    """
    batch = CollateAll(data_keys=events[0])(events)
    truth = Unwrapper()(batch)
    original = copy.deepcopy(truth)
    BuildManager(True, True, True, mode="truth", units="cm")(truth)
    keys = [
        "meta",
        "event_info",
        "points_label",
        "depositions_label",
        "truth_fragments",
        "truth_particles",
        "truth_interactions",
    ]
    writer = HDF5Writer(file_name=str(path), keys=keys, format_version=2)
    writer(truth, {})
    writer.finalize()
    writer.close()
    reader = HDF5Reader(file_keys=str(path))
    for index, event in enumerate(events):
        restored = reader[index]
        BuildManager(True, True, True, mode="truth", units="cm")(restored)
        # BuildManager converts parsed truth to cm; voxel centers use +0.5.
        expected = event["meta"].to_cm(
            original["clust_label"][index].coords, center=True
        )
        np.testing.assert_allclose(restored["points_label"], expected)
        assert np.isclose(
            restored["depositions_label"].sum(), event["data"].features.sum()
        )
        for key in ("truth_fragments", "truth_particles", "truth_interactions"):
            assert len(restored[key]) == len(truth[key][index])
            for before, after in zip(truth[key][index], restored[key]):
                assert after.units == "cm"
                assert before.id == after.id
                np.testing.assert_allclose(before.points, after.points)
                for field in (
                    "parent_id",
                    "children_id",
                    "orig_id",
                    "pid",
                    "is_primary",
                    "vertex",
                    "energy_deposit",
                    "fragment_ids",
                    "particle_ids",
                ):
                    if hasattr(before, field):
                        np.testing.assert_allclose(
                            getattr(before, field), getattr(after, field)
                        )
        for particle in restored["truth_particles"]:
            source = event["particles"][particle.orig_id]
            assert particle.pid == source.pid
            assert particle.is_primary == (source.interaction_primary > 0)
        for particle in truth["truth_particles"][index]:
            for fragment in particle.fragments:
                assert fragment.particle_id == particle.id
        for interaction in truth["truth_interactions"][index]:
            for particle in interaction.particles:
                assert particle.interaction_id == interaction.id
    return truth
