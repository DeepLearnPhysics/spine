from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spine.post.truth.match import MatchProcessor


def make_obj(
    index=(0,),
    points=((0.0, 0.0, 0.0),),
    is_truth=False,
    units="px",
    orig_index=None,
):
    index = np.asarray(index, dtype=np.int64)
    points = np.asarray(points, dtype=np.float32)
    obj = SimpleNamespace(
        is_truth=is_truth,
        units=units,
        index=index,
        points=points,
        points_g4=points,
        points_adapt=points,
        orig_index=(
            index if orig_index is None else np.asarray(orig_index, dtype=np.int64)
        ),
    )
    return obj


class FakeMeta:
    def to_px(self, coords, floor=True):
        return np.floor(coords).astype(np.int64) if floor else coords

    def index(self, coords):
        coords = np.asarray(coords, dtype=np.int64)
        return coords[:, 0] + 10 * coords[:, 1] + 100 * coords[:, 2]


def test_match_processor_validates_configuration():
    with pytest.raises(AssertionError, match="Must specify"):
        MatchProcessor()

    with pytest.raises(AssertionError, match="Invalid matching mode"):
        MatchProcessor(particle={"match_mode": "closest"})

    with pytest.raises(AssertionError, match="Invalid overlap"):
        MatchProcessor(particle={"overlap_mode": "distance"})

    with pytest.raises(AssertionError, match="Only IoU"):
        MatchProcessor(particle={"overlap_mode": "count", "weight_overlap": True})


def test_prepare_overlap_index_preserves_sorted_and_uniques_unsorted():
    sorted_index = np.array([1, 3, 5], dtype=np.int32)
    unsorted_index = np.array([3, 1, 3], dtype=np.int32)

    assert np.array_equal(MatchProcessor.prepare_overlap_index(sorted_index), [1, 3, 5])
    assert np.array_equal(MatchProcessor.prepare_overlap_index(unsorted_index), [1, 3])


def test_match_processor_matches_particles_by_index():
    reco = make_obj(index=(0, 1), is_truth=False)
    truth = make_obj(index=(1, 2), is_truth=True)
    processor = MatchProcessor(particle=True)

    result = processor.process({"reco_particles": [reco], "truth_particles": [truth]})

    assert result["particle_matches_r2t"] == [(reco, truth)]
    assert result["particle_matches_t2r"] == [(truth, reco)]
    assert reco.is_matched is True
    assert np.array_equal(reco.match_ids, np.array([0], dtype=np.int32))
    assert reco.match_overlaps[0] > 0.0


def test_match_processor_handles_empty_and_directional_modes():
    reco = make_obj(index=(0,), is_truth=False)
    processor = MatchProcessor(particle={"match_mode": "reco_to_truth"})

    result = processor.process({"reco_particles": [reco], "truth_particles": []})

    assert result["particle_matches_r2t"] == [(reco, None)]
    assert result["particle_matches_r2t_overlap"] == [-1.0]
    assert "particle_matches_t2r" not in result
    assert reco.is_matched is False

    truth = make_obj(index=(0,), is_truth=True)
    processor = MatchProcessor(particle={"match_mode": "truth_to_reco"})
    result = processor.process({"reco_particles": [], "truth_particles": [truth]})
    assert result["particle_matches_t2r"] == [(truth, None)]
    assert "particle_matches_r2t" not in result


def test_match_processor_uses_g4_points_and_meta():
    reco = make_obj(
        points=((1.2, 0.0, 0.0),),
        is_truth=False,
        units="cm",
    )
    truth = make_obj(
        points=((1.0, 0.0, 0.0),),
        is_truth=True,
        units="cm",
    )
    processor = MatchProcessor(particle=True, truth_point_mode="points_g4")

    with pytest.raises(AssertionError, match="metadata"):
        processor.process({"reco_particles": [reco], "truth_particles": [truth]})

    result = processor.process(
        {"reco_particles": [reco], "truth_particles": [truth], "meta": FakeMeta()}
    )

    assert result["particle_matches_r2t"] == [(reco, truth)]


def test_match_processor_handles_ghost_orig_index_and_meta_fallback():
    reco = make_obj(index=(0,), orig_index=(5,), is_truth=False)
    truth = make_obj(index=(0,), orig_index=(5,), is_truth=True)
    processor = MatchProcessor(particle={"ghost": True})

    result = processor.process({"reco_particles": [reco], "truth_particles": [truth]})

    assert result["particle_matches_r2t"] == [(reco, truth)]

    bad = make_obj(index=(0, 1), orig_index=(5,), is_truth=False)
    with pytest.raises(AssertionError, match="orig_index"):
        processor.process({"reco_particles": [bad], "truth_particles": [truth]})

    processor = MatchProcessor(particle={"ghost": True, "use_orig_index": False})
    with pytest.raises(AssertionError, match="metadata"):
        processor.process({"reco_particles": [reco], "truth_particles": [truth]})

    reco.units = "cm"
    truth.units = "cm"
    result = processor.process(
        {"reco_particles": [reco], "truth_particles": [truth], "meta": FakeMeta()}
    )
    assert result["particle_matches_r2t"] == [(reco, truth)]


def test_match_processor_chamfer_distance_mode():
    reco = make_obj(points=((0.0, 0.0, 0.0),), is_truth=False)
    truth = make_obj(points=((0.0, 0.0, 0.0),), is_truth=True)
    far_truth = make_obj(points=((10.0, 0.0, 0.0),), is_truth=True)
    processor = MatchProcessor(particle={"overlap_mode": "chamfer", "min_overlap": 1.0})
    processor.matchers["particle"].fn = lambda reco, truth: np.array(
        [[0.0, 10.0]], dtype=np.float32
    )

    result = processor.process(
        {"reco_particles": [reco], "truth_particles": [truth, far_truth]}
    )

    assert result["particle_matches_r2t"] == [(reco, truth)]
    assert reco.match_overlaps[0] == pytest.approx(0.0)
