from __future__ import annotations

import pytest

from spine.post.base import PostBase


class DummyPost(PostBase):
    name = "dummy"

    def process(self, data):
        return {"updated": data["value"] + 1}


def test_post_base_validates_configuration():
    with pytest.raises(TypeError, match="obj_type"):
        DummyPost(obj_type=1)

    with pytest.raises(ValueError, match="run_mode"):
        DummyPost(run_mode="bad")

    with pytest.raises(ValueError, match="Object type"):
        DummyPost(obj_type="bad")

    with pytest.raises(ValueError, match="truth_point_mode"):
        DummyPost(truth_point_mode="bad")

    with pytest.raises(ValueError, match="incompatible"):
        DummyPost(truth_point_mode="points_adapt", truth_dep_mode="depositions_g4")

    with pytest.raises(ValueError, match="truth_dep_mode"):
        DummyPost(truth_dep_mode="bad")

    with pytest.raises(ValueError, match="pid_mode"):
        DummyPost(pid_mode="bad")


def test_post_base_filters_entry_and_reports_missing_required_input():
    post = DummyPost()
    post.update_keys({"value": True})

    result = post({"value": [9]}, entry=0)

    assert result == {"updated": 10}
    with pytest.raises(KeyError, match="missing an essential"):
        post({"index": 0})


def test_post_base_object_keys_and_upstream():
    post = DummyPost(obj_type=("particle", "interaction"), run_mode="reco")
    post.update_upstream("source")

    assert post.obj_keys == ["reco_particles", "reco_interactions"]
    assert post._upstream == ("source",)


def test_post_base_truth_accessors():
    post = DummyPost(
        truth_point_mode="points",
        truth_dep_mode="depositions",
        pid_mode="chi2_pid",
    )

    class TruthObject:
        is_truth = True
        points = "truth_points"
        sources = "truth_sources"
        depositions = "truth_depositions"
        index = "truth_index"
        pid = 2
        units = "cm"

    truth = TruthObject()

    assert post.get_points(truth) == "truth_points"
    assert post.get_sources(truth) == "truth_sources"
    assert post.get_depositions(truth) == "truth_depositions"
    assert post.get_index(truth) == "truth_index"
    assert post.get_pid(truth) == 2
    post.check_units(truth)


def test_post_base_reco_accessors_and_units():
    post = DummyPost(pid_mode="chi2_pid")

    class RecoObject:
        is_truth = False
        points = "reco_points"
        sources = "reco_sources"
        depositions = "reco_depositions"
        index = "reco_index"
        chi2_pid = 3
        units = "px"

    reco = RecoObject()

    assert post.get_points(reco) == "reco_points"
    assert post.get_sources(reco) == "reco_sources"
    assert post.get_depositions(reco) == "reco_depositions"
    assert post.get_index(reco) == "reco_index"
    assert post.get_pid(reco) == 3
    with pytest.raises(ValueError, match="Coordinates"):
        post.check_units(reco)
