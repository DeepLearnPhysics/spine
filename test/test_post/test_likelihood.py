from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from spine.post.optical import likelihood as likelihood_mod


class FakeVector:
    def __init__(self):
        self.values = []

    def push_back(self, value):
        self.values.append(value)


class FakePoint:
    def __init__(self):
        self.x = 1.0
        self.y = 2.0
        self.z = 3.0


class FakeMatch:
    def __init__(self, tpc_id=0, flash_id: int | None = 0):
        self.tpc_id = tpc_id
        self.flash_id = flash_id
        self.tpc_point = FakePoint()
        self.score = 0.5
        self.hypothesis = [1.0, 2.0]


class FakeQCluster:
    def __init__(self):
        self.idx = -1
        self.time = -1
        self.payload = []

    def __iadd__(self, other):
        self.payload.append(other)
        return self


class FakeFlash:
    def __init__(self):
        self.idx = -1
        self.time = -1.0
        self.pe_v = FakeVector()
        self.pe_err_v = FakeVector()


class FakeManager:
    def __init__(self):
        self.objects = []
        self.config = None
        self.reset = False

    def Configure(self, cfg):
        self.config = cfg

    def Reset(self):
        self.reset = True
        self.objects.clear()

    def Add(self, obj):
        self.objects.append(obj)

    def Match(self):
        return [FakeMatch()]


class FakeLightPath:
    def __init__(self):
        self.config = None
        self.calls = []

    def Configure(self, cfg):
        self.config = cfg

    def MakeQCluster(self, *args):
        self.calls.append(args)
        return {"args": args}


class FakeFactory:
    def __init__(self, light_path):
        self.light_path = light_path

    def create(self, *args):
        return self.light_path


class FakeFactoryGetter:
    def __init__(self, light_path):
        self.factory = FakeFactory(light_path)

    def get(self):
        return self.factory


class FakeCfgGetter:
    def __getitem__(self, key):
        return lambda name: {"key": key, "name": name}


class FakeCfg:
    get = FakeCfgGetter()


class FakeDetectorSpecs:
    loaded = None

    @classmethod
    def GetME(cls, path):
        cls.loaded = path


class FakeFlashMatch:
    def __init__(self):
        self.light_path = FakeLightPath()
        self.DetectorSpecs = FakeDetectorSpecs
        self.CustomAlgoFactory = FakeFactoryGetter(self.light_path)
        self.manager = FakeManager()
        self.created_cfg = None

    def CreateFMParamsFromFile(self, path):
        self.created_cfg = path
        return FakeCfg()

    def FlashMatchManager(self):
        return self.manager

    def QCluster_t(self):
        return FakeQCluster()

    def Flash_t(self):
        return FakeFlash()

    def as_geoalgo_trajectory(self, array):
        return array

    def as_ndarray(self, obj):
        return np.array([obj.idx], dtype=np.float32)


def configure_fake_backend(monkeypatch, tmp_path):
    basedir = tmp_path / "fmatch"
    (basedir / "dat").mkdir(parents=True)
    (basedir / "build" / "lib").mkdir(parents=True)
    (basedir / "dat" / "detector_specs_demo.cfg").write_text("detector")
    cfg = tmp_path / "flash.cfg"
    cfg.write_text("cfg")
    monkeypatch.setenv("FMATCH_BASEDIR", str(basedir))
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.delenv("FMATCH_DATADIR", raising=False)
    fake = FakeFlashMatch()
    monkeypatch.setattr(likelihood_mod, "get_flashmatch", lambda: fake)
    return fake, cfg


def make_matcher(monkeypatch, tmp_path, **kwargs):
    fake, cfg = configure_fake_backend(monkeypatch, tmp_path)
    matcher = likelihood_mod.LikelihoodFlashMatcher(
        cfg=str(cfg),
        detector="demo",
        scaling=cast(Any, "2.0"),
        alpha=cast(Any, "0.2"),
        recombination_mip=cast(Any, "0.6"),
        **kwargs,
    )
    return matcher, fake


def test_likelihood_flash_matcher_backend_validation(monkeypatch, tmp_path):
    monkeypatch.delenv("FMATCH_BASEDIR", raising=False)
    with pytest.raises(ValueError, match="FMATCH_BASEDIR"):
        likelihood_mod.LikelihoodFlashMatcher(cfg="missing.cfg", detector="demo")

    basedir = tmp_path / "fmatch"
    (basedir / "dat").mkdir(parents=True)
    (basedir / "build" / "lib").mkdir(parents=True)
    monkeypatch.setenv("FMATCH_BASEDIR", str(basedir))
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    with pytest.raises(FileNotFoundError, match="detector"):
        likelihood_mod.LikelihoodFlashMatcher(cfg="missing.cfg", detector="demo")

    (basedir / "dat" / "detector_specs_demo.cfg").write_text("detector")
    monkeypatch.setattr(likelihood_mod, "get_flashmatch", lambda: FakeFlashMatch())
    with pytest.raises(FileNotFoundError, match="flash-matcher"):
        likelihood_mod.LikelihoodFlashMatcher(cfg="missing.cfg", detector="demo")


def test_get_flashmatch_loads_optional_module(monkeypatch):
    fake = SimpleNamespace(flashmatch=FakeFlashMatch())
    monkeypatch.setitem(sys.modules, "flashmatch", fake)

    assert likelihood_mod.get_flashmatch() is fake.flashmatch


def test_likelihood_flash_matcher_initializes_backend(monkeypatch, tmp_path):
    fake, cfg = configure_fake_backend(monkeypatch, tmp_path)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    rel_cfg = cfg_dir / "rel.cfg"
    rel_cfg.write_text("cfg")

    matcher = likelihood_mod.LikelihoodFlashMatcher(
        cfg="rel.cfg", detector="demo", parent_path=str(cfg_dir)
    )

    assert fake.DetectorSpecs.loaded is not None
    assert fake.DetectorSpecs.loaded.endswith("detector_specs_demo.cfg")
    assert fake.created_cfg == str(rel_cfg)
    assert matcher.mgr is fake.manager
    assert fake.light_path.config == {
        "key": "flashmatch::FMParams",
        "name": "LightPath",
    }


def test_likelihood_flash_matcher_uses_default_detector_specs(monkeypatch, tmp_path):
    basedir = tmp_path / "fmatch"
    (basedir / "dat").mkdir(parents=True)
    (basedir / "build" / "lib").mkdir(parents=True)
    (basedir / "dat" / "detector_specs.cfg").write_text("detector")
    cfg = tmp_path / "flash.cfg"
    cfg.write_text("cfg")
    monkeypatch.setenv("FMATCH_BASEDIR", str(basedir))
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    fake = FakeFlashMatch()
    monkeypatch.setattr(likelihood_mod, "get_flashmatch", lambda: fake)

    likelihood_mod.LikelihoodFlashMatcher(cfg=str(cfg), detector=None)

    assert fake.DetectorSpecs.loaded is not None
    assert fake.DetectorSpecs.loaded.endswith("detector_specs.cfg")


def test_likelihood_flash_matcher_makes_qclusters_and_flashes(monkeypatch, tmp_path):
    matcher, fake = make_matcher(monkeypatch, tmp_path)
    interactions = [
        SimpleNamespace(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            depositions=np.array([1.0, -1.0, 2.0], dtype=np.float32),
        ),
        SimpleNamespace(
            points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            depositions=np.array([1.0], dtype=np.float32),
        ),
    ]
    flashes = [SimpleNamespace(time=4.0, pe_per_ch=np.array([1.0, 2.0]))]

    qclusters = matcher.make_qcluster_list(interactions)
    flash_v, returned_flashes = matcher.make_flash_list(flashes)

    assert len(qclusters) == 1
    assert qclusters[0].idx == 0
    assert len(fake.light_path.calls) == 1
    assert flash_v[0].idx == 0
    assert flash_v[0].time == 4.0
    assert flash_v[0].pe_v.values == [1.0, 2.0]
    assert returned_flashes is flashes


def test_likelihood_flash_matcher_legacy_qcluster(monkeypatch, tmp_path):
    matcher, fake = make_matcher(monkeypatch, tmp_path, legacy=True)
    interactions = [
        SimpleNamespace(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            depositions=np.array([1.0, 2.0], dtype=np.float32),
        )
    ]

    matcher.make_qcluster_list(interactions)

    assert len(fake.light_path.calls[0]) == 2


def test_likelihood_flash_matcher_runs_and_fetches_results(monkeypatch, tmp_path):
    matcher, fake = make_matcher(monkeypatch, tmp_path)
    interactions = [
        SimpleNamespace(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            depositions=np.array([1.0, 2.0], dtype=np.float32),
        )
    ]
    flashes = [SimpleNamespace(time=4.0, pe_per_ch=np.array([1.0]))]

    result = matcher.get_matches(interactions, flashes)

    assert result[0][0] is interactions[0]
    assert result[0][1] is flashes[0]
    assert fake.manager.reset is True
    assert len(fake.manager.objects) == 2
    assert matcher.qcluster_v is not None
    assert matcher.flash_v is not None
    assert matcher.matches is not None
    assert matcher.get_qcluster(0) is matcher.qcluster_v[0]
    assert np.array_equal(matcher.get_qcluster(0, array=True), np.array([0.0]))
    assert matcher.get_flash(0) is matcher.flash_v[0]
    assert np.array_equal(matcher.get_flash(0, array=True), np.array([0.0]))
    assert matcher.get_match(0) is matcher.matches[0]
    assert matcher.get_matched_flash(0) is matcher.flash_v[0]
    assert matcher.get_t0(0) == 4.0


def test_likelihood_flash_matcher_getter_errors(monkeypatch, tmp_path):
    matcher, _ = make_matcher(monkeypatch, tmp_path)

    assert matcher.get_matches([], []) == []

    with pytest.raises(ValueError, match="qcluster_v"):
        matcher.get_qcluster(0)
    with pytest.raises(ValueError, match="flash_v"):
        matcher.get_flash(0)
    with pytest.raises(ValueError, match="run flash matching"):
        matcher.get_match(0)
    with pytest.raises(ValueError, match="run flash matching"):
        matcher.get_matched_flash(0)
    with pytest.raises(ValueError, match="make_qcluster_list"):
        matcher.run_flash_matching()

    matcher.qcluster_v = [FakeQCluster()]
    matcher.qcluster_v[0].idx = 3
    matcher.flash_v = [FakeFlash()]
    matcher.flash_v[0].idx = 4
    matcher.matches = [FakeMatch(tpc_id=0, flash_id=None)]

    with pytest.raises(IndexError, match="TPC"):
        matcher.get_qcluster(0)
    with pytest.raises(IndexError, match="Flash"):
        matcher.get_flash(0)
    assert matcher.get_match(0) is None
    assert matcher.get_matched_flash(0) is None
    assert matcher.get_t0(0) is None

    matcher.qcluster_v[0].idx = 0
    assert matcher.get_matched_flash(0) is None

    matcher.qcluster_v[0].idx = 3
    matcher.matches = [FakeMatch(tpc_id=0, flash_id=10)]
    with pytest.raises(IndexError, match="Flash"):
        matcher.get_matched_flash(3)
