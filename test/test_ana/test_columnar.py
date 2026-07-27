"""End-to-end tests for configurable columnar analysis execution."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spine.data import ObjectList, RecoParticle, RunInfo, TruthParticle
from spine.driver import Driver
from spine.io.write import HDF5Writer


def _config(path: Path, log_dir: Path, columnar: bool) -> dict:
    return {
        "base": {
            "iterations": -1,
            "log_dir": str(log_dir),
            "overwrite_log": True,
            "verbosity": "critical",
        },
        "io": {
            "reader": {
                "name": "hdf5",
                "file_keys": str(path),
                "columnar": columnar,
                "chunk_size": 2,
            }
        },
        "ana": {
            "overwrite": True,
            "save": {
                "obj_type": "particle",
                "particle": ["id", "pid", "size"],
                "run_mode": "both",
                "match_mode": "reco_to_truth",
            },
        },
    }


def _run(config: dict) -> None:
    driver = Driver(config)
    assert driver.iterations is not None
    for iteration in range(driver.iterations):
        driver.process(
            entry=iteration,
            iteration=iteration,
            epoch=(iteration + 1) / driver.io.iter_per_epoch,
        )
    driver.cleanup()


def test_configured_event_and_columnar_save_are_identical(tmp_path):
    """Reader policy should switch execution without changing CSV content."""
    path = tmp_path / "objects.h5"
    truth = TruthParticle(
        id=0,
        pid=2,
        index=np.asarray([1, 2], dtype=np.int32),
    )
    reco = RecoParticle(
        id=0,
        pid=2,
        index=np.asarray([1, 2], dtype=np.int32),
        is_matched=True,
        match_ids=np.asarray([0], dtype=np.int32),
        match_overlaps=np.asarray([0.75], dtype=np.float32),
    )
    data = {
        "index": np.asarray([0, 1]),
        "run_info": [
            RunInfo(run=1, event=10),
            RunInfo(run=1, event=11),
        ],
        "reco_particles": [
            ObjectList([reco], RecoParticle()),
            ObjectList([], RecoParticle()),
        ],
        "truth_particles": [
            ObjectList([truth], TruthParticle()),
            ObjectList([], TruthParticle()),
        ],
    }
    with HDF5Writer(str(path), overwrite=True, format_version=2) as writer:
        writer(data, cfg={})

    event_dir = tmp_path / "event"
    columnar_dir = tmp_path / "columnar"
    _run(_config(path, event_dir, False))
    _run(_config(path, columnar_dir, True))

    for name in ("save_reco_particles.csv", "save_truth_particles.csv"):
        assert (event_dir / name).read_bytes() == (columnar_dir / name).read_bytes()
