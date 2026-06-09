import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def value_db(tmp_path):
    """Create a minimal ICARUS-style value calibration database."""
    path = tmp_path / "icarus_gain_v1.db"
    stem = path.stem
    conn = sqlite3.connect(path)
    conn.execute(
        f"CREATE TABLE {stem}_iovs "
        "(iov_id INTEGER, begin_time INTEGER, active INTEGER)"
    )
    conn.execute(
        f"CREATE TABLE {stem}_data " "(__iov_id INTEGER, channel INTEGER, gain REAL)"
    )
    conn.executemany(
        f"INSERT INTO {stem}_iovs VALUES (?, ?, ?)",
        [(1, 1000000100, 1), (2, 1000000200, 1), (3, 1000000300, 0)],
    )
    conn.executemany(
        f"INSERT INTO {stem}_data VALUES (?, ?, ?)",
        [
            (1, 0, 2.0),
            (1, 1, 3.0),
            (2, 0, 4.0),
            (2, 1, 5.0),
            (3, 0, 9.0),
            (3, 1, 9.0),
        ],
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def transparency_db(tmp_path):
    """Create a minimal ICARUS-style map calibration database."""
    path = tmp_path / "icarus_transparency_v1.db"
    stem = path.stem
    conn = sqlite3.connect(path)
    conn.execute(
        f"CREATE TABLE {stem}_iovs "
        "(iov_id INTEGER, begin_time INTEGER, active INTEGER)"
    )
    conn.execute(
        f"CREATE TABLE {stem}_data "
        "(__iov_id INTEGER, tpc TEXT, ybin INTEGER, zbin INTEGER, "
        "ylow REAL, yhigh REAL, zlow REAL, zhigh REAL, scale REAL)"
    )
    conn.execute(f"INSERT INTO {stem}_iovs VALUES (?, ?, ?)", (1, 1000000100, 1))
    rows = []
    for tpc_id, tpc in enumerate(("EE", "EW", "WE", "WW")):
        for ybin in range(2):
            for zbin in range(2):
                rows.append(
                    (
                        1,
                        tpc,
                        ybin,
                        zbin,
                        float(ybin),
                        float(ybin + 1),
                        float(zbin),
                        float(zbin + 1),
                        float(tpc_id + ybin + zbin + 1),
                    )
                )
    conn.executemany(
        f"INSERT INTO {stem}_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    conn.close()
    return path


class FakeTPC:
    def __init__(self, anode_pos, drift_axis=0, dimensions=(10.0, 10.0, 10.0)):
        self.anode_pos = anode_pos
        self.drift_axis = drift_axis
        self.drift_dir = np.array([1.0, 0.0, 0.0])
        self.dimensions = np.asarray(dimensions, dtype=float)


class FakeTPCSet:
    num_modules = 1
    num_chambers_per_module = 2
    num_chambers = 2

    def __init__(self):
        self._tpcs = [[FakeTPC(0.0), FakeTPC(10.0)]]

    def __getitem__(self, index):
        return self._tpcs[index]


class FakeGeo:
    def __init__(self):
        self.tpc = FakeTPCSet()

    def get_volume_index(self, sources, module_id, tpc_id):
        return np.where((sources[:, 0] == module_id) & (sources[:, 1] == tpc_id))[0]

    def get_closest_tpc_indexes(self, points):
        return [np.where(points[:, 0] <= 5.0)[0], np.where(points[:, 0] > 5.0)[0]]

    def translate(self, points, source_module, target_module):
        return points + float(target_module - source_module)


@pytest.fixture
def fake_geo():
    return FakeGeo()
