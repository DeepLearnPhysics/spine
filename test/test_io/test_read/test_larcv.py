"""Tests for the LArCV reader."""

from collections import defaultdict

import pytest

import spine.io.read.larcv as larcv_read_module
from spine.io.read import LArCVReader
from spine.utils.conditional import ROOT, ROOT_AVAILABLE


class CountingLArCV:
    """Track accesses to the lazily resolved LArCV namespace."""

    def __init__(self):
        self.name_calls = 0

    def __getattribute__(self, name):
        if name == "__name__":
            object.__setattr__(self, "name_calls", self.name_calls + 1)
            return "larcv"
        return object.__getattribute__(self, name)


@pytest.mark.skipif(not ROOT_AVAILABLE, reason="ROOT is required to read LArCV files.")
def test_larcv_reader(larcv_data):
    """Tests the loading of a LArCV file."""
    # Get the list of tree keys in the larcv file
    root_file = ROOT.TFile(larcv_data, "r")
    num_entries = None
    tree_keys = []
    for tree in root_file.GetListOfKeys():
        tree_keys.append(tree.GetName().split("_tree")[0])
        if num_entries is None:
            num_entries = root_file.Get(tree.GetName()).GetEntries()

    root_file.Close()

    # Intialize the reader
    reader = LArCVReader(
        file_keys=larcv_data,
        tree_keys=tree_keys,
        create_run_map=True,
        run_info_key=tree_keys[0],
    )

    # Check that the number of events in the dataset is as expected
    assert reader.num_entries == num_entries

    # Load every entry, check that they contain what is expected
    for entry in reader:
        for key in tree_keys:
            assert key in entry

    # Check that the run map exists
    assert reader.run_map is not None
    assert reader.num_entries == len(reader.run_map)

    # Check all the available entry restriction modes
    reader.process_entry_list(n_entry=2)
    assert len(reader) == 2

    reader.process_entry_list(n_skip=2)
    assert len(reader) == reader.num_entries - 2

    reader.process_entry_list(n_entry=3, n_skip=2)
    assert len(reader) == 3

    reader.process_entry_list(entry_list=[1, 3, 4])
    assert len(reader) == 3

    reader.process_entry_list(skip_entry_list=[1, 3, 4])
    assert len(reader) == reader.num_entries - 3

    reader.process_entry_list(run_event_list=[tuple(reader.run_info[0])])
    reader.get_run_event(*reader.run_info[0])
    assert len(reader) == 1

    reader.process_entry_list(skip_run_event_list=[tuple(reader.run_info[0])])
    reader.get_run_event(*reader.run_info[1])
    assert len(reader) == reader.num_entries - 1

    # Try loading a file list
    reader = LArCVReader(file_keys=[larcv_data, larcv_data], tree_keys=tree_keys[:1])
    assert reader.num_entries == 2 * num_entries
    for i in range(len(reader)):
        reader[i]

    # Check that the internal indexing makes sense
    assert len(reader.file_index) == reader.num_entries
    assert len(reader.file_offsets) == 2
    assert reader.file_offsets[0] == 0
    assert reader.file_offsets[1] == num_entries

    # Check that the internals do not get compromised on sequential
    # restrictions when using a file list
    reader.process_entry_list(n_entry=2)
    assert len(reader) == 2

    reader.process_entry_list(n_skip=2)
    assert len(reader) == reader.num_entries - 2

    # Try to restrict the number of files to be loaded
    reader = LArCVReader(
        file_keys=[larcv_data, larcv_data], tree_keys=tree_keys[:1], limit_num_files=1
    )
    assert reader.num_entries == num_entries


@pytest.mark.skipif(not ROOT_AVAILABLE, reason="ROOT is required to read LArCV files.")
def test_larcv_reader_enables_run_map_for_run_event_filters(larcv_data):
    """Run-event restrictions should force run-map creation during initialization."""
    root_file = ROOT.TFile(larcv_data, "r")
    tree_keys = [tree.GetName().split("_tree")[0] for tree in root_file.GetListOfKeys()]
    root_file.Close()

    reader = LArCVReader(
        file_keys=larcv_data,
        tree_keys=tree_keys,
        run_info_key=tree_keys[0],
        run_event_list=[],
    )
    assert reader.run_map is not None


def test_larcv_reader_requires_dependencies_and_tree_keys(monkeypatch):
    """Reader initialization should guard missing dependencies and tree keys."""
    monkeypatch.setattr(
        LArCVReader,
        "process_file_paths",
        lambda self, *args, **kwargs: setattr(self, "file_paths", ["dummy.root"]),
    )

    class MissingModule:
        def __getattr__(self, name):
            raise RuntimeError(name)

    monkeypatch.setattr(larcv_read_module, "ROOT_AVAILABLE", False)
    monkeypatch.setattr(larcv_read_module, "LARCV_AVAILABLE", True)
    monkeypatch.setattr(larcv_read_module, "ROOT", MissingModule())
    monkeypatch.setattr(
        larcv_read_module, "larcv", type("DummyLArCV", (), {"__name__": "larcv"})()
    )
    with pytest.raises(ImportError, match="ROOT"):
        LArCVReader(file_keys="dummy.root", tree_keys=["sparse3d"])

    monkeypatch.setattr(larcv_read_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(larcv_read_module, "LARCV_AVAILABLE", False)
    monkeypatch.setattr(
        larcv_read_module, "ROOT", type("DummyROOT", (), {"__name__": "ROOT"})()
    )
    monkeypatch.setattr(larcv_read_module, "larcv", MissingModule())
    with pytest.raises(ImportError, match="larcv"):
        LArCVReader(file_keys="dummy.root", tree_keys=["sparse3d"])

    monkeypatch.setattr(larcv_read_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(larcv_read_module, "LARCV_AVAILABLE", True)
    monkeypatch.setattr(
        larcv_read_module, "ROOT", type("DummyROOT", (), {"__name__": "ROOT"})()
    )
    monkeypatch.setattr(
        larcv_read_module, "larcv", type("DummyLArCV", (), {"__name__": "larcv"})()
    )
    with pytest.raises(ValueError, match="tree_keys"):
        LArCVReader(file_keys="dummy.root", tree_keys=[])


def test_larcv_reader_list_data_filters_non_tree_keys(monkeypatch):
    """The tree scanner should ignore malformed and unsupported ROOT keys."""

    class DummyKey:
        def __init__(self, name):
            self.name = name

        def GetName(self):
            return self.name

    class DummyFile:
        def GetListOfKeys(self):
            return [
                DummyKey("not_a_branch"),
                DummyKey("particle_tree"),
                DummyKey("unknown_label_tree"),
                DummyKey("sparse3d_data_tree"),
                DummyKey("particle_label_tree"),
            ]

    class DummyTFile:
        @staticmethod
        def Open(*args, **kwargs):
            return DummyFile()

    class DummyROOTModule:
        TFile = DummyTFile

    monkeypatch.setattr(larcv_read_module, "ROOT", DummyROOTModule())
    result = LArCVReader.list_data("dummy.root")

    assert result == {
        "sparse3d": ["sparse3d_data"],
        "cluster3d": [],
        "particle": ["particle_label"],
    }


def test_larcv_reader_reinitializes_trees_on_pid_change(monkeypatch):
    """Cross-process reader use should rebuild process-local TChains."""

    created = defaultdict(list)

    class DummyChain:
        def __init__(self, name):
            self.name = name
            self.files = []
            self.entries = []
            self.sparse3d_branch = self
            created[name].append(self)

        def AddFile(self, path):
            self.files.append(path)

        def GetEntries(self):
            return len(self.files)

        def GetEntry(self, entry):
            self.entries.append(entry)

    dummy_root = type("DummyROOT", (), {"TChain": DummyChain})
    dummy_larcv = CountingLArCV()

    monkeypatch.setattr(larcv_read_module, "ROOT", dummy_root)
    monkeypatch.setattr(larcv_read_module, "larcv", dummy_larcv)

    reader = object.__new__(LArCVReader)
    reader.entry_index = [0]
    reader.file_paths = ["dummy.root"]
    reader.trees = {"sparse3d": None}
    reader.trees_ready = False
    reader.trees_pid = None
    reader.get_file_index = lambda idx: 0
    reader.get_file_entry_index = lambda idx: 0
    reader.get_source_provenance = lambda file_idx, file_entry_idx: {}

    pid_iter = iter([100, 200])
    monkeypatch.setattr(larcv_read_module.os, "getpid", lambda: next(pid_iter))

    first = reader.get(0)
    second = reader.get(0)

    assert first["sparse3d"] is created["sparse3d_tree"][0]
    assert second["sparse3d"] is created["sparse3d_tree"][1]
    assert len(created["sparse3d_tree"]) == 2
    assert dummy_larcv.name_calls == 2


def test_larcv_reader_rejects_mismatched_tree_entries(monkeypatch):
    """Reader initialization should reject tree lists with inconsistent sizes."""

    class DummyChain:
        def __init__(self, name):
            self.name = name
            self.entries = 0

        def AddFile(self, path):
            del path
            if self.name == "sparse3d_tree":
                self.entries += 2
            else:
                self.entries += 3

        def GetEntries(self):
            return self.entries

    monkeypatch.setattr(
        LArCVReader,
        "process_file_paths",
        lambda self, *args, **kwargs: setattr(self, "file_paths", ["dummy.root"]),
    )
    monkeypatch.setattr(larcv_read_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(larcv_read_module, "LARCV_AVAILABLE", True)
    monkeypatch.setattr(larcv_read_module, "larcv", CountingLArCV())
    monkeypatch.setattr(
        larcv_read_module,
        "ROOT",
        type("DummyROOT", (), {"TChain": DummyChain})(),
    )

    with pytest.raises(ValueError, match="Mismatch between the number of entries"):
        LArCVReader(file_keys="dummy.root", tree_keys=["sparse3d", "particle"])


def test_larcv_reader_requires_valid_run_info_key(monkeypatch):
    """Run-map creation should require a run-info key present in tree_keys."""

    class DummyInfo:
        @staticmethod
        def run():
            return 1

        @staticmethod
        def subrun():
            return 2

        @staticmethod
        def event():
            return 3

    class DummyChain:
        def __init__(self, name):
            self.name = name
            self.entries = 0
            self.sparse3d_branch = DummyInfo()

        def AddFile(self, path):
            del path
            self.entries += 1

        def GetEntries(self):
            return self.entries

        def GetEntry(self, entry):
            del entry

    monkeypatch.setattr(
        LArCVReader,
        "process_file_paths",
        lambda self, *args, **kwargs: setattr(self, "file_paths", ["dummy.root"]),
    )
    monkeypatch.setattr(larcv_read_module, "ROOT_AVAILABLE", True)
    monkeypatch.setattr(larcv_read_module, "LARCV_AVAILABLE", True)
    monkeypatch.setattr(larcv_read_module, "larcv", CountingLArCV())
    monkeypatch.setattr(
        larcv_read_module,
        "ROOT",
        type("DummyROOT", (), {"TChain": DummyChain})(),
    )

    with pytest.raises(ValueError, match="run_info_key"):
        LArCVReader(
            file_keys="dummy.root",
            tree_keys=["sparse3d"],
            create_run_map=True,
            run_info_key="particle",
        )
