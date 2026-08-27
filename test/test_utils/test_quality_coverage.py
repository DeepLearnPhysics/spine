"""Coverage of shared utility state machines and optional-runtime adapters."""

from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from spine.data import Flash, TensorBatch
from spine.utils import conditional
from spine.utils.docstring import merge_ancestor_docstrings
from spine.utils.ghost import ChargeRescaler
from spine.utils.jit import numbafy
from spine.utils.manager import ModuleManager
from spine.utils.optical import FlashMerger
from spine.utils.stopwatch import Stopwatch, StopwatchManager, Time
from spine.utils.torch import devices, runtime


def test_optional_dependency_proxy_states(monkeypatch):
    """Lazy proxies should cover successful, missing, and annotation-only use."""
    missing = conditional._MissingNamespace("missing")
    assert missing.lowercase is missing
    assert missing.Tensor is conditional._MissingType
    with pytest.raises(ImportError, match="missing"):
        missing()

    module = ModuleType("available_test_module")
    module.value = 7
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    lazy = conditional._LazyModule(module.__name__, "available", "unavailable")
    assert "lazy module" in repr(lazy)
    assert lazy.value == 7
    assert repr(lazy) == repr(module)

    attribute = conditional._LazyAttribute(
        module.__name__, "callable", "available.callable", "unavailable"
    )
    module.callable = lambda value=3: SimpleNamespace(value=value)
    assert "lazy attribute" in repr(attribute)
    assert attribute(4).value == 4
    assert attribute.__name__ == "<lambda>"
    assert repr(attribute) == repr(module.callable)

    missing_module = conditional._LazyModule("not_a_real_spine_module", "x", "needed")
    with pytest.raises(ImportError, match="needed"):
        missing_module._load()
    missing_attribute = conditional._LazyAttribute(
        "not_a_real_spine_module", "x", "x", "attribute needed"
    )
    with pytest.raises(ImportError, match="attribute needed"):
        missing_attribute._load()

    missing_torch = conditional._MissingTorch("torch", "torch", "torch needed")
    assert missing_torch.Tensor is conditional._MissingType
    assert missing_torch.CustomTensor is conditional._MissingType
    assert isinstance(missing_torch.utils, conditional._MissingNamespace)


def test_optional_dependency_availability_branches(monkeypatch):
    """Availability detection must handle docs, preloads, and bad specs."""
    monkeypatch.setenv("SPINE_DOC_BUILD", "1")
    assert not conditional._module_available("torch")
    monkeypatch.delenv("SPINE_DOC_BUILD")

    modules = __import__("sys").modules
    monkeypatch.setitem(modules, "coverage_none", None)
    assert not conditional._module_available("coverage_none")
    monkeypatch.setitem(modules, "ROOT", ModuleType("ROOT"))
    assert conditional._module_available("ROOT")
    monkeypatch.setitem(modules, "larcv", SimpleNamespace(larcv=object()))
    assert conditional._module_available("larcv")
    monkeypatch.setitem(modules, "mocked_optional", ModuleType("mocked_optional"))
    modules["mocked_optional"].__spec__ = object()
    assert conditional._module_available("mocked_optional")

    monkeypatch.delitem(modules, "definitely_absent", raising=False)
    monkeypatch.setattr(conditional.importlib.util, "find_spec", lambda _name: None)
    assert not conditional._module_available("definitely_absent")


def test_numbafy_list_return_and_cuda_rng_restore(monkeypatch):
    """Torch adapters should convert list returns and restore CUDA RNG state."""
    import torch

    @numbafy(cast_args=["values"], keep_torch=True, ref_arg="values")
    def split(values):
        return [values, values + 1]

    result = split(torch.tensor([1.0]))
    assert all(isinstance(value, torch.Tensor) for value in result)

    state = runtime.capture_rng_state()
    called = []
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "set_rng_state", called.append)
    state["cuda"] = torch.tensor([1], dtype=torch.uint8)
    runtime.restore_rng_state(state)
    assert len(called) == 1

    def bad_spec(_name):
        raise ValueError("bad spec")

    monkeypatch.setattr(conditional.importlib.util, "find_spec", bad_spec)
    assert not conditional._module_available("definitely_absent")


def test_docstring_merging_covers_all_structures():
    """Attribute inheritance should ignore malformed parents and merge valid ones."""

    class Empty:
        __doc__ = None

    merge_ancestor_docstrings(Empty)

    class NoAttributes:
        """No structured attributes."""

    merge_ancestor_docstrings(NoAttributes)

    class Parent:
        pass

    Parent.__doc__ = """Parent.

Attributes
----------
parent : int
    Parent value.

Notes
-----
Not part of the attribute section.
"""

    class Child(Empty, NoAttributes, Parent):
        """Child without an attribute section."""

    merge_ancestor_docstrings(Child)
    assert "parent : int" in Child.__doc__
    assert "Not part" not in Child.__doc__

    class ChildWithAttributes(Parent):
        pass

    ChildWithAttributes.__doc__ = """Child.

Attributes
----------
child : str
    Child value.
"""

    merge_ancestor_docstrings(ChildWithAttributes)
    assert ChildWithAttributes.__doc__.index(
        "parent : int"
    ) < ChildWithAttributes.__doc__.index("child : str")


def test_stopwatch_and_manager_state_machine(monkeypatch):
    """Stopwatches should reject invalid transitions and aggregate valid ones."""
    assert Time(3.0, 4.0) + Time(1.0, 2.0) == Time(4.0, 6.0)
    assert Time(3.0, 4.0) - Time(1.0, 2.0) == Time(2.0, 2.0)
    assert Time(3.0, 3.0) == 3.0
    assert Time(3.0, 4.0).copy() == Time(3.0, 4.0)
    monkeypatch.setattr("spine.utils.stopwatch.time.time", lambda: 8.0)
    monkeypatch.setattr("spine.utils.stopwatch.time.process_time", lambda: 2.0)
    assert Time.current() == Time(8.0, 2.0)

    watch = Stopwatch()
    with pytest.raises(ValueError, match="not been started"):
        watch.stop = Time(1.0, 1.0)
    with pytest.raises(ValueError, match="not been started"):
        watch.pause = Time(1.0, 1.0)
    assert np.isnan(watch.time.wall) and np.isnan(watch.time.cpu)
    assert np.isnan(watch.time_sum.wall) and np.isnan(watch.time_sum.cpu)

    watch.start = Time(1.0, 1.0)
    assert watch.running and not watch.paused
    with pytest.raises(ValueError, match="not been stopped"):
        _ = watch.time
    with pytest.raises(ValueError, match="not been stopped"):
        _ = watch.time_sum
    with pytest.raises(ValueError, match="Cannot restart"):
        watch.start = Time(2.0, 2.0)
    watch.pause = Time(3.0, 4.0)
    assert watch.paused and not watch.running
    with pytest.raises(ValueError, match="not been stopped"):
        _ = watch.time
    with pytest.raises(ValueError, match="not been stopped"):
        _ = watch.time_sum
    watch.start = Time(5.0, 6.0)
    watch.stop = Time(8.0, 10.0)
    assert watch.time == Time(5.0, 7.0)
    assert watch.time_sum == Time(5.0, 7.0)
    with pytest.raises(ValueError, match="more than once"):
        watch.stop = Time(9.0, 9.0)
    with pytest.raises(ValueError, match="has been stopped"):
        watch.pause = Time(9.0, 9.0)

    manager = StopwatchManager()
    manager.initialize(["one", "two"])
    assert set(manager.keys()) == {"one", "two"}
    assert len(list(manager.values())) == 2
    assert len(list(manager.items())) == 2
    for operation in (
        manager.start,
        manager.stop,
        manager.pause,
        manager.time,
        manager.time_sum,
    ):
        with pytest.raises(KeyError):
            operation("missing")
    with pytest.raises(KeyError):
        manager.reset("missing")

    manager.start(["one", "two"])
    manager.pause("one")
    manager.reset_if_active()
    assert not any(item.running or item.paused for item in manager.values())
    manager.start(["one", "two"])
    manager.stop(["one", "two"])
    assert manager.time("one") == manager.time_sum("one")
    assert set(manager.times()) == {"one", "two"}
    assert set(manager.times_sum()) == {"one", "two"}
    manager.reset()

    other = StopwatchManager()
    other.initialize("watch")
    manager.update(other)
    manager.update(other, prefix="nested")
    assert {"watch", "nested_watch"} <= set(manager.keys())


def test_module_manager_scalar_batch_and_validation():
    """Shared managers should merge scalar/batch products and validate cardinality."""

    class Manager(ModuleManager):
        def __init__(self, module):
            self.modules = {"module": module}
            self.watch = StopwatchManager()
            self.watch.initialize("module")

    scalar = {"index": 3}
    Manager(lambda data, entry=None: {"value": data["index"]})(scalar)
    assert scalar["value"] == 3

    batch = {"index": [4, 5]}
    Manager(lambda data, entry=None: {"value": data["index"][entry]})(batch)
    assert batch["value"] == [4, 5]

    def partial(_data, entry=None):
        return None if entry else {"value": entry}

    with pytest.raises(ValueError, match="returned 1 values"):
        Manager(partial)({"index": [0, 1]})


def test_charge_rescaler_numpy_torch_and_collection_fallback():
    """Charge sharing should agree for NumPy/Torch and handle missing collection hits."""
    rows = np.array(
        [
            [6.0, 9.0, 12.0, 0, 1, 2],
            [6.0, 9.0, 12.0, 0, 3, -1],
        ],
        dtype=np.float32,
    )
    expected = ChargeRescaler().process_single(rows)
    assert np.allclose(expected, [8.0, 6.0])
    collection = ChargeRescaler(collection_only=True).process_single(rows.copy())
    assert np.allclose(collection, [12.0, 6.0])

    from spine.utils.conditional import torch

    tensor_rows = torch.tensor(rows)
    assert np.allclose(
        ChargeRescaler().process_single(tensor_rows).cpu().numpy(), expected
    )
    batch = TensorBatch(np.vstack((rows, rows)), counts=[2, 2])
    assert ChargeRescaler()(batch).shape == (4,)


def test_flash_merger_all_dispatch_paths():
    """Flash merging should respect time windows and optional volume partitioning."""
    assert FlashMerger()([])[1].shape == (0, 1)
    with pytest.raises(AssertionError, match="two numbers"):
        FlashMerger(window=[0.0])

    flashes = [
        Flash(id=5, volume_id=0, time=0.0, time_width=0.1, total_pe=1.0),
        Flash(id=6, volume_id=1, time=0.5, time_width=0.1, total_pe=2.0),
        Flash(id=7, volume_id=0, time=3.0, time_width=0.1, total_pe=3.0),
    ]
    merged, indexes = FlashMerger(threshold=1.0)(flashes)
    assert len(merged) == 2 and indexes == [[0, 1], [2]]
    assert merged[0].volume_id == 0 and merged[1].id == 1

    separate, indexes = FlashMerger(threshold=1.0, combine_volumes=False)(flashes)
    assert len(separate) == 3
    assert sorted(int(index[0]) for index in indexes) == [0, 1, 2]

    windowed, _ = FlashMerger(threshold=1.0, window=[1.0, 2.0])(flashes[:2])
    assert len(windowed) == 2


def test_cuda_device_selection_with_mocked_runtime(monkeypatch):
    """GPU selection should validate optional runtime and visible-device counts."""
    monkeypatch.setattr(devices, "TORCH_AVAILABLE", False)
    assert devices.set_visible_devices() == 0
    with pytest.raises(ImportError, match="PyTorch"):
        devices.set_visible_devices(gpus=[0])

    monkeypatch.setattr(devices, "TORCH_AVAILABLE", True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(devices.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="does not match"):
        devices.set_visible_devices(gpus=[0], world_size=2)
    assert devices.set_visible_devices(gpus=[1]) == 1
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "1"
    with pytest.raises(AssertionError, match="exceeds"):
        devices.set_visible_devices(world_size=3)
    monkeypatch.setenv("RANK", "0")
    assert devices.set_visible_devices(world_size=3) == 3


def test_runtime_helpers_with_mocked_optional_states(monkeypatch):
    """Runtime adapters should exercise CPU, CUDA, distributed, and writer paths."""
    monkeypatch.setattr(runtime, "TORCH_AVAILABLE", False)
    assert not runtime.cuda_is_available()
    assert runtime.cuda_mem_info() == (0, 0)
    assert runtime.cuda_max_memory_allocated() == 0
    assert runtime.distributed_all_gather_object("x") == ["x"]
    with pytest.raises(ImportError, match="PyTorch"):
        runtime.require_torch("testing")

    monkeypatch.setattr(runtime, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "mem_get_info", lambda: (2, 8))
    monkeypatch.setattr(runtime.torch.cuda, "max_memory_allocated", lambda: 4)
    assert runtime.cuda_is_available()
    assert runtime.cuda_mem_info() == (2, 8)
    assert runtime.cuda_max_memory_allocated() == 4

    calls = []
    monkeypatch.setattr(runtime.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        runtime.torch.distributed, "barrier", lambda: calls.append("barrier")
    )
    monkeypatch.setattr(runtime.torch.distributed, "get_world_size", lambda: 2)

    def gather(objects, obj):
        objects[:] = [obj, "remote"]

    monkeypatch.setattr(runtime.torch.distributed, "all_gather_object", gather)
    runtime.distributed_barrier()
    assert calls == ["barrier"]
    assert runtime.distributed_all_gather_object("local") == ["local", "remote"]

    class Writer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        runtime, "import_module", lambda _name: SimpleNamespace(SummaryWriter=Writer)
    )
    assert runtime.create_summary_writer("logs", flush_secs=2).kwargs == {
        "log_dir": "logs",
        "flush_secs": 2,
    }

    def missing_writer(_name):
        raise ModuleNotFoundError("tensorboard")

    monkeypatch.setattr(runtime, "import_module", missing_writer)
    with pytest.raises(ImportError, match="TensorBoard"):
        runtime.create_summary_writer("logs")


def test_numbafy_conversion_and_validation_paths():
    """JIT wrappers should validate named casts and preserve Torch outputs."""
    from spine.utils.conditional import torch

    @numbafy(
        cast_args=["values"], list_args=["groups"], keep_torch=True, ref_arg="values"
    )
    def convert(values, groups=None):
        return values + (0 if groups is None else len(groups))

    result = convert(torch.tensor([1.0]), groups=[1, 2])
    assert torch.is_tensor(result) and result.item() == 3.0
    assert convert(np.array([1.0]), groups=None).item() == 1.0

    @numbafy(cast_args=["missing"])
    def bad_cast(values):
        return values

    with pytest.raises(AssertionError, match="cast_args"):
        bad_cast(np.array([1.0]))

    @numbafy(list_args=["missing"])
    def bad_list(values):
        return values

    with pytest.raises(AssertionError, match="list_args"):
        bad_list(np.array([1.0]))

    @numbafy(cast_args=["values"])
    def invalid_input(values):
        return values

    with pytest.raises(TypeError, match="Can only convert"):
        invalid_input([1.0])

    @numbafy(keep_torch=True, ref_arg="values")
    def invalid_output(values):
        return "invalid"

    with pytest.raises(TypeError, match="Return type not recognized"):
        invalid_output(torch.tensor([1.0]))
