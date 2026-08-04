"""Unit tests for model discovery and manager configuration handling."""

from types import SimpleNamespace

import numpy as np
import pytest

from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.manager import ModelManager
from spine.utils.conditional import TORCH_AVAILABLE, torch


class FakeWatch:
    """Minimal stopwatch payload used by the manager reset test."""

    def __init__(self):
        self.running = False
        self.paused = False


class FakeWatchManager:
    """Minimal stopwatch manager with call recording."""

    def __init__(self):
        self.calls = []
        self.watches = {}

    def initialize(self, key):
        self.calls.append(("initialize", key))
        self.watches.setdefault(key, FakeWatch())

    def start(self, key):
        self.calls.append(("start", key))
        self.watches.setdefault(key, FakeWatch()).running = True

    def stop(self, key):
        self.calls.append(("stop", key))
        self.watches.setdefault(key, FakeWatch()).running = False

    def reset(self):
        self.calls.append(("reset", None))
        for watch in self.watches.values():
            watch.running = False
            watch.paused = False

    def reset_if_active(self):
        for watch in self.watches.values():
            if watch.running or watch.paused:
                self.reset()
                break


def test_model_manager_resets_stale_watch_before_call():
    """ModelManager clears stale stopwatch state before forwarding."""
    manager = object.__new__(ModelManager)
    manager.train = False
    manager.to_numpy = False
    manager.watch = FakeWatchManager()
    manager.watch.initialize("forward")
    manager.watch.start("forward")
    manager.forward = lambda data, iteration: {"value": data["index"] + iteration}

    result = manager({"index": 2}, iteration=3)

    assert result == {"value": 5}
    assert manager.watch.calls[:4] == [
        ("initialize", "forward"),
        ("start", "forward"),
        ("reset", None),
        ("start", "forward"),
    ]
    assert manager.watch.calls[-1] == ("stop", "forward")


def test_clean_config_returns_sanitized_copy():
    """Manager-only weight settings must not leak into model constructors."""

    modules = {
        "backbone": {
            "depth": 5,
            "weight_path": "weights.ckpt",
            "freeze_weights": True,
            "nested": [{"model_name": "old_name", "width": 32}],
        }
    }

    cleaned = ModelManager.clean_config(modules)

    assert cleaned == {"backbone": {"depth": 5, "nested": [{"width": 32}]}}
    assert modules["backbone"]["weight_path"] == "weights.ckpt"
    assert modules["backbone"]["nested"][0]["model_name"] == "old_name"


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_inference_manager_does_not_construct_loss(monkeypatch):
    """Pure inference should not require loss configuration or construction."""

    loss_calls = []

    class TestNetwork(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.linear = torch.nn.Linear(network["width"], 1)

        def forward(self, data):
            return {"prediction": self.linear(data)}

    class TestLoss(torch.nn.Module):
        def __init__(self, **modules):
            super().__init__()
            loss_calls.append(modules)

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda name: (TestNetwork, TestLoss),
    )

    modules = {
        "network": {"width": 2},
        "network_loss": {"reduction": "mean"},
    }
    manager = ModelManager(
        name="test",
        modules=modules,
        network_input={"data": "data"},
    )

    assert manager.loss_fn is None
    assert loss_calls == []
    assert modules == {
        "network": {"width": 2},
        "network_loss": {"reduction": "mean"},
    }


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_manager_partitions_network_and_loss_configuration(monkeypatch):
    """Networks receive model blocks while losses receive the complete config."""

    network_calls = []
    loss_calls = []

    class TestNetwork(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            network_calls.append(network)

    class TestLoss(torch.nn.Module):
        def __init__(self, network, network_loss):
            super().__init__()
            loss_calls.append((network, network_loss))

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda name: (TestNetwork, TestLoss),
    )

    modules = {
        "network": {"width": 2},
        "network_loss": {"reduction": "mean"},
    }
    ModelManager(
        name="test",
        modules=modules,
        network_input={"data": "data"},
        loss_input={"target": "target"},
    )

    assert network_calls == [{"width": 2}]
    assert loss_calls == [({"width": 2}, {"reduction": "mean"})]


def make_bare_manager(**attributes):
    """Construct a manager shell for testing independent lifecycle methods."""
    manager = object.__new__(ModelManager)
    defaults = {
        "train": False,
        "device": "cpu",
        "dtype": torch.float32,
        "input_dict": {},
        "loss_dict": None,
        "time_dependent": False,
    }
    defaults.update(attributes)
    for name, value in defaults.items():
        setattr(manager, name, value)
    return manager


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"modules": []}, TypeError, "modules.*mapping"),
        ({"network_input": []}, TypeError, "network_input.*map"),
        ({"loss_input": []}, TypeError, "loss_input.*map"),
        (
            {"train": {"optimizer": {"name": "Adam"}}},
            ValueError,
            "Training requires",
        ),
        ({"dtype": "not_a_dtype"}, ValueError, "Unknown PyTorch dtype"),
    ],
)
def test_manager_validates_top_level_configuration(monkeypatch, kwargs, error, message):
    """Manager-owned configuration contracts fail before model construction."""

    class Network(torch.nn.Module):
        pass

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (Network, None),
    )
    config = {"name": "test", "modules": {}, "network_input": {}}
    config.update(kwargs)
    with pytest.raises(error, match=message):
        ModelManager(**config)


def test_manager_requires_a_loss_implementation(monkeypatch):
    """Supplying loss inputs to a lossless model is rejected explicitly."""

    class Network(torch.nn.Module):
        pass

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (Network, None),
    )
    with pytest.raises(ValueError, match="does not define a loss"):
        ModelManager(
            name="test",
            modules={},
            network_input={},
            loss_input={"target": "target"},
        )


def test_initialize_train_validates_save_cadence(monkeypatch, tmp_path):
    """Step and epoch saving are exclusive and epoch saving needs dataset size."""
    manager = make_bare_manager(net=torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="both `save_step` and `save_epoch`"):
        manager.initialize_train(
            optimizer={"name": "Adam"},
            save_step=1,
            save_epoch=1.0,
        )
    with pytest.raises(ValueError, match="requires `iter_per_epoch`"):
        manager.initialize_train(
            optimizer={"name": "Adam"},
            save_epoch=1.0,
        )

    scheduler = object()
    monkeypatch.setattr("spine.model.manager.optim_factory", lambda *_args: object())
    monkeypatch.setattr(
        "spine.model.manager.lr_sched_factory",
        lambda *_args: scheduler,
    )
    manager.initialize_train(
        optimizer={"name": "Adam"},
        weight_prefix=str(tmp_path / "weights" / "snapshot"),
        save_epoch=0.5,
        iter_per_epoch=10,
        lr_scheduler={"name": "StepLR"},
    )
    assert manager.save_step == 5
    assert manager.lr_scheduler is scheduler
    assert (tmp_path / "weights").is_dir()


def test_prepare_data_converts_batches_and_validates_required_keys():
    """Network and loss inputs are independently mapped and moved to Torch."""
    manager = make_bare_manager(
        input_dict={"x": "data", "context": "context"},
        loss_dict={"target": "label"},
    )
    data = TensorBatch(np.ones((2, 1), dtype=np.float32), counts=[2])
    label = TensorBatch(np.zeros(2, dtype=np.float32), counts=[2])

    network, loss = manager.prepare_data(
        {"data": data, "context": "kept", "label": label}
    )
    assert not network["x"].is_numpy
    assert network["context"] == "kept"
    assert not loss["target"].is_numpy

    with pytest.raises(ValueError, match="provide `data`"):
        manager.prepare_data({"context": "kept", "label": label})
    with pytest.raises(ValueError, match="provide `label`"):
        manager.prepare_data({"data": data, "context": "kept"})


def test_forward_routes_static_and_time_dependent_losses():
    """The loss receives model products and optionally the current iteration."""

    class Network:
        def __call__(self, **inputs):
            return {"prediction": inputs["x"] * 2}

    class Loss:
        def __init__(self):
            self.iteration = "unset"

        def __call__(self, prediction, target, iteration="unset"):
            self.iteration = iteration
            return {"loss": (prediction - target).square().mean()}

    loss = Loss()
    manager = make_bare_manager(
        net=Network(),
        loss_fn=loss,
        input_dict={"x": "x"},
        loss_dict={"target": "target"},
    )
    result = manager.forward({"x": torch.tensor([2.0]), "target": torch.tensor([3.0])})
    assert result["loss"].item() == 1.0
    assert loss.iteration == "unset"

    manager.time_dependent = True
    manager.forward(
        {"x": torch.tensor([2.0]), "target": torch.tensor([3.0])},
        iteration=7,
    )
    assert loss.iteration == 7


def test_backward_steps_optimizer_scheduler_and_model_buffers():
    """Training updates gradients, the scheduler, and model-owned buffers."""

    class Network(torch.nn.Linear):
        def __init__(self):
            super().__init__(1, 1)
            self.buffer_updates = 0

        def update_buffers(self):
            self.buffer_updates += 1

    class Counter:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    net = Network()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler = Counter()
    manager = make_bare_manager(
        net=net,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    manager.backward(net(torch.ones((1, 1))).sum())

    assert scheduler.steps == 1
    assert net.buffer_updates == 1


def test_call_validates_training_outputs_and_iteration():
    """Train calls require a loss and iteration before checkpoint scheduling."""
    manager = make_bare_manager(
        train=True,
        to_numpy=False,
        optimizer=SimpleNamespace(zero_grad=lambda **_kwargs: None),
        watch=FakeWatchManager(),
        save_step=None,
        main_process=True,
    )
    for key in ("forward", "backward", "save"):
        manager.watch.initialize(key)
    manager.forward = lambda *_args: {}
    with pytest.raises(RuntimeError, match="must return a `loss`"):
        manager({}, iteration=0)

    manager.forward = lambda *_args: {"loss": torch.tensor(0.0)}
    manager.backward = lambda _loss: None
    with pytest.raises(ValueError, match="provide iteration"):
        manager({})


def test_cast_to_numpy_handles_supported_products_and_rejects_unknowns():
    """All public model result categories use stable NumPy representations."""
    manager = make_bare_manager()
    tensor = TensorBatch(torch.ones((2, 1)), counts=[2])
    indexes = IndexBatch(torch.tensor([0, 1]), spans=[2], counts=[2])

    class Convertible:
        def to_tensor_batch(self):
            return tensor

    result = {
        "number": 1.5,
        "scalar": torch.tensor(2.0),
        "tensor": tensor,
        "indexes": indexes,
        "convertible": Convertible(),
        "tensor_list": [tensor, Convertible()],
    }
    manager.cast_to_numpy(result)
    assert result["number"] == 1.5
    assert result["scalar"] == 2.0
    assert result["tensor"].is_numpy
    assert result["indexes"].is_numpy
    assert result["convertible"].is_numpy
    assert all(value.is_numpy for value in result["tensor_list"])

    with pytest.raises(ValueError, match="Cannot cast output bad"):
        manager.cast_to_numpy({"bad": object()})


def test_cast_to_numpy_handles_structured_cluster_labels():
    """Structured labels retain their particle-aware batch representation."""
    data = TensorBatch(
        torch.tensor([[0, 0, 0, 0, 1, 0]], dtype=torch.float32),
        counts=[1],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    result = {"label": ClusterLabelBatch(data)}
    make_bare_manager().cast_to_numpy(result)
    assert result["label"].is_numpy


def test_save_state_writes_checkpoint_and_requires_prefix(tmp_path):
    """Checkpoint serialization records model, optimizer, step, and epoch."""
    net = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    manager = make_bare_manager(
        net=net,
        optimizer=optimizer,
        distributed=False,
        weight_prefix=None,
    )
    with pytest.raises(ValueError, match="weight prefix"):
        manager.save_state(1, 0.5)

    manager.weight_prefix = str(tmp_path / "snapshot")
    manager.save_state(3, 1.5)
    checkpoint = torch.load(tmp_path / "snapshot-3.ckpt", weights_only=True)
    assert checkpoint["global_step"] == 3
    assert checkpoint["global_epoch"] == 1.5


def test_manager_reports_missing_torch(monkeypatch):
    """Manager construction fails immediately when PyTorch is unavailable."""
    monkeypatch.setattr("spine.model.manager.TORCH_AVAILABLE", False)
    with pytest.raises(ImportError, match="PyTorch is required"):
        ModelManager(name="test", modules={}, network_input={})


def test_manager_wraps_network_and_loss_construction_errors(monkeypatch):
    """Constructor failures identify whether the network or objective failed."""

    class BadNetwork(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            raise ValueError("bad network")

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (BadNetwork, None),
    )
    with pytest.raises(ValueError, match="Failed to instantiate"):
        ModelManager(name="test", modules={}, network_input={})

    class Network(torch.nn.Module):
        pass

    class BadLoss(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            raise ValueError("bad loss")

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (Network, BadLoss),
    )
    with pytest.raises(ValueError, match="Failed to instantiate"):
        ModelManager(
            name="test",
            modules={},
            network_input={},
            loss_input={"target": "target"},
        )


def test_manager_weight_path_selection_and_validation(monkeypatch, tmp_path):
    """Global paths, lists, and wildcard ensembles are mutually consistent."""

    class Network(torch.nn.Module):
        pass

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (Network, None),
    )
    weight_list = tmp_path / "weights.txt"
    weight_list.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="both `weight_path` and `weight_list`"):
        ModelManager(
            name="test",
            modules={},
            network_input={},
            weight_path="missing.ckpt",
            weight_list=str(weight_list),
        )
    with pytest.raises(ValueError, match="Weight file not found"):
        ModelManager(
            name="test",
            modules={},
            network_input={},
            weight_path=str(tmp_path / "missing*.ckpt"),
        )
    with pytest.raises(ValueError, match="No weight paths"):
        ModelManager(
            name="test",
            modules={},
            network_input={},
            weight_list=str(weight_list),
        )

    first = tmp_path / "first.ckpt"
    second = tmp_path / "second.ckpt"
    first.touch()
    second.touch()
    manager = ModelManager(
        name="test",
        modules={},
        network_input={},
        weight_path=str(tmp_path / "*.ckpt"),
    )
    assert sorted(manager.weight_path) == sorted([str(first), str(second)])


def test_freeze_weights_handles_nested_modules_and_missing_parameters():
    """Configured submodules freeze matching parameters or fail explicitly."""

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.empty = torch.nn.Identity()

    net = Network()
    manager = make_bare_manager(
        net=net,
        model_cfg={"encoder": {"freeze_weights": True}},
    )
    manager.freeze_weights()
    assert not any(parameter.requires_grad for parameter in net.encoder.parameters())
    assert not net.encoder.training

    manager.model_cfg = {"missing": {"freeze_weights": True}}
    with pytest.raises(AttributeError):
        manager.freeze_weights()

    manager.model_cfg = {"empty": {"freeze_weights": True}}
    with pytest.raises(ValueError, match="Could not find any weights"):
        manager.freeze_weights()


def test_manager_saves_on_configured_step():
    """Training calls invoke checkpointing at the configured main-process cadence."""
    manager = make_bare_manager(
        train=True,
        to_numpy=False,
        optimizer=SimpleNamespace(zero_grad=lambda **_kwargs: None),
        watch=FakeWatchManager(),
        save_step=1,
        main_process=True,
    )
    for key in ("forward", "backward", "save"):
        manager.watch.initialize(key)
    manager.forward = lambda *_args: {"loss": torch.tensor(0.0)}
    manager.backward = lambda _loss: None
    saved = []
    manager.save_state = lambda iteration, epoch: saved.append((iteration, epoch))

    manager({}, iteration=2, epoch=0.5)

    assert saved == [(2, 0.5)]


def test_module_weight_path_must_exist(tmp_path):
    """Nested pretrained-module paths are validated before deserialization."""
    manager = make_bare_manager(
        model_name="test",
        model_cfg={"encoder": {"weight_path": str(tmp_path / "missing.ckpt")}},
        net=torch.nn.Linear(1, 1),
    )
    with pytest.raises(ValueError, match="Weight file not found for module"):
        manager.load_weights(None)
