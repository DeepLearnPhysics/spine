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


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_manager_optimizes_trainable_loss_state_without_weight_decay(monkeypatch):
    """Learned objective state should be optimized in a no-decay group."""

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_variance = torch.nn.Parameter(torch.tensor(0.0))

    monkeypatch.setattr(
        "spine.model.manager.model_factory", lambda _name: (Network, Loss)
    )
    manager = ModelManager(
        name="test",
        modules={},
        network_input={"data": "data"},
        loss_input={"target": "target"},
        train={
            "optimizer": {"name": "SGD", "lr": 0.1, "weight_decay": 0.2},
            "gradient_tracking": {},
        },
    )

    assert len(manager.optimizer.param_groups) == 2
    assert manager.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)
    assert manager.optimizer.param_groups[1]["weight_decay"] == 0.0
    assert manager.optimizer.param_groups[1]["params"] == [manager.loss_fn.log_variance]
    assert set(manager.gradient_tracker.groups) == {"global"}


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_manager_tracks_gradients_after_freezing_and_before_updates(monkeypatch):
    """Training should publish canonical metrics for active parameter groups."""

    class Network(torch.nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1)
            self.head = torch.nn.Linear(1, 1)

        def forward(self, data):
            return {"prediction": self.head(self.encoder(data))}

    class Loss(torch.nn.Module):
        def __init__(self, **_modules):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, prediction, target):
            return {"loss": self.scale * (prediction - target).square().mean()}

    monkeypatch.setattr(
        "spine.model.manager.model_factory", lambda _name: (Network, Loss)
    )
    manager = ModelManager(
        name="test",
        modules={
            "encoder": {"freeze_weights": True},
            "head": {},
        },
        network_input={"data": "data"},
        loss_input={"target": "target"},
        train={
            "optimizer": {"name": "SGD", "lr": 0.1},
            "gradient_tracking": {
                "groups": {
                    "head": "network.head.*",
                    "objective": "loss.*",
                }
            },
        },
    )

    tracked_names = list(manager.gradient_tracker.named_parameters)
    assert not any(name.startswith("network.encoder.") for name in tracked_names)
    assert any(name.startswith("network.head.") for name in tracked_names)
    result = manager(
        {"data": torch.ones((1, 1)), "target": torch.zeros((1, 1))},
        iteration=0,
    )

    assert result["gradient_sampled"] == 1
    assert result["gradient_global_norm"] > 0.0
    assert result["gradient_head_missing_fraction"] == 0.0
    assert result["gradient_objective_missing_fraction"] == 0.0


def test_manager_disables_gradient_tracking_by_default():
    """Legacy training should not construct or emit gradient diagnostics."""
    network = torch.nn.Linear(1, 1)
    manager = make_bare_manager(
        net=network,
        optimizer=torch.optim.SGD(network.parameters(), lr=0.1),
    )
    parameter = next(manager.net.parameters())
    metrics = manager.backward(parameter.sum())
    assert metrics == {}


def test_initialize_train_validates_gradient_tracking_type():
    """Manager-level gradient configuration should reject non-mappings."""
    manager = make_bare_manager(net=torch.nn.Linear(1, 1))
    with pytest.raises(TypeError, match="gradient_tracking.*boolean or mapping"):
        manager.initialize_train(
            optimizer={"name": "Adam"},
            gradient_tracking=[],
        )


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
        "watch": FakeWatchManager(),
        "lr_scheduler": None,
        "lr_scheduler_interval": "step",
        "lr_scheduler_monitor": None,
        "scheduler_resume": "restore",
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
    with pytest.raises(TypeError, match="resume.*boolean"):
        manager.initialize_train(optimizer={"name": "Adam"}, resume="yes")
    with pytest.raises(ValueError, match="restore_optimizer"):
        manager.initialize_train(
            optimizer={"name": "Adam"},
            restore_optimizer=True,
            resume=False,
        )
    with pytest.raises(TypeError, match="scheduler_resume.*string"):
        manager.initialize_train(
            optimizer={"name": "Adam"},
            scheduler_resume=False,
        )
    with pytest.raises(ValueError, match="'restore' or 'restart'"):
        manager.initialize_train(
            optimizer={"name": "Adam"},
            scheduler_resume="continue",
        )
    with pytest.raises(ValueError, match="requires `lr_scheduler`"):
        manager.initialize_train(
            optimizer={"name": "Adam"},
            scheduler_resume="restart",
        )

    scheduler = object()
    scheduler_cfg = []
    monkeypatch.setattr("spine.model.manager.optim_factory", lambda *_args: object())
    monkeypatch.setattr(
        "spine.model.manager.lr_sched_factory",
        lambda cfg, _optimizer: scheduler_cfg.append(cfg) or scheduler,
    )
    manager.initialize_train(
        optimizer={"name": "Adam"},
        weight_prefix=str(tmp_path / "weights" / "snapshot"),
        save_epoch=0.5,
        iter_per_epoch=10,
        lr_scheduler={
            "name": "ReduceLROnPlateau",
            "interval": "validation",
            "monitor": "loss",
        },
    )
    assert manager.save_step == 5
    assert ("initialize", "save") in manager.watch.calls
    assert manager.lr_scheduler is scheduler
    assert manager.lr_scheduler_interval == "validation"
    assert manager.lr_scheduler_monitor == "loss"
    assert scheduler_cfg == [{"name": "ReduceLROnPlateau"}]
    assert not manager.resume_training
    assert (tmp_path / "weights").is_dir()


def test_initialize_train_selects_resume_mode_from_weight_path():
    """A single training checkpoint should enable non-strict automatic resume."""
    manager = make_bare_manager(
        net=torch.nn.Linear(1, 1),
        configured_weight_path="snapshot.ckpt",
    )

    manager.initialize_train(optimizer={"name": "Adam"})

    assert ("initialize", "save") not in manager.watch.calls
    assert manager.resume_training
    assert manager.restore_optimizer
    assert not manager.strict_resume
    assert manager.load_training_progress
    assert manager.scheduler_resume == "restore"


def test_initialize_train_requires_checkpoint_for_explicit_resume():
    """Strict resume should require one global checkpoint path."""
    manager = make_bare_manager(
        net=torch.nn.Linear(1, 1),
        configured_weight_path=None,
    )

    with pytest.raises(ValueError, match="requires a global `weight_path`"):
        manager.initialize_train(optimizer={"name": "Adam"}, resume=True)


@pytest.mark.parametrize(
    ("scheduler", "error", "message"),
    [
        ("StepLR", TypeError, "must be a mapping"),
        ({"name": "StepLR", "interval": "batch"}, ValueError, "interval"),
        (
            {"name": "StepLR", "interval": "step", "monitor": "loss"},
            ValueError,
            "interval: validation",
        ),
    ],
)
def test_initialize_train_validates_scheduler_policy(
    monkeypatch, scheduler, error, message
):
    """Manager-owned scheduler trigger options should fail during setup."""
    manager = make_bare_manager(net=torch.nn.Linear(1, 1))
    monkeypatch.setattr("spine.model.manager.optim_factory", lambda *_args: object())

    with pytest.raises(error, match=message):
        manager.initialize_train(
            optimizer={"name": "Adam"},
            lr_scheduler=scheduler,
        )


def test_initialize_train_warns_for_implicit_and_legacy_scheduler_intervals(
    monkeypatch,
):
    """Legacy scheduler cadence spellings should remain usable during migration."""
    manager = make_bare_manager(net=torch.nn.Linear(1, 1))
    monkeypatch.setattr("spine.model.manager.optim_factory", lambda *_args: object())
    monkeypatch.setattr("spine.model.manager.lr_sched_factory", lambda *_args: object())

    with pytest.warns(FutureWarning, match="was not specified"):
        manager.initialize_train(
            optimizer={"name": "Adam"},
            lr_scheduler={"name": "StepLR"},
        )
    assert manager.lr_scheduler_interval == "step"

    with pytest.warns(FutureWarning, match="checkpoint.*deprecated"):
        manager.initialize_train(
            optimizer={"name": "Adam"},
            lr_scheduler={"name": "StepLR", "interval": "checkpoint"},
        )
    assert manager.lr_scheduler_interval == "validation"


def test_training_rejects_weight_list(monkeypatch, tmp_path):
    """Checkpoint collections have no defined training-resume semantics."""

    class Network(torch.nn.Module):
        pass

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (Network, Network),
    )
    weight_list = tmp_path / "weights.txt"
    weight_list.write_text("snapshot.ckpt\n", encoding="utf-8")

    with pytest.raises(ValueError, match="only supported for inference"):
        ModelManager(
            name="test",
            modules={},
            network_input={},
            loss_input={"target": "target"},
            weight_list=str(weight_list),
            train={"optimizer": {"name": "Adam"}},
        )


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


def test_scheduler_steps_only_at_configured_boundary_with_optional_metric():
    """Schedulers should support explicit boundaries and monitored validation."""

    class Scheduler:
        def __init__(self):
            self.values = []

        def step(self, *values):
            self.values.append(values)

    scheduler = Scheduler()
    manager = make_bare_manager(
        lr_scheduler=scheduler,
        lr_scheduler_interval="validation",
        lr_scheduler_monitor=None,
    )
    manager.step_scheduler("validation")
    assert scheduler.values == [()]

    manager.lr_scheduler_monitor = "loss"
    manager.step_scheduler("validation", {"loss": 0.25})
    assert scheduler.values[-1] == (0.25,)
    with pytest.raises(KeyError, match="metric `loss`"):
        manager.step_scheduler("validation", {"accuracy": 1.0})

    manager.lr_scheduler_interval = "step"
    manager.step_scheduler("validation", {"loss": 0.1})
    assert len(scheduler.values) == 2


def test_training_rejects_fully_frozen_network():
    """Training should fail before iteration when no parameter can be updated."""
    manager = make_bare_manager(net=torch.nn.Linear(1, 1), train=True)
    manager.net.requires_grad_(False)

    with pytest.raises(ValueError, match="all model weights are frozen"):
        manager._validate_trainable_parameters()

    manager.train = False
    manager._validate_trainable_parameters()


def test_manager_rejects_fully_frozen_training_configuration(monkeypatch):
    """Configured freezes should be validated during manager construction."""

    class Network(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1)

    class Loss(torch.nn.Module):
        def __init__(self, **_modules):
            super().__init__()

    monkeypatch.setattr(
        "spine.model.manager.model_factory",
        lambda _name: (Network, Loss),
    )

    with pytest.raises(ValueError, match="all model weights are frozen"):
        ModelManager(
            name="test",
            modules={"encoder": {"freeze_weights": True}},
            network_input={"data": "data"},
            loss_input={"target": "target"},
            train={"optimizer": {"name": "Adam"}},
        )


def test_backward_rejects_detached_loss():
    """A detached objective should produce a concise model-level diagnostic."""
    manager = make_bare_manager()

    with pytest.raises(RuntimeError, match="loss does not require gradients"):
        manager.backward(torch.tensor(1.0))

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    manager.gradient_tracker = SimpleNamespace(collect=lambda _iteration: {})
    with pytest.raises(ValueError, match="requires the current training iteration"):
        manager.backward(parameter.square())


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
    manager.backward = lambda _loss, _iteration=None: {}
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


def test_save_state_writes_rich_checkpoint_and_requires_prefix(tmp_path, monkeypatch):
    """Checkpoint serialization should record state, provenance and checksum."""
    net = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    monkeypatch.setattr(
        "spine.model.checkpoint._discover_git_state",
        lambda: ("abc123", False),
    )
    manager = make_bare_manager(
        net=net,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        distributed=False,
        weight_prefix=None,
    )
    with pytest.raises(ValueError, match="weight prefix"):
        manager.save_state(1, 0.5)

    manager.weight_prefix = str(tmp_path / "snapshot")
    checkpoint_path = manager.save_state(
        3,
        1.5,
        {"metrics": {"loss": 0.25}},
        config={"model": {"name": "test"}},
        datasets={"train": {"files": ["train.root"]}},
        runtime_state={"world_size": 1, "ranks": []},
        completion={"reason": "graceful_stop", "signal": "SIGUSR1"},
    )
    checkpoint = torch.load(tmp_path / "snapshot-3.ckpt", weights_only=True)
    assert checkpoint["format_version"] == 2
    assert checkpoint["manifest"]["git_revision"] == "abc123"
    assert "runtime_state" in checkpoint["manifest"]["contents"]
    assert checkpoint["config"]["model"]["name"] == "test"
    assert checkpoint["datasets"]["train"]["files"] == ["train.root"]
    assert checkpoint["global_step"] == 3
    assert checkpoint["global_epoch"] == 1.5
    assert checkpoint["lr_scheduler"] == scheduler.state_dict()
    assert checkpoint["runtime_state"] == {"world_size": 1, "ranks": []}
    assert checkpoint["validation"] == {"metrics": {"loss": 0.25}}
    assert checkpoint["completion"] == {
        "reason": "graceful_stop",
        "signal": "SIGUSR1",
    }
    assert (tmp_path / "snapshot-3.ckpt.sha256").exists()
    assert checkpoint_path == str(tmp_path / "snapshot-3.ckpt")

    best_path = manager.save_best_state(checkpoint_path)
    assert best_path == str(tmp_path / "snapshot-best.ckpt")
    assert (tmp_path / "snapshot-best.ckpt.sha256").exists()


def test_save_and_load_state_preserves_trainable_loss_parameters(tmp_path):
    """Adaptive objective state should round-trip outside inference weights."""

    class Loss(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.log_variance = torch.nn.Parameter(torch.tensor(value))

    net = torch.nn.Linear(1, 1)
    source_loss = Loss(1.5)
    optimizer = torch.optim.SGD(
        [
            {"params": list(net.parameters())},
            {"params": list(source_loss.parameters()), "weight_decay": 0.0},
        ],
        lr=0.1,
    )
    manager = make_bare_manager(
        net=net,
        loss_fn=source_loss,
        optimizer=optimizer,
        distributed=False,
        weight_prefix=str(tmp_path / "adaptive"),
    )
    checkpoint_path = manager.save_state(0, None)
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    assert checkpoint["loss_state_dict"]["log_variance"].item() == pytest.approx(1.5)

    target_loss = Loss(-2.0)
    target = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=torch.nn.Linear(1, 1),
        loss_fn=target_loss,
        distributed=False,
        train=False,
    )
    target.load_weights(checkpoint_path)

    assert target_loss.log_variance.item() == pytest.approx(1.5)


def test_strict_resume_requires_trainable_loss_state(monkeypatch, tmp_path):
    """Legacy checkpoints cannot exactly resume an adaptive objective."""

    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_variance = torch.nn.Parameter(torch.tensor(0.0))

    path = tmp_path / "legacy.ckpt"
    path.touch()
    net = torch.nn.Linear(1, 1)
    checkpoint = {
        "state_dict": net.state_dict(),
        "optimizer": {},
        "global_step": 0,
    }
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=net,
        loss_fn=Loss(),
        train=True,
        restore_optimizer=True,
        resume_training=True,
        strict_resume=True,
        optimizer=SimpleNamespace(load_state_dict=lambda _state: None),
    )
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: checkpoint)

    with pytest.raises(KeyError, match="trainable loss state"):
        manager.load_weights(str(path))


def test_invalid_loss_state_is_rejected_before_network_mutation(tmp_path):
    """Checkpoint validation should remain atomic across model and objective."""

    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_variance = torch.nn.Parameter(torch.tensor(0.0))

    source = torch.nn.Linear(1, 1)
    with torch.no_grad():
        source.weight.fill_(4.0)
        source.bias.fill_(3.0)
    path = tmp_path / "invalid-loss.ckpt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "loss_state_dict": {"wrong_name": torch.tensor(1.0)},
        },
        path,
    )

    target = torch.nn.Linear(1, 1)
    initial = {name: value.clone() for name, value in target.state_dict().items()}
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=target,
        loss_fn=Loss(),
        distributed=False,
        train=False,
    )

    with pytest.raises(ValueError, match="loss state does not match"):
        manager.load_weights(str(path))

    for name, value in target.state_dict().items():
        assert torch.equal(value, initial[name])


def test_automatic_resume_warns_when_adaptive_loss_state_is_missing(tmp_path):
    """Automatic legacy resume should restart, rather than hide, loss state."""

    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_variance = torch.nn.Parameter(torch.tensor(0.0))

    net = torch.nn.Linear(1, 1)
    path = tmp_path / "legacy-auto.ckpt"
    torch.save({"state_dict": net.state_dict()}, path)
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=net,
        loss_fn=Loss(),
        distributed=False,
        train=True,
        restore_optimizer=False,
        resume_training=True,
        strict_resume=False,
    )

    with pytest.warns(RuntimeWarning, match="adaptive loss balancing will restart"):
        manager.load_weights(str(path))


@pytest.mark.parametrize(
    ("loss_state", "configured_loss", "error", "message"),
    [
        (
            {"log_variance": torch.tensor(1.0)},
            False,
            ValueError,
            "configured objective does not",
        ),
        ([], True, TypeError, "must be a mapping"),
        (
            {"log_variance": 1.0},
            True,
            TypeError,
            "is not a tensor",
        ),
        (
            {"log_variance": torch.ones(2)},
            True,
            ValueError,
            "shape",
        ),
        (
            {"log_variance": torch.tensor(1, dtype=torch.int64)},
            True,
            ValueError,
            "dtype",
        ),
    ],
)
def test_checkpoint_rejects_incompatible_loss_state(
    tmp_path,
    loss_state,
    configured_loss,
    error,
    message,
):
    """Loss checkpoint structure must match before any tensor is restored."""

    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_variance = torch.nn.Parameter(torch.tensor(0.0))

    net = torch.nn.Linear(1, 1)
    path = tmp_path / "incompatible-loss.ckpt"
    torch.save(
        {
            "state_dict": net.state_dict(),
            "loss_state_dict": loss_state,
        },
        path,
    )
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=net,
        loss_fn=Loss() if configured_loss else None,
        distributed=False,
        train=not configured_loss,
        strict_resume=not configured_loss,
    )

    with pytest.raises(error, match=message):
        manager.load_weights(str(path))


def test_evaluate_restores_training_state_without_gradients():
    """Validation calls should reuse and restore the live training modules."""

    class Network(torch.nn.Module):
        def forward(self, data):
            return {"prediction": data * 2.0}

    class Loss(torch.nn.Module):
        def forward(self, target, prediction):
            return {"loss": torch.mean((prediction - target) ** 2)}

    manager = make_bare_manager(
        train=True,
        net=Network(),
        loss_fn=Loss(),
        input_dict={"data": "data"},
        loss_dict={"target": "target"},
        time_dependent=False,
        to_numpy=False,
    )
    manager.net.train()
    manager.loss_fn.train()

    result = manager.evaluate(
        {"data": torch.tensor([2.0]), "target": torch.tensor([3.0])}
    )

    assert result["loss"].item() == 1.0
    assert not result["loss"].requires_grad
    assert manager.train
    assert manager.net.training
    assert manager.loss_fn.training

    manager.to_numpy = True
    result = manager.evaluate(
        {"data": torch.tensor([2.0]), "target": torch.tensor([3.0])}
    )
    assert result["loss"] == 1.0


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


def test_manager_configures_ranked_device_anomaly_and_ddp(monkeypatch):
    """Ranked managers use the selected CUDA device and optional runtime hooks."""
    calls = {}

    class Network(torch.nn.Module):
        def to(self, **kwargs):
            calls["to"] = kwargs
            return self

    monkeypatch.setattr(
        "spine.model.manager.model_factory", lambda _name: (Network, None)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        torch.autograd,
        "set_detect_anomaly",
        lambda enabled, check_nan: calls.update(anomaly=(enabled, check_nan)),
    )

    def wrap(module, **kwargs):
        calls["ddp"] = kwargs
        return module

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", wrap)
    manager = ModelManager(
        name="test",
        modules={},
        network_input={},
        rank=1,
        distributed=True,
        detect_anomaly=True,
    )

    assert manager.device == "cuda:2"
    assert calls["to"]["device"] == "cuda:2"
    assert calls["anomaly"] == (True, True)
    assert calls["ddp"]["device_ids"] == [2]


def test_manager_wraps_trainable_loss_state_with_ddp(monkeypatch):
    """Distributed training should synchronize learned objective parameters."""
    wrapped = []

    class Network(torch.nn.Module):
        def to(self, **_kwargs):
            return self

    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_variance = torch.nn.Parameter(torch.tensor(0.0))

        def to(self, **_kwargs):
            return self

    monkeypatch.setattr(
        "spine.model.manager.model_factory", lambda _name: (Network, Loss)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    def wrap(module, **_kwargs):
        wrapped.append(module)
        return module

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", wrap)
    manager = ModelManager(
        name="test",
        modules={},
        network_input={},
        loss_input={"target": "target"},
        distributed=True,
        rank=0,
    )

    assert wrapped == [manager.net, manager.loss_fn]


def test_loss_balancing_configuration_is_loss_only():
    """Balancing policy must not leak into network constructors."""
    modules = {
        "network": {"width": 2},
        "network_loss": {"reduction": "mean"},
        "loss_balancing": {"name": "uncertainty"},
    }

    selected = ModelManager.select_network_modules(modules)

    assert selected == {"network": {"width": 2}}


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


def test_manager_exposes_configured_checkpoint_boundaries():
    """Checkpoint scheduling should be exposed to driver orchestration."""
    manager = make_bare_manager(
        train=True,
        to_numpy=False,
        optimizer=SimpleNamespace(zero_grad=lambda **_kwargs: None),
        watch=FakeWatchManager(),
        save_step=3,
        main_process=True,
    )
    for key in ("forward", "backward", "save"):
        manager.watch.initialize(key)
    manager.forward = lambda *_args: {"loss": torch.tensor(0.0)}
    manager.backward = lambda _loss, _iteration=None: {}
    manager({}, iteration=2, epoch=0.5)

    assert manager.should_save(2)
    assert not manager.should_save(1)

    manager.start_iteration = 10
    assert manager.should_save(12)
    assert not manager.should_save(11)


def test_module_weight_path_must_exist(tmp_path):
    """Nested pretrained-module paths are validated before deserialization."""
    manager = make_bare_manager(
        model_name="test",
        model_cfg={"encoder": {"weight_path": str(tmp_path / "missing.ckpt")}},
        net=torch.nn.Linear(1, 1),
    )
    with pytest.raises(ValueError, match="Weight file not found for module"):
        manager.load_weights(None)


def test_load_weights_supports_legacy_torch_and_restores_optimizer(
    monkeypatch, tmp_path
):
    """Checkpoint loading retries old Torch APIs and restores training state."""
    path = tmp_path / "main.ckpt"
    path.touch()
    net = torch.nn.Linear(1, 1)
    checkpoint = {
        "state_dict": net.state_dict(),
        "optimizer": {"state": "restored"},
        "global_step": 6,
        "validation": {"metrics": {"loss": 0.5}},
    }
    calls = []

    def load(_file, **kwargs):
        calls.append(kwargs)
        if "weights_only" in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return checkpoint

    optimizer = SimpleNamespace(
        load_state_dict=lambda state: calls.append({"optimizer": state})
    )
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=net,
        train=True,
        restore_optimizer=True,
        optimizer=optimizer,
    )
    monkeypatch.setattr(torch, "load", load)
    with pytest.warns(RuntimeWarning) as records:
        manager.load_weights(str(path))

    assert len(calls) == 3
    assert calls[-1] == {"optimizer": checkpoint["optimizer"]}
    assert len(records) == 2
    assert "global epoch" in str(records[0].message)
    assert "RNG" in str(records[1].message)
    assert manager.start_iteration == 7
    assert manager.checkpoint_validation == checkpoint["validation"]


def test_load_weights_restores_complete_available_training_state(monkeypatch, tmp_path):
    """Resume mode should restore scheduler and expose runtime provenance."""
    path = tmp_path / "resume.ckpt"
    path.touch()
    net = torch.nn.Linear(1, 1)
    checkpoint = {
        "format_version": 2,
        "manifest": {"spine_version": "test"},
        "config": {"train": {"resume": True}},
        "datasets": {"train": {"files": ["train.root"]}},
        "state_dict": net.state_dict(),
        "optimizer": {"state": "optimizer"},
        "lr_scheduler": {"state": "scheduler"},
        "runtime_state": {"world_size": 1, "ranks": []},
        "global_step": 6,
        "global_epoch": 3.5,
    }
    calls = []
    optimizer = SimpleNamespace(
        load_state_dict=lambda state: calls.append(("optimizer", state))
    )
    scheduler = SimpleNamespace(
        load_state_dict=lambda state: calls.append(("scheduler", state))
    )
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=net,
        train=True,
        restore_optimizer=True,
        resume_training=True,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: checkpoint)

    manager.load_weights(str(path))

    assert calls == [
        ("optimizer", checkpoint["optimizer"]),
        ("scheduler", checkpoint["lr_scheduler"]),
    ]
    assert manager.start_iteration == 7
    assert manager.start_epoch == 3.5
    assert manager.checkpoint_manifest == checkpoint["manifest"]
    assert manager.checkpoint_config == checkpoint["config"]
    assert manager.checkpoint_datasets == checkpoint["datasets"]
    assert manager.checkpoint_runtime_state == checkpoint["runtime_state"]


def test_resume_can_restart_configured_scheduler_and_preserve_optimizer_state(
    monkeypatch, tmp_path
):
    """Scheduler restart should retain tensors but restore configured group values."""
    path = tmp_path / "resume.ckpt"
    path.touch()
    net = torch.nn.Linear(1, 1)
    checkpoint = {
        "state_dict": net.state_dict(),
        "optimizer": {
            "state": {0: {"momentum_buffer": "saved"}},
            "param_groups": [
                {
                    "params": [0],
                    "lr": 1.0e-5,
                    "initial_lr": 1.0e-3,
                    "momentum": 0.8,
                }
            ],
        },
        "lr_scheduler": {"last_epoch": 100, "eta_min": 1.0e-5},
    }

    class Optimizer:
        def __init__(self):
            self.state = {}
            self.param_groups = []

        def load_state_dict(self, state):
            self.state = dict(state["state"])
            self.param_groups = [dict(group) for group in state["param_groups"]]

    scheduler_loads = []
    optimizer = Optimizer()
    scheduler = SimpleNamespace(
        load_state_dict=lambda state: scheduler_loads.append(state)
    )
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=net,
        train=True,
        restore_optimizer=True,
        resume_training=True,
        scheduler_resume="restart",
        optimizer=optimizer,
        lr_scheduler=scheduler,
        _scheduler_restart_param_groups=[
            {"lr": 2.0e-3, "initial_lr": 2.0e-3, "momentum": 0.9}
        ],
    )
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: checkpoint)

    with pytest.warns(RuntimeWarning):
        manager.load_weights(str(path))

    assert optimizer.state == {0: {"momentum_buffer": "saved"}}
    assert optimizer.param_groups == [
        {
            "params": [0],
            "lr": 2.0e-3,
            "initial_lr": 2.0e-3,
            "momentum": 0.9,
        }
    ]
    assert not scheduler_loads


def test_scheduler_restart_rejects_parameter_group_count_mismatch():
    """Scheduler restart requires configured and restored groups to align."""
    manager = make_bare_manager(
        optimizer=SimpleNamespace(param_groups=[{"lr": 1.0e-3}]),
        _scheduler_restart_param_groups=[{"lr": 2.0e-3}, {"lr": 3.0e-3}],
    )

    with pytest.raises(ValueError, match="parameter-group counts differ"):
        manager._restore_scheduler_initial_param_groups()


@pytest.mark.parametrize("has_scheduler_state", [True, False])
def test_scheduler_restart_uses_fresh_pytorch_schedule(
    monkeypatch, tmp_path, has_scheduler_state
):
    """A real scheduler restart should retain moments and use new parameters."""
    path = tmp_path / "resume.ckpt"
    path.touch()
    source_net = torch.nn.Linear(1, 1)
    source_optimizer = torch.optim.SGD(source_net.parameters(), lr=1.0e-3, momentum=0.8)
    source_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        source_optimizer, T_max=10, eta_min=1.0e-5
    )
    source_net(torch.ones(1, 1)).sum().backward()
    source_optimizer.step()
    source_scheduler.step()
    checkpoint = {
        "state_dict": source_net.state_dict(),
        "optimizer": source_optimizer.state_dict(),
        "global_step": 0,
        "global_epoch": 0.5,
        "runtime_state": {"world_size": 1, "ranks": []},
    }
    if has_scheduler_state:
        checkpoint["lr_scheduler"] = source_scheduler.state_dict()

    target_net = torch.nn.Linear(1, 1)
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=target_net,
        train=True,
    )
    manager.initialize_train(
        optimizer={"name": "SGD", "lr": 2.0e-3, "momentum": 0.9},
        resume=True,
        scheduler_resume="restart",
        lr_scheduler={
            "name": "CosineAnnealingLR",
            "interval": "step",
            "T_max": 20,
            "eta_min": 1.0e-7,
        },
    )
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: checkpoint)

    if has_scheduler_state:
        manager.load_weights(str(path))
    else:
        with pytest.warns(RuntimeWarning, match="scheduler will restart"):
            manager.load_weights(str(path))

    assert manager.optimizer.state
    assert manager.optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-3)
    assert manager.optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)
    assert manager.lr_scheduler.last_epoch == 0
    assert manager.lr_scheduler.T_max == 20
    assert manager.lr_scheduler.eta_min == pytest.approx(1.0e-7)


def test_resume_legacy_checkpoint_reports_missing_training_state(monkeypatch, tmp_path):
    """Resume should reject missing optimizer state and warn for later additions."""
    path = tmp_path / "legacy.ckpt"
    path.touch()
    net = torch.nn.Linear(1, 1)
    checkpoint = {"state_dict": net.state_dict(), "global_step": 2}
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=net,
        train=True,
        restore_optimizer=True,
        resume_training=True,
        optimizer=SimpleNamespace(load_state_dict=lambda _state: None),
        lr_scheduler=SimpleNamespace(load_state_dict=lambda _state: None),
    )
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: checkpoint)

    with pytest.raises(KeyError, match="optimizer state"):
        manager.load_weights(str(path))

    checkpoint["optimizer"] = {}
    with pytest.warns(RuntimeWarning) as records:
        manager.load_weights(str(path))

    assert len(records) == 3
    assert "scheduler" in str(records[0].message)
    assert "global epoch" in str(records[1].message)
    assert "RNG" in str(records[2].message)


def test_automatic_resume_allows_legacy_checkpoint_without_optimizer(
    monkeypatch, tmp_path
):
    """Automatic resume should preserve legacy progress and restart its optimizer."""
    path = tmp_path / "legacy.ckpt"
    path.touch()
    net = torch.nn.Linear(1, 1)
    checkpoint = {"state_dict": net.state_dict(), "global_step": 2}
    optimizer_calls = []
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=net,
        train=True,
        restore_optimizer=True,
        resume_training=True,
        strict_resume=False,
        load_training_progress=True,
        optimizer=SimpleNamespace(
            load_state_dict=lambda state: optimizer_calls.append(state)
        ),
    )
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: checkpoint)

    with pytest.warns(RuntimeWarning) as records:
        manager.load_weights(str(path))

    assert manager.start_iteration == 3
    assert not optimizer_calls
    assert any("optimizer state" in str(record.message) for record in records)


def test_explicit_non_resume_loads_weights_without_progress(monkeypatch, tmp_path):
    """Explicit ``resume: false`` should begin new training from loaded weights."""
    path = tmp_path / "pretrained.ckpt"
    path.touch()
    net = torch.nn.Linear(1, 1)
    checkpoint = {
        "state_dict": net.state_dict(),
        "global_step": 9,
        "validation": {"metrics": {"loss": 0.5}},
        "manifest": {"spine_version": "test"},
    }
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=net,
        train=True,
        restore_optimizer=False,
        resume_training=False,
        load_training_progress=False,
    )
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: checkpoint)

    manager.load_weights(str(path))

    assert manager.start_iteration == 0
    assert manager.checkpoint_validation is None
    assert manager.checkpoint_manifest == checkpoint["manifest"]


def test_load_weights_targets_unwrapped_ddp_network(tmp_path):
    """DDP inference should reload bare checkpoint keys into the wrapped module."""
    path = tmp_path / "ddp.ckpt"
    source = torch.nn.Linear(1, 1)
    torch.save(
        {"state_dict": source.state_dict(), "global_step": 4},
        path,
    )
    target = torch.nn.Linear(1, 1)
    manager = make_bare_manager(
        model_name="test",
        model_cfg={},
        net=SimpleNamespace(module=target),
        distributed=True,
        train=False,
    )

    manager.load_weights(str(path))

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)
    assert manager.start_iteration == 5


def test_load_weights_does_not_hide_unrelated_type_errors(monkeypatch, tmp_path):
    """The compatibility retry catches only unsupported `weights_only` APIs."""
    path = tmp_path / "bad-api.ckpt"
    path.touch()
    manager = make_bare_manager(
        model_name="test",
        model_cfg={"test": {"weight_path": str(path)}},
        net=torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("bad file object")),
    )
    with pytest.raises(TypeError, match="bad file object"):
        manager.load_weights(None)


def test_load_weights_translates_nested_module_names(tmp_path) -> None:
    """Nested checkpoints remap their historical module prefix."""

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1)

    path = tmp_path / "encoder.ckpt"
    net = Network()
    torch.save(
        {
            "state_dict": {
                "old.weight": net.encoder.weight.detach().clone(),
                "old.bias": net.encoder.bias.detach().clone(),
            },
            "global_step": 0,
        },
        path,
    )
    manager = make_bare_manager(
        model_name="test",
        model_cfg={
            "encoder": {
                "weight_path": str(path),
                "model_name": "old",
            }
        },
        net=net,
    )
    manager.load_weights(None)
    assert manager.start_iteration == 0


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("prefixed", [False, True])
def test_scoped_root_and_nested_modules_accept_checkpoint_namespaces(
    tmp_path, nested, prefixed
) -> None:
    """Scoped imports accept standalone and full-chain checkpoint layouts."""

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.graph_spice = torch.nn.Linear(2, 1)

    source = torch.nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(4.0)
    source_state = {
        (f"graph_spice.{name}" if prefixed else name): value
        for name, value in source.state_dict().items()
    }
    path = tmp_path / f"weights-{nested}-{prefixed}.ckpt"
    torch.save({"state_dict": source_state, "global_step": 17}, path)

    net = Network() if nested else torch.nn.Linear(2, 1)
    manager = make_bare_manager(
        model_name="full_chain" if nested else "graph_spice",
        model_cfg={"graph_spice": {"weight_path": str(path)}},
        net=net,
    )
    manager.load_weights(None)

    target = net.graph_spice if nested else net
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)
    assert manager.start_iteration == 0
    assert manager.loaded_weight_sources[0]["model_name"] == (
        "graph_spice" if prefixed else ""
    )


def test_scoped_root_import_does_not_restore_training_state(tmp_path) -> None:
    """A same-named root module assignment remains a weights-only import."""
    path = tmp_path / "graph-spice.ckpt"
    source = torch.nn.Linear(1, 1)
    torch.save(
        {
            "state_dict": source.state_dict(),
            "optimizer": {"state": "must-not-load"},
            "global_step": 12,
        },
        path,
    )
    optimizer_loads = []
    target = torch.nn.Linear(1, 1)
    manager = make_bare_manager(
        model_name="graph_spice",
        model_cfg={"graph_spice": {"weight_path": str(path)}},
        net=target,
        train=True,
        restore_optimizer=True,
        optimizer=SimpleNamespace(
            load_state_dict=lambda state: optimizer_loads.append(state)
        ),
    )

    manager.load_weights(None)

    assert torch.equal(target.weight, source.weight)
    assert manager.start_iteration == 0
    assert not optimizer_loads


def test_scoped_import_rejects_ambiguous_checkpoint_namespaces(tmp_path) -> None:
    """Automatic namespace detection refuses conflicting complete matches."""
    path = tmp_path / "ambiguous.ckpt"
    source = torch.nn.Linear(1, 1)
    state = {}
    for name, value in source.state_dict().items():
        state[name] = value.detach().clone()
        state[f"graph_spice.{name}"] = value.detach().clone() + 1
    torch.save({"state_dict": state}, path)
    target = torch.nn.Linear(1, 1)
    initial = {name: value.clone() for name, value in target.state_dict().items()}
    manager = make_bare_manager(
        model_name="graph_spice",
        model_cfg={"graph_spice": {"weight_path": str(path)}},
        net=target,
    )

    with pytest.raises(ValueError, match="both completely match"):
        manager.load_weights(None)

    for name, value in target.state_dict().items():
        assert torch.equal(value, initial[name])


def test_scoped_import_explicit_namespace_resolves_ambiguity(tmp_path) -> None:
    """An explicit source model name strictly selects its checkpoint prefix."""
    path = tmp_path / "explicit.ckpt"
    source = torch.nn.Linear(1, 1)
    state = {}
    for name, value in source.state_dict().items():
        state[name] = value.detach().clone()
        state[f"preferred.{name}"] = value.detach().clone() + 2
    torch.save({"state_dict": state}, path)
    target = torch.nn.Linear(1, 1)
    manager = make_bare_manager(
        model_name="graph_spice",
        model_cfg={
            "graph_spice": {
                "weight_path": str(path),
                "model_name": "preferred",
            }
        },
        net=target,
    )

    manager.load_weights(None)

    assert torch.equal(target.weight, state["preferred.weight"])
    assert torch.equal(target.bias, state["preferred.bias"])


def test_scoped_import_validates_shapes_before_mutating_network(tmp_path) -> None:
    """A late shape mismatch must not copy earlier valid tensors."""
    path = tmp_path / "bad-shape.ckpt"
    target = torch.nn.Linear(2, 1)
    initial = {name: value.clone() for name, value in target.state_dict().items()}
    torch.save(
        {
            "state_dict": {
                "weight": torch.full_like(target.weight, 9.0),
                "bias": torch.zeros(2),
            }
        },
        path,
    )
    manager = make_bare_manager(
        model_name="graph_spice",
        model_cfg={"graph_spice": {"weight_path": str(path)}},
        net=target,
    )

    with pytest.raises(ValueError, match="has shape"):
        manager.load_weights(None)

    for name, value in target.state_dict().items():
        assert torch.equal(value, initial[name])


def test_scoped_import_rejects_unknown_destination_module(tmp_path) -> None:
    """A scoped assignment must identify a root model or direct child."""
    path = tmp_path / "unknown-module.ckpt"
    torch.save({"state_dict": {}}, path)
    manager = make_bare_manager(
        model_name="full_chain",
        model_cfg={"missing": {"weight_path": str(path)}},
        net=torch.nn.Linear(1, 1),
    )

    with pytest.raises(ValueError, match="Could not find destination module"):
        manager.load_weights(None)


def test_scoped_import_rejects_empty_destination_module(tmp_path) -> None:
    """A scoped destination must own parameters or persistent buffers."""

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.empty = torch.nn.Identity()

    path = tmp_path / "empty-module.ckpt"
    torch.save({"state_dict": {}}, path)
    manager = make_bare_manager(
        model_name="full_chain",
        model_cfg={"empty": {"weight_path": str(path)}},
        net=Network(),
    )

    with pytest.raises(ValueError, match="does not contain any"):
        manager.load_weights(None)


@pytest.mark.parametrize(
    ("bad_weight", "error", "message"),
    [
        (1, TypeError, "is not a tensor"),
        (torch.ones((1, 1), dtype=torch.int64), ValueError, "incompatible dtype"),
    ],
)
def test_scoped_import_rejects_invalid_parameter_values(
    tmp_path, bad_weight, error, message
) -> None:
    """Checkpoint values must be tensors with compatible data types."""
    path = tmp_path / f"bad-value-{message}.ckpt"
    target = torch.nn.Linear(1, 1)
    torch.save(
        {
            "state_dict": {
                "weight": bad_weight,
                "bias": target.bias.detach().clone(),
            }
        },
        path,
    )
    manager = make_bare_manager(
        model_name="graph_spice",
        model_cfg={"graph_spice": {"weight_path": str(path)}},
        net=target,
    )

    with pytest.raises(error, match=message):
        manager.load_weights(None)


def test_multiple_scoped_imports_are_validated_before_any_mutation(tmp_path) -> None:
    """A bad component must not leave earlier component weights installed."""

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(1, 1)
            self.second = torch.nn.Linear(1, 1)

    net = Network()
    initial = {name: value.clone() for name, value in net.state_dict().items()}
    first_path = tmp_path / "first.ckpt"
    second_path = tmp_path / "second.ckpt"
    torch.save(
        {
            "state_dict": {
                name: torch.full_like(value, 7.0)
                for name, value in net.first.state_dict().items()
            }
        },
        first_path,
    )
    torch.save(
        {
            "state_dict": {
                "weight": torch.full_like(net.second.weight, 8.0),
                "bias": torch.zeros(2),
            }
        },
        second_path,
    )
    # The loader pops configuration blocks from the end. This order resolves
    # the valid first component before discovering the malformed second one.
    manager = make_bare_manager(
        model_name="full_chain",
        model_cfg={
            "second": {"weight_path": str(second_path)},
            "first": {"weight_path": str(first_path)},
        },
        net=net,
    )

    with pytest.raises(ValueError, match="has shape"):
        manager.load_weights(None)

    for name, value in net.state_dict().items():
        assert torch.equal(value, initial[name])


def test_load_weights_reports_missing_main_and_nested_parameters(tmp_path) -> None:
    """Incomplete checkpoints identify both direct and remapped missing keys."""
    path = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {}, "global_step": 0}, path)
    net = torch.nn.Sequential(torch.nn.Linear(1, 1))
    manager = make_bare_manager(
        model_name="test",
        model_cfg={"test": {"weight_path": str(path)}},
        net=net,
    )
    with pytest.raises(ValueError, match="all necessary parameters"):
        manager.load_weights(None)

    manager.model_cfg = {"0": {"weight_path": str(path), "model_name": "old"}}
    with pytest.raises(ValueError, match="all necessary parameters"):
        manager.load_weights(None)
