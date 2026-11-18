"""Exercise allocation pauses, pending validation and checkpoint publication.

The real Driver loop/checkpoint methods are exercised with CPU-only model and
loader doubles. GPU/DDP restart equivalence remains a site integration check.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def driver(tmp_path, monkeypatch, *, start=0, limit=10, scheduled=False, early=False):
    """Build a CPU-only driver with controllable pause and validation behavior."""
    from spine.control import RunControl
    from spine.driver import Driver
    from spine.utils.stopwatch import StopwatchManager

    marker = tmp_path / "pause"
    outcome = tmp_path / "outcome.json"
    monkeypatch.setenv("SPINE_CONTINUATION_TOKEN", "attempt-1")
    monkeypatch.setenv("SPINE_PAUSE_FILE", str(marker))
    monkeypatch.setenv("SPINE_OUTCOME_FILE", str(outcome))
    drv = object.__new__(Driver)
    history = {
        "metrics": {"loss": 0.5},
        "early_stopping": {"bad_checks": 2},
        "best_checkpoint": {"best": 0.5},
    }
    saves = []

    def save(iteration, epoch, validation, **kwargs):
        cp = tmp_path / (
            "snapshot-initial.ckpt" if iteration == -1 else f"snapshot-{iteration}.ckpt"
        )
        cp.write_bytes(f"step-{iteration}".encode())
        Path(str(cp) + ".sha256").write_text(
            f"{hashlib.sha256(cp.read_bytes()).hexdigest()}  {cp.name}\n"
        )
        saves.append((iteration, validation, kwargs))
        return str(cp)

    drv.model = SimpleNamespace(
        train=True,
        start_iteration=start,
        start_epoch=float(start),
        device=None,
        checkpoint_validation=history,
        should_save=lambda _: scheduled,
        step_scheduler=Mock(),
        save_state=save,
        save_best_state=Mock(),
        watch=StopwatchManager(),
    )
    drv.validation = SimpleNamespace(
        initial=False,
        num_iterations=1,
        io=SimpleNamespace(dataset_provenance=lambda: {}),
        run=Mock(return_value={"loss": 0.6}),
        update_best_checkpoint=Mock(return_value=False),
        update_early_stopping=Mock(return_value=early),
        checkpoint_state=Mock(return_value=history),
        close=Mock(),
    )
    drv.iterations = limit
    drv.epochs = None
    drv.run_control = RunControl()
    drv.main_process = True
    drv.distributed = False
    drv.rank = None
    drv.world_size = 1
    drv.cfg = {}
    drv.io = SimpleNamespace(
        has_loader=True,
        iter_per_epoch=1,
        prepare_iteration=Mock(),
        checkpoint_state=lambda next_iteration: {"next_iteration": next_iteration},
        dataset_provenance=lambda: {},
        close=Mock(),
    )
    drv.log_manager = SimpleNamespace(append_tensorboard=Mock(), close=Mock())
    drv.watch = StopwatchManager()
    drv.initialize_log = Mock()
    drv.reduce_training_metrics = Mock()
    drv.log_stdout = Mock()
    drv.log = Mock()
    drv.cleanup = Mock()

    def process(**kwargs):
        marker.touch()
        return {"loss": 1.0}

    drv.process = process
    return drv, saves, outcome, history


def test_pause_preserves_validation_patience_and_runtime(tmp_path, monkeypatch):
    drv, saves, outcome, history = driver(tmp_path, monkeypatch)
    drv.run()
    result = json.loads(outcome.read_text())
    assert result["state"] == "paused"
    assert result["iteration"] == 0
    drv.validation.run.assert_not_called()
    drv.validation.update_early_stopping.assert_not_called()
    drv.model.save_best_state.assert_not_called()
    assert all(
        call.args[0] != "validation" for call in drv.model.step_scheduler.call_args_list
    )
    assert saves[0][1] == history
    assert saves[0][2]["runtime_state"]["ranks"][0]["io"]["next_iteration"] == 1
    assert saves[0][2]["completion"] is None
    drv.cleanup.assert_called_once()


@pytest.mark.parametrize("early", [True, False])
def test_terminal_wins_over_pause(tmp_path, monkeypatch, early):
    drv, saves, outcome, _ = driver(
        tmp_path, monkeypatch, limit=10 if early else 1, scheduled=early, early=early
    )
    drv.run()
    result = json.loads(outcome.read_text())
    assert result["state"] == "completed"
    assert result["reason"] == ("early_stopping" if early else "training_limit")
    assert len(saves) == 1


def test_pause_resume_preserves_absolute_iteration_limit(tmp_path, monkeypatch):
    progress = []
    for start in range(3):
        directory = tmp_path / str(start)
        directory.mkdir()
        drv, saves, outcome, _ = driver(directory, monkeypatch, start=start, limit=3)
        drv.run()
        result = json.loads(outcome.read_text())
        progress.append(result["iteration"])
        assert result["state"] == ("completed" if start == 2 else "paused")
    assert progress == [0, 1, 2]


def test_checkpoint_failure_never_publishes_outcome(tmp_path, monkeypatch):
    drv, _, outcome, _ = driver(tmp_path, monkeypatch)
    drv.model.save_state = Mock(side_effect=OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        drv.run()
    assert not outcome.exists()


def real_validation(drv, marker=None):
    """Use the real validation batch loop with cheap CPU model/loader doubles."""
    from spine.model.validation import ValidationManager

    val = object.__new__(ValidationManager)
    val.distributed = False
    val.main_process = False
    val.num_iterations = 3
    val.initialize_log = Mock(return_value=Mock())

    def load():
        if marker is not None:
            marker.touch()
        return {}

    val.io = SimpleNamespace(
        loader=SimpleNamespace(sampler=None),
        reset_loader=Mock(),
        load=load,
    )
    val.model = SimpleNamespace(evaluate=Mock(return_value={"loss": 0.6}))
    drv.validation.run = Mock(wraps=val.run)
    return val


def test_validation_interruption_preserves_policy_and_resumes_before_terminal_limit(
    tmp_path, monkeypatch
):
    first = tmp_path / "first"
    first.mkdir()
    drv, saves, outcome, history = driver(first, monkeypatch, limit=1, scheduled=True)
    # Request pause during the first validation batch, not during training.
    drv.process = Mock(return_value={"loss": 1.0})
    val = real_validation(drv, first / "pause")
    drv.run()
    result = json.loads(outcome.read_text())
    assert result["state"] == "paused"
    assert result["pending_validation"] is True
    assert val.model.evaluate.call_count == 1
    val.initialize_log.return_value.close.assert_called_once()
    pending_state = saves[0][1]
    assert pending_state["pending_validation"]["iteration"] == 0
    assert pending_state["metrics"] == history["metrics"]
    drv.validation.update_early_stopping.assert_not_called()
    drv.validation.update_best_checkpoint.assert_not_called()
    assert all(
        call.args[0] != "validation" for call in drv.model.step_scheduler.call_args_list
    )

    second = tmp_path / "second"
    second.mkdir()
    resumed, resumed_saves, resumed_outcome, _ = driver(
        second, monkeypatch, start=1, limit=1
    )
    resumed.model.checkpoint_validation = pending_state
    resumed.process = Mock(
        side_effect=AssertionError("No additional training permitted")
    )
    completed_val = real_validation(resumed)
    resumed.run()
    assert completed_val.model.evaluate.call_count == 3
    resumed.process.assert_not_called()
    resumed.validation.update_early_stopping.assert_called_once()
    resumed.validation.update_best_checkpoint.assert_called_once()
    resumed.model.step_scheduler.assert_called_once_with("validation", {"loss": 0.6})
    assert "pending_validation" not in resumed_saves[0][1]
    assert json.loads(resumed_outcome.read_text())["state"] == "completed"


def test_pending_validation_can_early_stop_before_more_training(tmp_path, monkeypatch):
    drv, saves, outcome, history = driver(
        tmp_path, monkeypatch, start=1, limit=10, early=True
    )
    drv.model.checkpoint_validation = {
        **history,
        "pending_validation": {
            "iteration": 0,
            "epoch": 1.0,
            "scheduled": True,
            "reason": "scheduled",
            "completion": None,
        },
    }
    drv.process = Mock(side_effect=AssertionError("Early stop must precede training"))
    real_validation(drv)
    drv.run()
    drv.process.assert_not_called()
    assert json.loads(outcome.read_text())["reason"] == "early_stopping"
    assert "pending_validation" not in saves[0][1]


def test_pause_after_final_validation_batch_uses_completed_checkpoint(
    tmp_path, monkeypatch
):
    drv, saves, outcome, _ = driver(tmp_path, monkeypatch, scheduled=True)
    drv.process = Mock(return_value={"loss": 1.0})
    val = real_validation(drv)

    def evaluate(*_):
        if val.model.evaluate.call_count == 3:
            (tmp_path / "pause").touch()
        return {"loss": 0.6}

    val.model.evaluate.side_effect = evaluate
    drv.run()
    assert drv.process.call_count == 1
    assert len(saves) == 1
    assert "pending_validation" not in saves[0][1]
    drv.validation.update_early_stopping.assert_called_once()
    assert json.loads(outcome.read_text())["state"] == "paused"


def test_interrupted_validation_restores_training_rng(tmp_path, monkeypatch):
    import random

    from spine.model.validation import ValidationInterrupted
    from spine.utils.torch import runtime

    drv, saves, _, _ = driver(tmp_path, monkeypatch, scheduled=True)
    drv.process = Mock(return_value={"loss": 1.0})
    monkeypatch.setattr(runtime, "capture_rng_state", random.getstate)
    monkeypatch.setattr(runtime, "restore_rng_state", random.setstate)
    before = random.getstate()

    def interrupted(*args, **kwargs):
        random.random()
        raise ValidationInterrupted()

    drv.validation.run.side_effect = interrupted
    drv.run()
    assert random.getstate() == before
    assert saves[0][2]["runtime_state"]["ranks"][0]["rng"] == before


def test_rank_zero_pause_is_collective_before_validation_load(tmp_path, monkeypatch):
    from spine.model.validation import ValidationInterrupted
    from spine.utils.torch import runtime

    drv, _, _, _ = driver(tmp_path, monkeypatch)
    # Pretend this is a non-main rank receiving another rank's pause decision.
    drv.main_process = False
    drv.continuation = SimpleNamespace(pause_requested=False)
    monkeypatch.setattr(runtime, "distributed_any", lambda requested, device: True)
    val = real_validation(drv)
    with pytest.raises(ValidationInterrupted):
        val.run(0, pause_requested=drv.continuation_pause_requested)
    val.model.evaluate.assert_not_called()
    val.io.reset_loader.assert_not_called()


def test_initial_validation_can_pause_without_training(tmp_path, monkeypatch):
    drv, saves, outcome, _ = driver(tmp_path, monkeypatch)
    drv.model.checkpoint_validation = None
    drv.validation.initial = True
    drv.process = Mock(side_effect=AssertionError("Initial validation still pending"))
    real_validation(drv, tmp_path / "pause")
    drv.run()
    assert saves[0][0] == -1
    assert saves[0][1]["pending_validation"]["reason"] == "initial"
    assert json.loads(outcome.read_text())["iteration"] == -1
    assert json.loads(outcome.read_text())["state"] == "paused"


def test_pending_validation_archives_partial_log(tmp_path):
    from spine.model.validation import ValidationManager

    val = object.__new__(ValidationManager)
    val.log_dir = str(tmp_path)
    val.distributed = False
    val.prefix_log = False
    val.overwrite_log = False
    val.csv_buffer_size = 1
    val.log_gpu_memory = False
    path = tmp_path / "validation_log-0000001.csv"
    path.write_text("partial")
    logger = val.initialize_log(0, retry=True)
    assert not path.exists()
    assert next(tmp_path.glob("*.interrupted-*")).read_text() == "partial"
    logger.close()


def test_checkpoint_cadence_survives_multiple_short_allocations():
    """Splitting a run must preserve the uninterrupted validation boundaries."""
    from spine.model.manager import ModelManager

    manager = SimpleNamespace(save_step=980, start_iteration=0)
    uninterrupted = [i for i in range(2500) if ModelManager.should_save(manager, i)]
    resumed = []
    for start, stop in ((0, 526), (526, 997), (997, 1649), (1649, 2500)):
        manager.start_iteration = start
        resumed.extend(
            i for i in range(start, stop) if ModelManager.should_save(manager, i)
        )
    assert uninterrupted == resumed == [979, 1959]
    manager.save_step = None
    assert not ModelManager.should_save(manager, 1959)


def test_resumed_driver_validates_at_original_epoch_boundary(tmp_path, monkeypatch):
    """A resumed allocation validates before pausing at its later boundary."""
    from spine.model.manager import ModelManager

    drv, saves, outcome, _ = driver(tmp_path, monkeypatch, start=978, limit=2000)
    drv.model.save_step = 980
    drv.model.should_save = lambda i: ModelManager.should_save(drv.model, i)
    processed = []

    def process(**kwargs):
        processed.append(kwargs)
        if len(processed) == 3:
            (tmp_path / "pause").touch()
        return {"loss": 1.0}

    drv.process = process
    drv.run()
    drv.validation.run.assert_called_once()
    assert [iteration for iteration, _, _ in saves] == [979, 980]
    result = json.loads(outcome.read_text())
    assert result["state"] == "paused"
    assert result["iteration"] == 980
    assert not result["pending_validation"]


@pytest.mark.parametrize("resume", [False, True])
def test_initial_validation_does_not_restart_on_resume(tmp_path, monkeypatch, resume):
    """An explicit baseline applies to fresh runs, never to resumed history."""
    drv, saves, outcome, _ = driver(tmp_path, monkeypatch)
    drv.validation.initial = True
    drv.model.checkpoint_validation = None
    drv.model.resume_training = resume
    drv.run()
    assert drv.validation.run.call_count == (0 if resume else 1)
    assert [iteration for iteration, _, _ in saves] == ([0] if resume else [-1, 0])
    assert json.loads(outcome.read_text())["state"] == "paused"
