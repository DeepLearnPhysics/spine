"""Tests for the full-chain calibration provider."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spine.data import TensorBatch
from spine.model.full_chain.providers import calibration as calibration_module
from spine.model.full_chain.providers.calibration import (
    CalibrationStage,
    build_calibration_stage,
)
from spine.model.full_chain.state import ChainState


def make_data(counts=(2,)) -> TensorBatch:
    """Build a canonical sparse tensor with one coordinate group and value."""
    batch_ids = np.repeat(np.arange(len(counts)), counts)
    rows = torch.zeros((sum(counts), 5), dtype=torch.float32)
    rows[:, 0] = torch.as_tensor(batch_ids)
    rows[:, 1] = torch.arange(sum(counts))
    rows[:, 4] = torch.arange(1, sum(counts) + 1)
    return TensorBatch(
        rows,
        counts=counts,
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )


def test_label_calibration_replaces_values_without_mutating_input() -> None:
    """Truth calibration copies the input and replaces its value feature."""
    data = make_data()
    energy = TensorBatch(torch.tensor([10.0, 20.0]), counts=[2])

    result = CalibrationStage("calibration", "label", None)(
        ChainState(data=data, energy_label=energy)
    )

    adapted = result.products["data"]
    assert adapted is not data
    assert data.values.torch_tensor().tolist() == [1.0, 2.0]
    assert adapted.values.torch_tensor().tolist() == [10.0, 20.0]
    assert result.outputs == {"data_adapt": adapted}


def test_label_calibration_requires_energy_truth() -> None:
    """Label mode fails clearly when its row-aligned target is unavailable."""
    with pytest.raises(ValueError, match="energy_label"):
        CalibrationStage("calibration", "label", None)(ChainState(data=make_data()))


def test_applied_calibration_routes_event_context_and_updates_points() -> None:
    """Applied calibration processes each module and preserves event context."""

    class Calibrator:
        update_points = True

        def __init__(self):
            self.calls = []

        def __call__(self, points, values, sources, run_id, **kwargs):
            self.calls.append((points.copy(), sources.copy(), run_id, kwargs))
            return points + 10.0, values * 2.0

    calibrator = Calibrator()
    data = make_data((1, 1))
    sources = TensorBatch(torch.tensor([[3], [4]]), counts=[1, 1])
    meta = [object()]
    run_info = [SimpleNamespace(run=17)]

    result = CalibrationStage("calibration", "apply", calibrator)(
        ChainState(
            data=data,
            sources=sources,
            meta=meta,
            run_info=run_info,
        )
    )

    adapted = result.products["data"]
    assert adapted.values.torch_tensor().tolist() == [2.0, 4.0]
    assert adapted.coords.torch_tensor()[:, 0].tolist() == [10.0, 11.0]
    assert [call[2] for call in calibrator.calls] == [17, 17]
    assert [call[3]["module_id"] for call in calibrator.calls] == [0, 1]
    assert all(call[3]["meta"] is meta[0] for call in calibrator.calls)


def test_applied_calibration_selects_value_from_multiple_features() -> None:
    """Applied calibration changes charge without touching auxiliary features."""

    class Calibrator:
        update_points = False

        def __init__(self):
            self.values = []

        def __call__(self, points, values, sources, run_id, **kwargs):
            self.values.append(values.copy())
            return points, values * 2.0

    base = make_data()
    auxiliary = torch.tensor([[10.0, 100.0], [20.0, 200.0]])
    data = TensorBatch(
        torch.cat((base.tensor, auxiliary), dim=1),
        counts=base.counts,
        has_batch_col=True,
        coord_cols=base.coord_cols,
    )
    calibrator = Calibrator()

    result = CalibrationStage("calibration", "apply", calibrator)(
        ChainState(data=data, meta=[object()])
    )

    adapted = result.products["data"]
    assert calibrator.values[0].tolist() == [1.0, 2.0]
    assert adapted.feature(0).torch_tensor().tolist() == [2.0, 4.0]
    assert adapted.feature(1).torch_tensor().tolist() == [10.0, 20.0]
    assert adapted.feature(2).torch_tensor().tolist() == [100.0, 200.0]


@pytest.mark.parametrize("meta", [None, []])
def test_applied_calibration_requires_metadata(meta) -> None:
    """Detector calibration cannot run without image metadata."""

    class Calibrator:
        update_points = False

    with pytest.raises(ValueError, match="requires `meta`"):
        CalibrationStage("calibration", "apply", Calibrator())(
            ChainState(data=make_data(), meta=meta)
        )


def test_applied_calibration_validates_manager_and_batch_layout() -> None:
    """Apply mode validates both its manager and metadata repetition."""
    with pytest.raises(RuntimeError, match="not initialized"):
        CalibrationStage("calibration", "apply", None)(
            ChainState(data=make_data(), meta=[object()])
        )

    calibrator = SimpleNamespace(update_points=False)
    with pytest.raises(ValueError, match="evenly cover"):
        CalibrationStage("calibration", "apply", calibrator)(
            ChainState(data=make_data((1, 1, 1)), meta=[object(), object()])
        )


def test_calibration_builder_validates_and_constructs_modes(monkeypatch) -> None:
    """The provider builder validates config types and owns apply managers."""
    with pytest.raises(ValueError, match="string `mode`"):
        build_calibration_stage("calibration", {}, object())
    with pytest.raises(TypeError, match="must be a mapping"):
        build_calibration_stage(
            "calibration", {"mode": "label", "calibration": []}, object()
        )
    with pytest.raises(ValueError, match="must be `apply` or `label`"):
        build_calibration_stage("calibration", {"mode": "bad"}, object())

    sentinel = object()
    monkeypatch.setattr(
        calibration_module,
        "CalibrationManager",
        lambda **config: sentinel if config == {"gain": 2} else None,
    )
    stage = build_calibration_stage(
        "calibration",
        {"mode": "apply", "calibration": {"gain": 2}},
        object(),
    )
    assert stage.calibrator is sentinel
