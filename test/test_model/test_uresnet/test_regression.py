"""Deterministic regression test for standalone UResNet inference."""

import copy
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

from spine.config import load_config_file
from spine.driver import Driver
from spine.utils.conditional import TORCH_AVAILABLE

from ..cases import INFERENCE_MODEL_CONFIGS

URESNET_REFERENCE_ATOL = 1.0e-5
URESNET_LOGIT_ATOL = 1.0e-3


@dataclass(frozen=True)
class UResNetOutput:
    """Compact representation of one UResNet inference batch."""

    loss: float
    accuracy: float
    shape: tuple[int, int]
    prediction_counts: tuple[int, ...]
    logit_sum: float
    logit_square_sum: float


URESNET_REFERENCE = UResNetOutput(
    loss=1.716552495956421,
    accuracy=0.02284232621311273,
    shape=(8099, 5),
    prediction_counts=(0, 0, 0, 7539, 560),
    logit_sum=87.22920227050781,
    logit_square_sum=754.7786865234375,
)


def make_uresnet_config(larcv_data: str, tmp_path: Path) -> dict:
    """Build the canonical UResNet inference configuration for CI."""
    cfg_path = INFERENCE_MODEL_CONFIGS["uresnet"]
    cfg = load_config_file(str(cfg_path), download=False)
    cfg["base"]["log_dir"] = str(tmp_path)
    cfg["io"]["loader"]["dataset"]["file_keys"] = larcv_data
    return cfg


def run_uresnet(cfg: dict) -> UResNetOutput:
    """Run one two-event UResNet batch and summarize its output."""
    result = Driver(copy.deepcopy(cfg)).process(iteration=0)
    logits = result["segmentation"].torch_tensor()
    predictions = logits.argmax(dim=1)

    return UResNetOutput(
        loss=float(result["loss"]),
        accuracy=float(result["accuracy"]),
        shape=tuple(logits.shape),
        prediction_counts=tuple(
            predictions.bincount(minlength=logits.shape[1]).tolist()
        ),
        logit_sum=float(logits.sum()),
        logit_square_sum=float(logits.square().sum()),
    )


def assert_uresnet_reference(output: UResNetOutput) -> None:
    """Compare UResNet output with the checked-in deterministic reference."""
    assert output.shape == URESNET_REFERENCE.shape
    assert output.prediction_counts == URESNET_REFERENCE.prediction_counts

    for name, tolerance in (
        ("loss", URESNET_REFERENCE_ATOL),
        ("accuracy", URESNET_REFERENCE_ATOL),
        ("logit_sum", URESNET_LOGIT_ATOL),
        ("logit_square_sum", URESNET_LOGIT_ATOL),
    ):
        value = getattr(output, name)
        reference = getattr(URESNET_REFERENCE, name)
        if value == reference:
            continue

        difference = abs(value - reference)
        if difference <= tolerance:
            warnings.warn(
                f"UResNet {name} differs from reference by {difference:.3g}: "
                f"{value} != {reference}",
                RuntimeWarning,
                stacklevel=2,
            )

        assert difference <= tolerance, (
            f"UResNet {name} differs from reference by {difference:.3g}: "
            f"{value} != {reference}"
        )


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_uresnet_larcv_regression(larcv_data: str, tmp_path: Path) -> None:
    """Check a deterministic inference batch containing two LArCV events."""
    cfg = make_uresnet_config(larcv_data, tmp_path)

    assert cfg["base"]["iterations"] == 1
    assert cfg["io"]["loader"]["minibatch_size"] == 2

    output = run_uresnet(cfg)
    repeat_output = run_uresnet(cfg)

    assert output == repeat_output
    assert_uresnet_reference(output)
