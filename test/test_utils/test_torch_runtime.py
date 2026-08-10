"""Tests for optional PyTorch runtime state helpers."""

import random

import numpy as np

from spine.utils.conditional import TORCH_AVAILABLE, torch
from spine.utils.torch import runtime


def test_rng_state_round_trip_restores_python_and_numpy():
    """Runtime state should reproduce process-local stochastic streams."""
    random.seed(13)
    np.random.seed(13)
    state = runtime.capture_rng_state()
    expected = [random.random(), np.random.random()]
    if TORCH_AVAILABLE:
        expected.append(torch.rand(1).item())

    random.seed(99)
    np.random.seed(99)
    runtime.restore_rng_state(state)

    result = [random.random(), np.random.random()]
    if TORCH_AVAILABLE:
        result.append(torch.rand(1).item())
    assert result == expected
