"""Tests for optional PyTorch runtime state helpers."""

import random

import numpy as np
import pytest

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


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_rng_state_restore_normalizes_tensors_to_cpu(monkeypatch):
    """Device-mapped checkpoint RNG tensors must be moved back to CPU."""

    class DeviceMappedState:
        def __init__(self, cpu_state):
            self.cpu_state = cpu_state
            self.detached = False

        def detach(self):
            self.detached = True
            return self

        def cpu(self):
            assert self.detached
            return self.cpu_state

    state = runtime.capture_rng_state()
    torch_cpu_state = object()
    cuda_cpu_state = object()
    torch_state = DeviceMappedState(torch_cpu_state)
    cuda_state = DeviceMappedState(cuda_cpu_state)
    state["torch"] = torch_state
    state["cuda"] = cuda_state

    restored = {}
    monkeypatch.setattr(
        runtime.torch, "set_rng_state", lambda value: restored.update(torch=value)
    )
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        runtime.torch.cuda,
        "set_rng_state",
        lambda value: restored.update(cuda=value),
    )

    runtime.restore_rng_state(state)

    assert torch_state.detached
    assert cuda_state.detached
    assert restored == {"torch": torch_cpu_state, "cuda": cuda_cpu_state}


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_cdist_fast_supports_declared_metrics():
    """The shared tensor distance helper must implement its stated contract."""
    first = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    second = torch.tensor([[3.0, 4.0]])

    assert torch.allclose(
        runtime.cdist_fast(first, second),
        torch.tensor([[5.0], [np.sqrt(8.0)]], dtype=first.dtype),
    )
    assert runtime.cdist_fast(first, second, "cityblock").tolist() == [[7.0], [4.0]]
    assert runtime.cdist_fast(first, second, "chebyshev").tolist() == [[4.0], [2.0]]
    with pytest.raises(ValueError, match="Unsupported distance metric"):
        runtime.cdist_fast(first, second, "cosine")
