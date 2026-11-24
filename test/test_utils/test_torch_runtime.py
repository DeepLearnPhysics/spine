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


def test_distributed_any_returns_local_value_without_process_group(monkeypatch):
    """Control flags should remain usable outside distributed execution."""
    if TORCH_AVAILABLE:
        monkeypatch.setattr(runtime.torch.distributed, "is_initialized", lambda: False)

    assert runtime.distributed_any(True)
    assert not runtime.distributed_any(False)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_distributed_any_reduces_remote_requests(monkeypatch):
    """A request raised on another rank should become visible locally."""
    calls = []
    monkeypatch.setattr(runtime.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.distributed, "is_initialized", lambda: True)

    def all_reduce(flag, op):
        calls.append(op)
        flag.fill_(1)

    monkeypatch.setattr(runtime.torch.distributed, "all_reduce", all_reduce)

    assert runtime.distributed_any(False, "cpu")
    assert calls == [runtime.torch.distributed.ReduceOp.MAX]


def test_gather_scalars_without_torch(monkeypatch):
    """Core-only installs retain useful single-process logging."""
    monkeypatch.setattr(runtime, "TORCH_AVAILABLE", False)
    assert runtime.distributed_all_gather_scalars([1.0, 2.0]) == [[1.0, 2.0]]


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_gather_scalars_uses_backend_device(monkeypatch, backend):
    """NCCL uses LOCAL_RANK's selected CUDA device; Gloo stays on CPU."""
    from types import SimpleNamespace

    calls = []

    def tensor(values, *, dtype, device):
        calls.append((dtype, device))
        return np.array(values)

    def gather(rows, local):
        rows[0][:] = local
        rows[1][:] = local + 1

    fake = SimpleNamespace(
        distributed=SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_backend=lambda: backend,
            get_world_size=lambda: 2,
            all_gather=gather,
        ),
        cuda=SimpleNamespace(current_device=lambda: 3),
        device=lambda *args: args,
        tensor=tensor,
        empty_like=np.empty_like,
        float64="float64",
    )
    monkeypatch.setattr(runtime, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(runtime, "torch", fake)
    assert runtime.distributed_all_gather_scalars([1.0, 2.0]) == [
        [1.0, 2.0],
        [2.0, 3.0],
    ]
    assert calls == [("float64", ("cuda", 3) if backend == "nccl" else ("cpu",))]


def _gather_scalars_worker(rank, rendezvous):
    from datetime import timedelta

    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        assert runtime.distributed_all_gather_scalars([float(rank), rank + 0.5]) == [
            [0.0, 0.5],
            [1.0, 1.5],
        ]
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_gather_scalars_two_processes(tmp_path):
    """Exercise the actual collective with two CPU ranks, including rank ordering."""
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo is not available")
    torch.multiprocessing.spawn(
        _gather_scalars_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
