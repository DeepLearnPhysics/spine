"""Regression tests for the full reconstruction chain."""

import copy
import os
import warnings
from pathlib import Path

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from spine.config import load_config_file
from spine.driver import Driver
from spine.utils.conditional import TORCH_AVAILABLE

FULL_CHAIN_REFERENCE = (
    (4.5380635261535645, 0.9802090525627136),
    (2.7023134231567383, 0.952360987663269),
    (3.301724910736084, 0.9800845384597778),
)
FULL_CHAIN_REFERENCE_ATOL = 1.0e-5


@pytest.fixture(name="cuda_available")
def fixture_cuda_available() -> bool:
    """Check whether CUDA is available to PyTorch.

    Returns
    -------
    bool
        True if PyTorch is installed and has access to at least one CUDA device
    """
    if not TORCH_AVAILABLE:
        return False

    import torch

    return torch.cuda.is_available()


@pytest.fixture(name="deterministic_cuda")
def fixture_deterministic_cuda(cuda_available: bool) -> bool:
    """Configure PyTorch CUDA execution for deterministic test behavior.

    Parameters
    ----------
    cuda_available : bool
        Whether PyTorch has access to a CUDA device

    Returns
    -------
    bool
        True if CUDA is available and deterministic flags were set
    """
    if not cuda_available:
        return False

    import torch

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return True


def make_full_chain_config(larcv_data: str, tmp_path: Path) -> dict:
    """Build a deterministic full-chain test configuration.

    Parameters
    ----------
    larcv_data : str
        Path to the small LArCV input file
    tmp_path : Path
        Temporary directory used for logs

    Returns
    -------
    dict
        Full-chain driver configuration
    """
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "test_full_chain.yaml"
    cfg = load_config_file(str(cfg_path))

    cfg["base"]["iterations"] = len(FULL_CHAIN_REFERENCE)
    cfg["base"]["seed"] = 0
    cfg["base"]["log_dir"] = str(tmp_path)
    cfg["base"]["prefix_log"] = False
    cfg["base"]["split_output"] = False
    cfg["io"]["loader"]["num_workers"] = 0
    cfg["io"]["loader"]["dataset"]["file_keys"] = larcv_data
    cfg["io"].pop("writer", None)

    return cfg


def run_full_chain(cfg: dict) -> tuple[tuple[float, float], ...]:
    """Run the full reconstruction chain and collect loss/accuracy values.

    Parameters
    ----------
    cfg : dict
        Driver configuration

    Returns
    -------
    tuple
        Loss and accuracy for each configured iteration
    """
    driver = Driver(copy.deepcopy(cfg))
    outputs = []
    for iteration in range(len(FULL_CHAIN_REFERENCE)):
        output = driver.process(iteration=iteration)
        outputs.append((float(output["loss"]), float(output["accuracy"])))

    return tuple(outputs)


def assert_full_chain_repeatable(
    outputs: tuple[tuple[float, float], ...],
    repeat_outputs: tuple[tuple[float, float], ...],
) -> None:
    """Check that the full chain is exactly repeatable on this machine."""
    assert outputs == repeat_outputs


def assert_full_chain_reference(outputs: tuple[tuple[float, float], ...]) -> None:
    """Check full-chain loss and accuracy against reference values.

    Small differences can occur across CPU architectures or math libraries.
    Warn on tiny drift so it stays visible, but fail on larger movement.

    Parameters
    ----------
    outputs : tuple
        Loss and accuracy values produced by the full chain

    Returns
    -------
    None
        This function does not return anything
    """
    for iteration, ((loss, accuracy), (ref_loss, ref_accuracy)) in enumerate(
        zip(outputs, FULL_CHAIN_REFERENCE)
    ):
        for name, value, reference in (
            ("loss", loss, ref_loss),
            ("accuracy", accuracy, ref_accuracy),
        ):
            if value == reference:
                continue

            difference = abs(value - reference)
            if difference <= FULL_CHAIN_REFERENCE_ATOL:
                warnings.warn(
                    f"Full-chain {name} at iteration {iteration} differs from "
                    f"reference by {difference:.3g}: {value} != {reference}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            assert difference <= FULL_CHAIN_REFERENCE_ATOL, (
                f"Full-chain {name} at iteration {iteration} differs from "
                f"reference by {difference:.3g}: {value} != {reference}"
            )


@pytest.mark.slow
def test_full_chain_larcv_regression(larcv_data: str, tmp_path: Path) -> None:
    """Test full-chain loss and accuracy on a small LArCV file.

    Parameters
    ----------
    larcv_data : str
        Path to the small LArCV input file
    tmp_path : Path
        Temporary directory used for logs

    Returns
    -------
    None
        This test does not return anything
    """
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch is required to run the full-chain regression test.")

    cfg = make_full_chain_config(larcv_data, tmp_path)
    outputs = run_full_chain(cfg)
    repeat_outputs = run_full_chain(cfg)
    assert_full_chain_repeatable(outputs, repeat_outputs)
    assert_full_chain_reference(outputs)


@pytest.mark.slow
@pytest.mark.gpu
def test_full_chain_larcv_regression_gpu(
    larcv_data: str, tmp_path: Path, deterministic_cuda: bool
) -> None:
    """Test full-chain loss and accuracy on one GPU process.

    Parameters
    ----------
    larcv_data : str
        Path to the small LArCV input file
    tmp_path : Path
        Temporary directory used for logs
    deterministic_cuda : bool
        Whether CUDA is available and deterministic flags were set

    Returns
    -------
    None
        This test does not return anything
    """
    if not deterministic_cuda:
        pytest.skip("A CUDA-capable PyTorch installation is required.")

    cfg = make_full_chain_config(larcv_data, tmp_path)
    cfg["base"]["world_size"] = 1
    outputs = run_full_chain(cfg)
    assert_full_chain_reference(outputs)
