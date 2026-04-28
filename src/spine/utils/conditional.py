"""Module that handles conditional imports for optional packages.

Currently wraps the following packages:
- ROOT: only needed when parsing ROOT-format data
- larcv: only needed when reading larcv-format data in parsers
- torch: only needed when running ML models and training
- MinkowskiEngine: only needed when running sparse CNNs
"""

import os
from typing import TYPE_CHECKING, Any, Optional
from warnings import warn

__all__ = [
    "ROOT",
    "larcv",
    "torch",
    "ME",
    "MF",
    "ROOT_AVAILABLE",
    "LARCV_AVAILABLE",
    "TORCH_AVAILABLE",
    "ME_AVAILABLE",
]


# Initialize availability flags
ROOT_AVAILABLE = False
LARCV_AVAILABLE = False
TORCH_AVAILABLE = False
ME_AVAILABLE = False

# If ROOT is available, load it
if TYPE_CHECKING:
    # Type checkers see the real ROOT module
    import ROOT

    ROOT_AVAILABLE = True
else:
    # Runtime does conditional import
    ROOT: Optional[Any]
    try:
        import ROOT

        ROOT_AVAILABLE = True
    except ModuleNotFoundError:
        warn("ROOT could not be found, cannot parse LArCV data.")
        ROOT = None


# If LArCV is available, load it
if TYPE_CHECKING:
    # Type checkers see the real larcv module
    from larcv import larcv

    LARCV_AVAILABLE = True
else:
    # Runtime does conditional import
    larcv: Optional[Any]
    try:
        from larcv import larcv

        LARCV_AVAILABLE = True
    except ModuleNotFoundError:
        warn("larcv could not be found, cannot parse LArCV data.")
        larcv = None


# If torch is available, load it
if TYPE_CHECKING:
    # Type checkers see the real torch module
    import torch

    TORCH_AVAILABLE = True
else:
    # Runtime does conditional import
    torch: Optional[Any]
    try:
        import torch

        if not hasattr(torch, "Tensor"):
            raise ImportError("Imported torch module has no Tensor attribute.")

        TORCH_AVAILABLE = True
    except (ModuleNotFoundError, ImportError):
        warn("PyTorch could not be found, ML functionality disabled.")

        class DummyTorch:
            """Minimal torch stand-in for annotations and isinstance checks."""

            class Tensor:
                """Dummy tensor type."""

            def __getattr__(self, name: str) -> Any:
                raise ImportError("PyTorch is required for this functionality.")

        torch = DummyTorch()

# If MinkowskiEngine is available, load it with the right number of threads
if TYPE_CHECKING:
    # Type checkers see the real MinkowskiEngine modules
    import MinkowskiEngine as ME
    import MinkowskiFunctional as MF

    ME_AVAILABLE = True
else:
    # Runtime does conditional import
    ME: Optional[Any]
    MF: Optional[Any]
    try:
        if os.environ.get("OMP_NUM_THREADS") is None:
            os.environ["OMP_NUM_THREADS"] = "16"
        import MinkowskiEngine as ME
        import MinkowskiFunctional as MF

        ME_AVAILABLE = True
    except ModuleNotFoundError:
        warn("MinkowskiEngine could not be found, cannot run sparse CNNs.")

        class DummyME:
            """Minimal MinkowskiEngine stand-in for annotations and checks."""

            class SparseTensor:
                """Dummy sparse tensor type."""

            def __getattr__(self, name: str) -> Any:
                raise ImportError("MinkowskiEngine is required for this functionality.")

        ME = DummyME()
        MF = None
