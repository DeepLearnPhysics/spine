"""Module that handles conditional imports for optional packages.

Currently wraps the following packages:
- ROOT: only needed when parsing ROOT-format data
- larcv: only needed when reading larcv-format data in parsers
- torch: only needed when running ML models and training
- MinkowskiEngine: only needed when running sparse CNNs
"""

import os
from warnings import warn

# If ROOT is available, load it
try:
    import ROOT

    ROOT_AVAILABLE = True
except ModuleNotFoundError:
    warn("ROOT could not be found, cannot parse LArCV data.")
    ROOT = None
    ROOT_AVAILABLE = False


# If LArCV is available, load it
try:
    from larcv import larcv

    LARCV_AVAILABLE = True
except ModuleNotFoundError:
    warn("larcv could not be found, cannot parse LArCV data.")
    larcv = None
    LARCV_AVAILABLE = False


# If torch is available, load it
try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    warn("PyTorch could not be found, ML functionality disabled.")

    # Create a mock torch module for type annotations
    class MockTorch:
        """Mock torch module that provides attributes for type annotations."""

        class Tensor:
            """Mock torch.Tensor for type annotations."""

            pass

        def __getattr__(self, name):
            """Return None for any other attributes."""
            return None

    torch = MockTorch()
    TORCH_AVAILABLE = False


# If MinkowskiEngine is available, load it with the right number of threads
try:
    if os.environ.get("OMP_NUM_THREADS") is None:
        os.environ["OMP_NUM_THREADS"] = "16"
    import MinkowskiEngine as ME
    import MinkowskiFunctional as MF

    ME_AVAILABLE = True
except ModuleNotFoundError:
    warn("MinkowskiEngine could not be found, cannot run sparse CNNs.")
    ME = None
    MF = None
    ME_AVAILABLE = False
