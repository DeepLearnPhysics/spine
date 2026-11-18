"""Optional sparse runtime requirements for shared UResNet tests."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("MinkowskiEngine")
pytest.importorskip("torch_cluster")
