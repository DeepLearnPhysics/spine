"""Shared interface utilities for the optional OpT0Finder dependency."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["OpT0FinderLightModel", "get_flashmatch", "load_flashmatch_config"]


def get_flashmatch() -> Any:
    """Load the optional OpT0Finder Python bindings at runtime."""
    return import_module("flashmatch").flashmatch


def load_flashmatch_config(
    cfg: str, detector: str | None, parent_path: str | None = None
) -> tuple[Any, Any]:
    """Load an OpT0Finder configuration and detector description.

    This performs only the common environment, detector and configuration
    setup. In particular, it does not construct a flash-matching manager, so a
    configuration containing only one flash-hypothesis algorithm is sufficient
    for callers which only need an optical response prediction.

    Parameters
    ----------
    cfg : str
        Path to an OpT0Finder configuration file
    detector : str, optional
        Detector suffix used to select ``detector_specs_<detector>.cfg``. If
        omitted, the generic ``detector_specs.cfg`` file is used
    parent_path : str, optional
        Parent analysis-configuration directory used to resolve a relative
        ``cfg`` path

    Returns
    -------
    tuple[module, flashmatch::PSet]
        Loaded Python interface and parsed OpT0Finder configuration
    """
    # Add the OpT0Finder Python interface and shared library to their loaders
    basedir = os.getenv("FMATCH_BASEDIR")
    if basedir is None:
        raise ValueError(
            "You need to source OpT0Finder's configure.sh or set the "
            "FMATCH_BASEDIR environment variable before running flash "
            "matching."
        )
    python_path = os.path.join(basedir, "python")
    if python_path not in sys.path:
        sys.path.append(python_path)

    lib_path = os.path.join(basedir, "build/lib")
    loader_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_path not in loader_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{lib_path}:{loader_path}" if loader_path else lib_path
        )

    # OpT0Finder uses its data directory to resolve auxiliary model resources
    if "FMATCH_DATADIR" not in os.environ:
        os.environ["FMATCH_DATADIR"] = os.path.join(basedir, "dat")

    # Load the detector geometry shared by all OpT0Finder algorithms
    filename = "detector_specs.cfg"
    if detector is not None:
        filename = f"detector_specs_{detector}.cfg"
    det_cfg = os.path.join(basedir, "dat", filename)
    if not os.path.isfile(det_cfg):
        raise FileNotFoundError(f"Cannot find detector specification file: {det_cfg}.")

    flashmatch = get_flashmatch()
    flashmatch.DetectorSpecs.GetME(det_cfg)

    # Resolve the algorithm configuration relative to the parent YAML file
    if parent_path is not None and not os.path.isfile(cfg):
        cfg = os.path.join(parent_path, cfg)
    if not os.path.isfile(cfg):
        raise FileNotFoundError(f"Cannot find flash-matcher config: {cfg}")

    return flashmatch, flashmatch.CreateFMParamsFromFile(cfg)


class OpT0FinderLightModel:
    """Query one OpT0Finder flash-hypothesis algorithm for a light source.

    The input photon weights are normalized to one before constructing the
    ``QCluster_t``. Summing the resulting per-channel prediction therefore
    yields an effective detector response in PE per emitted photon, including
    the configured channel masks and optical efficiencies.
    """

    def __init__(
        self,
        cfg: str,
        detector: str | None,
        parent_path: str | None = None,
        algorithm: str = "SemiAnalyticalModel",
    ) -> None:
        """Initialize a standalone OpT0Finder light-response model.

        Parameters
        ----------
        cfg : str
            OpT0Finder configuration containing the selected algorithm block.
            A full ``FlashMatchManager`` configuration is not required
        detector : str, optional
            Detector to use for the optical detector specifications
        parent_path : str, optional
            Parent directory used to resolve a relative ``cfg`` path
        algorithm : str, default 'SemiAnalyticalModel'
            Name of the OpT0Finder flash-hypothesis algorithm to query
        """
        flashmatch, fm_cfg = load_flashmatch_config(cfg, detector, parent_path)
        hypothesis = flashmatch.FlashHypothesisFactory.get().create(
            algorithm, algorithm
        )
        algo_cfg = fm_cfg.get["flashmatch::FMParams"](algorithm)
        hypothesis.Configure(algo_cfg)

        self.flashmatch = flashmatch
        self.hypothesis = hypothesis
        self.algorithm = algorithm

    def get_response(
        self,
        points: ArrayLike,
        weights: ArrayLike | None = None,
    ) -> float:
        """Return the total predicted PE per emitted photon.

        Parameters
        ----------
        points : Sequence[float] or Sequence[Sequence[float]]
            One ``(3)`` point or an ``(N, 3)`` point cloud in detector
            coordinates, in cm
        weights : Sequence[float], optional
            Non-negative relative photon yield at each point. If omitted, all
            points receive equal weight. The weights are normalized internally
            so their absolute scale does not affect the response

        Returns
        -------
        float
            Total predicted PE per emitted photon. A non-finite or non-positive
            response indicates that the source cannot support a light estimate
        """
        point_array = np.asarray(points, dtype=np.float64)
        if point_array.shape == (3,):
            point_array = point_array[None, :]
        if (
            point_array.ndim != 2
            or point_array.shape[1] != 3
            or len(point_array) == 0
            or not np.all(np.isfinite(point_array))
        ):
            raise ValueError("Light-model points must have finite shape (N, 3).")

        if weights is None:
            weight_array = np.ones(len(point_array), dtype=np.float64)
        else:
            weight_array = np.asarray(weights, dtype=np.float64)
            if weight_array.shape != (len(point_array),):
                raise ValueError(
                    "Light-model weights must contain one value per point."
                )
        if (
            np.any(~np.isfinite(weight_array))
            or np.any(weight_array < 0.0)
            or np.sum(weight_array) <= 0.0
        ):
            raise ValueError(
                "Light-model weights must be finite, non-negative and have a "
                "positive sum."
            )

        # Normalize to one emitted photon so the output is an effective response
        weight_array = weight_array / np.sum(weight_array)
        qcluster = self.flashmatch.QCluster_t()
        for point, weight in zip(point_array, weight_array):
            qcluster.push_back(
                self.flashmatch.QPoint_t(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    float(weight),
                )
            )

        estimate = self.hypothesis.GetEstimate(qcluster)
        return float(np.sum(np.asarray(list(estimate.pe_v), dtype=np.float64)))
