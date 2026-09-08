"""Training-time randomization of sparse convolution lattice phase."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch

from spine.model.sparse import SparseTensor

__all__ = ["LatticePhase"]


class LatticePhase(torch.nn.Module):
    """Randomly translate sparse backend coordinates during training.

    The operation changes only the coordinate map consumed by sparse
    convolutions. Canonical input coordinates remain available through the
    resulting :class:`SparseTensor`, so position-dependent feature channels
    and public model outputs stay in their original frame.
    """

    def __init__(
        self,
        dimension: int,
        period: int | Sequence[int],
    ) -> None:
        """Initialize the per-event lattice-phase sampler.

        Parameters
        ----------
        dimension : int
            Number of spatial coordinate dimensions.
        period : int or sequence of int
            Number of lattice phases sampled on each axis. A scalar applies
            to every dimension.

        Raises
        ------
        TypeError
            If a period is boolean or non-integral.
        ValueError
            If the dimension or any period is not positive, or the sequence
            length does not match ``dimension``.
        """
        super().__init__()
        if dimension < 1:
            raise ValueError("Lattice phase dimension must be positive.")

        if isinstance(period, Integral) and not isinstance(period, bool):
            periods = (int(period),) * dimension
        else:
            if isinstance(period, (str, bytes)) or not isinstance(period, Sequence):
                raise TypeError(
                    "Lattice phase `period` must be an integer or sequence."
                )
            if len(period) != dimension:
                raise ValueError(
                    f"Lattice phase period has dimension {len(period)}, "
                    f"expected {dimension}."
                )
            periods = tuple(period)

        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in periods
        ):
            raise TypeError("Every lattice phase period must be an integer.")
        if any(int(value) < 1 for value in periods):
            raise ValueError("Every lattice phase period must be positive.")

        self.dimension = dimension
        self.period = tuple(int(value) for value in periods)

    @classmethod
    def from_config(
        cls,
        dimension: int,
        config: Mapping[str, Any],
        total_stride: int | None = None,
    ) -> "LatticePhase":
        """Build a sampler from a validated CNN ``lattice`` block.

        Parameters
        ----------
        dimension : int
            Number of spatial coordinate dimensions.
        config : mapping
            Lattice configuration containing a numeric ``period`` or the
            string ``"auto"``.
        total_stride : int, optional
            Effective stride of the deepest CNN lattice. This is used as the
            period when automatic selection is requested.

        Returns
        -------
        LatticePhase
            Configured training-time phase sampler.
        """
        if not isinstance(config, Mapping):
            raise TypeError("CNN `lattice` configuration must be a mapping.")
        cfg = dict(config)
        if "period" not in cfg:
            raise ValueError("CNN lattice configuration requires `period`.")
        period = cfg.pop("period")
        if cfg:
            unexpected = ", ".join(sorted(cfg))
            raise TypeError(f"Unexpected CNN lattice option(s): {unexpected}.")

        # The total stride enumerates every distinct phase of the deepest
        # lattice while automatically tracking changes to encoder depth.
        if period == "auto":
            if total_stride is None:
                raise ValueError(
                    "Automatic lattice period requires the CNN total stride."
                )
            period = total_stride

        return cls(dimension, period)

    def forward(self, tensor: SparseTensor) -> SparseTensor:
        """Apply one independently sampled phase to each sparse image."""
        if not self.training:
            return tensor

        # Sampling occurs on the coordinate device and is naturally governed
        # by PyTorch's per-process RNG state.
        phases = torch.stack(
            [
                torch.randint(
                    period,
                    (tensor.batch_size,),
                    device=tensor.C.device,
                    dtype=tensor.C.dtype,
                )
                for period in self.period
            ],
            dim=1,
        )
        return tensor.rephase(phases)
