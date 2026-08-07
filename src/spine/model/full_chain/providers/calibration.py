"""Voxel-calibration provider for the full reconstruction chain."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from spine.calib import CalibrationManager
from spine.data import TensorBatch

from ..registry import ProviderSpec, register_provider
from ..stage import ChainStage
from ..state import ChainState, StageResult


class CalibrationStage(ChainStage):
    """Attach calibrated or truth energy beside the input-charge tensor.

    Calibration copies the charge representation before changing values or
    coordinates, then makes the calibrated representation active without
    discarding charge. This prevents an optional calibration stage from
    mutating driver-owned input in place.
    """

    requires = frozenset({"point_data"})
    optional = frozenset({"energy_label", "meta", "run_info"})
    provides = frozenset({"point_data"})
    replaces = frozenset({"point_data"})

    def __init__(
        self,
        name: str,
        mode: str,
        calibrator: CalibrationManager | None,
    ) -> None:
        """Initialize calibration mode and correction manager.

        Parameters
        ----------
        name : str
            Stage name.
        mode : {"apply", "label"}
            Apply detector corrections or replace charge with truth energy.
        calibrator : CalibrationManager, optional
            Detector correction manager required by ``apply`` mode.
        """
        super().__init__(name)
        if mode not in {"apply", "label"}:
            raise ValueError("Calibration mode must be `apply` or `label`.")
        self.mode = mode
        self.calibrator = calibrator

    @staticmethod
    def _copy(data: TensorBatch) -> TensorBatch:
        """Copy a tensor batch while retaining logical metadata.

        Parameters
        ----------
        data : TensorBatch
            Source point tensor in input charge units.

        Returns
        -------
        TensorBatch
            Independent tensor with the same batching, schema, and metadata.
        """
        return TensorBatch(
            data.torch_tensor().clone(),
            data.counts,
            has_batch_col=data.has_batch_col,
            coord_cols=data.coord_cols,
            schema=data.schema,
            meta=data.meta,
        )

    def forward(self, state: ChainState) -> StageResult:
        """Apply calibration to a copy of the canonical charge tensor.

        Parameters
        ----------
        state : ChainState
            State containing aligned point data and optional calibration inputs.

        Returns
        -------
        StageResult
            Replacement aligned point family containing calibrated data while
            retaining the original charge representation.

        Raises
        ------
        ValueError
            If required labels or metadata are missing or cannot be aligned.
        RuntimeError
            If applied calibration was configured without a calibrator.
        """
        point_data = state.require("point_data", self.name)

        # Always calibrate from the preserved charge view. This keeps repeated
        # stage ordering deterministic and avoids modifying driver-owned data.
        data: TensorBatch = self._copy(point_data.data_q)
        value_column = int(data.feature_columns()[0])

        # Truth mode performs a direct row-aligned feature replacement.
        if self.mode == "label":
            energy_label: TensorBatch | None = state.get("energy_label")
            if energy_label is None:
                raise ValueError("Label calibration requires `energy_label`.")
            energy_label = point_data.align(energy_label)
            energy_values = energy_label.values.torch_tensor()
            data.torch_tensor()[:, value_column] = energy_values
        else:
            # Applied calibration requires event metadata and may additionally
            # use detector sources and time-dependent run information.
            if self.calibrator is None:
                raise RuntimeError("Calibration manager was not initialized.")
            meta = state.get("meta")
            if meta is None or len(meta) == 0:
                raise ValueError("Applied calibration requires `meta`.")
            sources = point_data.sources
            source_tensor = None if sources is None else sources.to_numpy().tensor
            run_info = state.get("run_info")
            repeat = data.batch_size // len(meta)
            if repeat * len(meta) != data.batch_size:
                raise ValueError("Metadata entries do not evenly cover the data batch.")

            # Calibrators operate event by event in NumPy and may update both
            # spatial coordinates and charge values.
            data_np = data.to_numpy()
            values_np = data_np.values
            for batch_id in range(data.batch_size):
                lower, upper = data.edges[batch_id : batch_id + 2]
                source_rows = (
                    None if source_tensor is None else source_tensor[lower:upper]
                )
                meta_index = batch_id // repeat
                run_id = None if run_info is None else run_info[meta_index].run
                points, values = self.calibrator(
                    data_np.coords[batch_id],
                    values_np[batch_id],
                    source_rows,
                    run_id,
                    meta=meta[meta_index],
                    module_id=batch_id % repeat,
                )
                if self.calibrator.update_points:
                    columns = data.coordinate_columns()
                    data.torch_tensor()[lower:upper, columns] = torch.as_tensor(
                        np.asarray(points).copy(),
                        dtype=data.dtype,
                        device=data.device,
                    )
                data.torch_tensor()[lower:upper, value_column] = torch.as_tensor(
                    np.asarray(values).copy(),
                    dtype=data.dtype,
                    device=data.device,
                )

        return StageResult({"point_data": point_data.with_calibration(data)})


def build_calibration_stage(
    name: str,
    config: dict[str, Any],
    _owner: Any,
) -> ChainStage:
    """Build a label or detector-calibration provider.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Provider configuration containing ``mode`` and optional
        ``calibration`` options.
    _owner : object
        Full-chain module owner, unused because calibration has no trainable
        module.

    Returns
    -------
    ChainStage
        Configured calibration adapter.
    """
    mode = config.get("mode")
    if not isinstance(mode, str):
        raise ValueError("Calibration requires a string `mode`.")
    calibration = config.get("calibration")
    if calibration is None:
        calibration = {}
    elif not isinstance(calibration, dict):
        raise TypeError("Calibration configuration must be a mapping.")
    manager = CalibrationManager(**calibration) if mode == "apply" else None
    return CalibrationStage(name, mode, manager)


PROVIDER_SPEC = register_provider(ProviderSpec("calibration", build_calibration_stage))
