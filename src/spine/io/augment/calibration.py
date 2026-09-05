"""Calibration-parameter augmentation for detector-response images."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral
from typing import Any

import numpy as np

from spine.calib import CalibrationManager
from spine.calib.smearing import SmearingCalibrator
from spine.data import Meta, TensorData

from .base import AugmentBase


class CalibrationAugment(AugmentBase):
    """Vary detector calibration parameters while preserving image topology.

    The default ``vary_response`` mode interprets input values as nominal raw
    detector response. It first applies the nominal correction chain and then
    runs the thrown chain in reverse, producing the response expected under
    the sampled detector conditions. Coordinates and sparse row counts are
    never changed.
    """

    name = "calibration"
    _modes = ("vary_response", "calibrate", "simulate")

    def __init__(
        self,
        features: Mapping[str, int | Sequence[int]],
        nominal: Mapping[str, Any],
        throws: Mapping[str, Mapping[str, Any]],
        mode: str = "vary_response",
        sources: str | None = None,
        run_info: str | None = None,
        noise: Mapping[str, Any] | None = None,
        p: float = 1.0,
    ) -> None:
        """Initialize calibration-aware response augmentation.

        Parameters
        ----------
        features : mapping[str, int or sequence[int]]
            Feature-relative response columns for each tensor product.
        nominal : mapping
            Nominal :class:`spine.calib.CalibrationManager` configuration.
        throws : mapping
            Parameter distributions keyed by calibration module label. A
            parameter accepts ``distribution: normal`` with ``sigma``,
            ``distribution: uniform`` with ``range``, or ``choices``. Numeric
            distributions may set ``relative: true`` and ``scope`` to either
            ``image`` or ``tpc``. ``clip`` bounds the sampled multiplicative
            factor for relative throws and the final value for absolute ones.
            A module may instead provide ``choices`` of complete configuration
            overrides, useful for paired forward and inverse signal-response
            functions.
        mode : {'vary_response', 'calibrate', 'simulate'}, default 'vary_response'
            Transformation to apply. ``calibrate`` runs the thrown correction
            chain, ``simulate`` runs its inverse, and ``vary_response`` maps
            nominal raw response into thrown raw response.
        sources : str, optional
            Key of a row-aligned ``TensorData`` containing ``(module, TPC)``
            source pairs. Geometry is used to infer TPCs when omitted.
        run_info : str, optional
            Key of the event run-information product used by database-backed
            calibration constants.
        noise : mapping, optional
            Configuration for a :class:`SmearingCalibrator` applied after the
            deterministic calibration transformation.
        p : float, default 1.0
            Event-level probability of applying this augmentation.
        """
        if mode not in self._modes:
            raise ValueError(
                f"Calibration augmentation mode not recognized: {mode}. "
                f"Must be one of {self._modes}."
            )
        if not isinstance(nominal, Mapping) or not nominal:
            raise ValueError("Calibration augmentation requires a nominal config.")
        if not isinstance(throws, Mapping) or not throws:
            raise ValueError("Calibration augmentation requires parameter throws.")

        self.features = self._parse_features(features)
        self.nominal_config: dict[str, Any] = deepcopy(dict(nominal))
        self.throws = deepcopy(dict(throws))
        self.mode = mode
        self.sources = sources
        self.run_info = run_info
        self.p = self._probability(p)

        # The nominal manager is reusable. Thrown managers are event-specific,
        # with constants sampled from these resolved nominal values.
        self.nominal = CalibrationManager(**self.nominal_config)
        self._validate_throws()
        if mode != "calibrate":
            self.nominal.validate_inverse()

        self.noise = None
        if noise is not None:
            self.noise = SmearingCalibrator(**dict(noise))

    def apply(
        self,
        data: dict[str, Any],
        meta: Meta,
        keys: list[str],
        context: dict[str, Any],
    ) -> tuple[dict[str, Any], Meta]:
        """Apply one shared calibration throw to configured image products.

        Parameters
        ----------
        data : dict
            Event products containing the selected response tensors.
        meta : Meta
            Spatial metadata used to convert voxel coordinates to centimeters.
        keys : list[str]
            Coordinate-bearing product keys discovered by the manager. The
            explicit :attr:`features` mapping controls selection instead.
        context : dict
            Shared augmentation context, unused by this module.

        Returns
        -------
        tuple[dict, Meta]
            Mutated event products and unchanged spatial metadata.
        """
        if np.random.rand() >= self.p:
            return data, meta

        run_id = self._resolve_run_id(data)
        thrown = CalibrationManager(**self._sample_config(run_id))
        if self.mode != "calibrate":
            thrown.validate_inverse()

        sources = self._resolve_sources(data)
        noise_sample = None
        if self.noise is not None and self.noise.scope == "image":
            noise_sample = self.noise.sample()

        for key, columns in self.features.items():
            product = self._resolve_product(data, key, columns)
            coords = np.asarray(product.coordinate_data)
            values = np.asarray(product.features)
            one_dimensional = values.ndim == 1
            if one_dimensional:
                values = values.reshape(-1, 1)

            if sources is not None and len(sources) != len(values):
                raise ValueError(
                    f"Calibration sources contain {len(sources)} rows, but "
                    f"product `{key}` contains {len(values)}."
                )

            for column in columns:
                response = values[:, column].astype(np.float64, copy=True)
                response = self._transform(
                    coords, response, sources, run_id, meta, thrown
                )
                if self.noise is not None:
                    response = self.noise.process(response, noise_sample)
                values[:, column] = response.astype(values.dtype, copy=False)

            product.features = values[:, 0] if one_dimensional else values

        return data, meta

    def _transform(
        self,
        points: np.ndarray,
        values: np.ndarray,
        sources: np.ndarray | None,
        run_id: int | None,
        meta: Meta,
        thrown: CalibrationManager,
    ) -> np.ndarray:
        """Run the configured correction/response composition."""
        if self.mode == "calibrate":
            _, values = thrown(points, values, sources, run_id, meta=meta)
        elif self.mode == "simulate":
            _, values = thrown(points, values, sources, run_id, meta=meta, inverse=True)
        else:
            # Pass through calibrated space so only detector conditions—not
            # the physical deposition represented by the image—are varied.
            _, values = self.nominal(points, values, sources, run_id, meta=meta)
            _, values = thrown(points, values, sources, run_id, meta=meta, inverse=True)
        return values

    def _sample_config(self, run_id: int | None) -> dict[str, Any]:
        """Return a concrete calibration configuration for one event throw."""
        config: dict[str, Any] = deepcopy(self.nominal_config)
        num_tpcs = self.nominal.geo.tpc.num_chambers
        self._materialize_constants(config, run_id, num_tpcs)

        for label, throw_config in self.throws.items():
            module_config = config[label]
            if "choices" in throw_config:
                if len(throw_config) != 1:
                    raise ValueError(
                        f"Module throw `{label}` cannot combine choices with "
                        "parameter distributions."
                    )
                choices = throw_config["choices"]
                choice = choices[np.random.randint(len(choices))]
                module_config.update(deepcopy(dict(choice)))
                for parameter in choice:
                    self._remove_database_config(module_config, parameter)
                continue

            module = self.nominal.modules[label]
            for parameter, spec in throw_config.items():
                nominal = self._nominal_value(
                    module, module_config, parameter, run_id, num_tpcs
                )
                module_config[parameter] = self._sample_parameter(
                    nominal, spec, num_tpcs
                )

                # A sampled constant supersedes its database source.
                self._remove_database_config(module_config, parameter)

        return config

    def _materialize_constants(
        self,
        config: dict[str, Any],
        run_id: int | None,
        num_tpcs: int,
    ) -> None:
        """Replace database-backed constants with this event's values.

        Besides making the sampled configuration self-contained, this avoids
        reopening calibration databases for every augmented event.
        """
        parameters = {"gain": ("gain",), "lifetime": ("lifetime", "driftv")}
        for label, module in self.nominal.modules.items():
            name = self.nominal.module_names[label]
            if name not in parameters:
                continue

            module_config = config[label]
            for parameter in parameters[name]:
                constant = getattr(module, parameter)
                module_config[parameter] = [
                    constant.get(tpc_id, run_id) for tpc_id in range(num_tpcs)
                ]
                self._remove_database_config(module_config, parameter)

    @staticmethod
    def _remove_database_config(config: dict[str, Any], parameter: str) -> None:
        """Remove database options superseded by one concrete parameter."""
        prefix = f"{parameter}_db"
        for key in tuple(config):
            if key == prefix or key.startswith(f"{prefix}_"):
                config.pop(key)

    @staticmethod
    def _nominal_value(
        module: Any,
        config: Mapping[str, Any],
        parameter: str,
        run_id: int | None,
        num_tpcs: int,
    ) -> Any:
        """Resolve a configured or database-backed nominal parameter."""
        constant = getattr(module, parameter, None)
        if constant is not None and hasattr(constant, "get"):
            return np.asarray(
                [constant.get(tpc_id, run_id) for tpc_id in range(num_tpcs)]
            )
        if parameter not in config:
            raise ValueError(
                f"Cannot throw unknown calibration parameter `{parameter}`."
            )
        return config[parameter]

    @staticmethod
    def _sample_parameter(nominal: Any, spec: Any, num_tpcs: int) -> Any:
        """Sample one scalar or per-TPC calibration parameter."""
        if not isinstance(spec, Mapping):
            raise TypeError("Calibration parameter throws must be mappings.")
        if "choices" in spec:
            choices = spec["choices"]
            if (
                not isinstance(choices, Sequence)
                or isinstance(choices, (str, bytes))
                or len(choices) == 0
            ):
                raise ValueError("Calibration throw choices must be nonempty.")
            return deepcopy(choices[np.random.randint(len(choices))])

        distribution = spec.get("distribution", "normal")
        relative = bool(spec.get("relative", False))
        scope = spec.get("scope", "image")
        if scope not in ("image", "tpc"):
            raise ValueError("Calibration throw scope must be `image` or `tpc`.")

        nominal_array = np.asarray(nominal, dtype=np.float64)
        size = num_tpcs if scope == "tpc" else None
        if distribution == "normal":
            sigma = float(spec.get("sigma", 0.0))
            if not np.isfinite(sigma) or sigma < 0.0:
                raise ValueError("Normal throw sigma must be nonnegative and finite.")
            perturbation = np.random.normal(1.0 if relative else 0.0, sigma, size)
            sample = (
                nominal_array * perturbation
                if relative
                else nominal_array + perturbation
            )
        elif distribution == "uniform":
            bounds = np.asarray(spec.get("range"), dtype=np.float64)
            if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
                raise ValueError("Uniform throws require a finite two-value range.")
            if bounds[0] > bounds[1]:
                raise ValueError("Uniform throw range must be ordered.")
            perturbation = np.random.uniform(bounds[0], bounds[1], size)
            sample = nominal_array * perturbation if relative else perturbation
        else:
            raise ValueError(
                f"Calibration throw distribution not recognized: {distribution}."
            )

        sample = np.asarray(sample)
        clip = spec.get("clip")
        if clip is not None:
            bounds = np.asarray(clip, dtype=np.float64)
            if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
                raise ValueError("Calibration throw clip must contain two values.")
            if bounds[0] > bounds[1]:
                raise ValueError("Calibration throw clip must be ordered.")
            if relative:
                perturbation = np.clip(perturbation, bounds[0], bounds[1])
                sample = nominal_array * perturbation
            else:
                sample = np.clip(sample, bounds[0], bounds[1])
        return float(sample) if sample.ndim == 0 else sample.tolist()

    def _validate_throws(self) -> None:
        """Validate throw module labels and high-level choice structure."""
        for label, config in self.throws.items():
            if label not in self.nominal.modules:
                raise ValueError(
                    f"Calibration throw refers to unknown module `{label}`."
                )
            if not isinstance(config, Mapping) or not config:
                raise ValueError(
                    f"Calibration throw for `{label}` must be a nonempty mapping."
                )
            if "choices" in config:
                choices = config["choices"]
                if (
                    not isinstance(choices, Sequence)
                    or isinstance(choices, (str, bytes))
                    or len(choices) == 0
                ):
                    raise ValueError(
                        f"Calibration module choices for `{label}` must be nonempty."
                    )
                if not all(isinstance(choice, Mapping) for choice in choices):
                    raise TypeError(
                        f"Calibration module choices for `{label}` must be mappings."
                    )

    @staticmethod
    def _parse_features(
        features: Mapping[str, int | Sequence[int]],
    ) -> dict[str, np.ndarray]:
        """Normalize selected response columns into integer arrays."""
        if not isinstance(features, Mapping) or not features:
            raise ValueError("Calibration augmentation requires a feature mapping.")

        result = {}
        for key, columns in features.items():
            if not isinstance(key, str):
                raise TypeError("Calibration product names must be strings.")
            if isinstance(columns, Integral) and not isinstance(columns, bool):
                columns = (int(columns),)
            array = np.asarray(columns)
            if (
                array.ndim != 1
                or len(array) == 0
                or not np.issubdtype(array.dtype, np.integer)
                or np.issubdtype(array.dtype, np.bool_)
            ):
                raise TypeError(
                    f"Calibration feature columns for `{key}` must be a "
                    "nonempty sequence of integers."
                )
            array = array.astype(np.int64, copy=False)
            if np.any(array < 0) or len(np.unique(array)) != len(array):
                raise ValueError(
                    f"Calibration feature columns for `{key}` must be unique "
                    "and non-negative."
                )
            result[key] = array
        return result

    @staticmethod
    def _resolve_product(
        data: dict[str, Any], key: str, columns: np.ndarray
    ) -> TensorData:
        """Fetch and validate one configured coordinate-bearing tensor."""
        if key not in data:
            raise KeyError(f"Calibration product `{key}` is missing from the event.")
        product = data[key]
        if not isinstance(product, TensorData):
            raise TypeError(f"Calibration product `{key}` must be a TensorData.")
        if product.coordinate_data is None:
            raise ValueError(f"Calibration product `{key}` must carry coordinates.")

        values = np.asarray(product.features)
        if values.ndim not in (1, 2):
            raise ValueError(
                f"Calibration product `{key}` must have one- or "
                "two-dimensional features."
            )
        width = 1 if values.ndim == 1 else values.shape[1]
        if int(np.max(columns)) >= width:
            raise IndexError(
                f"Calibration feature columns for `{key}` exceed its width {width}."
            )
        return product

    def _resolve_sources(self, data: dict[str, Any]) -> np.ndarray | None:
        """Resolve optional per-row detector source identifiers."""
        if self.sources is None:
            return None
        if self.sources not in data:
            raise KeyError(f"Calibration source product `{self.sources}` is missing.")
        product = data[self.sources]
        if not isinstance(product, TensorData):
            raise TypeError("Calibration sources must be stored in a TensorData.")
        sources = np.asarray(product.features)
        if sources.ndim != 2 or sources.shape[1] != 2:
            raise ValueError("Calibration sources must have shape (N, 2).")
        return sources.astype(np.int64, copy=False)

    def _resolve_run_id(self, data: dict[str, Any]) -> int | None:
        """Resolve the optional event run identifier."""
        if self.run_info is None:
            return None
        if self.run_info not in data:
            raise KeyError(f"Calibration run product `{self.run_info}` is missing.")
        run_info = data[self.run_info]
        if not hasattr(run_info, "run"):
            raise TypeError("Calibration run information must provide a `run` field.")
        return int(run_info.run)

    @staticmethod
    def _probability(value: float) -> float:
        """Validate and return an event-level application probability."""
        value = float(value)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("Calibration augmentation `p` must lie in [0, 1].")
        return value
