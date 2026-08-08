"""Class to build all require representations."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from spine.data import ClusterLabelData, TensorData

from .fragment import FragmentBuilder
from .interaction import InteractionBuilder
from .particle import ParticleBuilder
from .utils import RunMode, Units, get_batch_size, is_single_index


class BuildManager:
    """Manager which constructs data representations based on the chain output.

    Takes care of two scenarios:
      - Interpret the raw output of the reconstruction chain
      - Load up existing objects stored as dictionaries
    """

    # List of recognized run modes
    _run_modes = ("reco", "truth", "both", "all")

    # List of recognized units
    _units = ("cm", "px")

    # Name of input data products needed to build representations. These names
    # are not set in stone; they can be set in the configuration
    _default_sources: ClassVar[tuple[tuple[str, tuple[str, ...]], ...]] = (
        ("data_tensor", ("data_calib", "data_adapt", "data")),
        ("data_q_tensor", ("data_adapt", "data")),
        ("label_tensor", ("clust_label",)),
        ("label_adapt_tensor", ("clust_label_adapt",)),
        ("label_g4_tensor", ("clust_label_g4",)),
        ("depositions_q_label", ("charge_label",)),
        ("graph_label", ("graph_label",)),
        ("orig_index", ("orig_index",)),
        ("orig_index_label", ("orig_index_label", "orig_index")),
        ("sources", ("sources_adapt", "sources")),
        ("sources_label", ("sources_label",)),
        ("particles", ("particles",)),
        ("neutrinos", ("neutrinos",)),
        ("flashes", ("flashes",)),
        ("crthits", ("crthits",)),
    )

    def __init__(
        self,
        fragments: bool,
        particles: bool,
        interactions: bool,
        mode: RunMode = "both",
        units: Units = "cm",
        sources: dict[str, str | tuple[str, ...] | list[str]] | None = None,
        lite: bool = False,
    ) -> None:
        """Initializes the build manager.

        Parameters
        ----------
        fragments : bool
            Build/load RecoFragment/TruthFragment objects
        particles : bool
            Build/load RecoParticle/TruthParticle objects
        interactions : bool
            Build/load RecoInteraction/TruthInteraction objects
        mode : str, default 'both'
            Whether to construct reconstructed objects, true objects or both
        sources : Dict[str, str], optional
            Dictionary which maps the necessary data products onto a name
            in the input/output dictionary of the reconstruction chain.
        lite : bool, default False
            If `True`, the objects being loaded are lite and do not map
            to long-form attributes. Simply load the matches.
        """
        # Check on the mode, store it
        if mode not in self._run_modes:
            raise ValueError(
                f"Run mode not recognized: {mode}. Must be one {self._run_modes}"
            )
        self.mode = mode

        # Check on the units, store them
        if units not in self._units:
            raise ValueError(
                f"Units not recognized: {units}. Must be one {self._units}"
            )
        self.units = units

        # Resolve the per-instance source mapping
        sources_dict = dict(self._default_sources)
        if sources is not None:
            for key, value in sources.items():
                if key not in sources_dict:
                    raise KeyError(
                        "Unexpected data product specified in `sources`: "
                        f"{key}. Should be one of {list(sources_dict.keys())}."
                    )
                if isinstance(value, str):
                    sources_dict[key] = (value,)
                else:
                    if not isinstance(value, Sequence) or isinstance(
                        value, (str, bytes)
                    ):
                        raise TypeError(
                            "Source overrides must be strings or sequences of strings."
                        )
                    if not all(isinstance(item, str) for item in value):
                        raise TypeError(
                            "Source override sequences must contain only strings."
                        )
                    sources_dict[key] = tuple(value)

        self.sources = sources_dict

        # Initialize the builders
        self.builders = OrderedDict()
        if fragments:
            self.builders["fragment"] = FragmentBuilder(mode, units)
        if particles:
            self.builders["particle"] = ParticleBuilder(mode, units)
        if interactions:
            if not particles:
                raise ValueError(
                    "Interactions are built from particles. If `interactions` "
                    "is True, so must `particles` be."
                )
            self.builders["interaction"] = InteractionBuilder(mode, units)

        # Store whether to load the long-form attributes or not
        self.lite = lite

    def __call__(self, data: dict[str, Any]) -> None:
        """Build the representations for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of input data and model outputs

        Notes
        -----
        Modifies the data dictionary in place.
        """
        # If this is the first time the builders are called, build
        # the objects shared between fragments/particles/interactions
        load = True
        if not self.lite and "points" not in data and "points_label" not in data:
            load = False
            if is_single_index(data["index"]):
                sources = self.build_sources(data)
            else:
                sources = defaultdict(list)
                for entry in range(get_batch_size(data["index"])):
                    sources_e = self.build_sources(data, entry)
                    for key, val in sources_e.items():
                        sources[key].append(val)

            data.update(**sources)

        # Loop over builders
        for name, builder in self.builders.items():
            # Build representations
            builder(data)

            # Generate match pairs from stored matches
            if load and self.mode in ["both", "all"]:
                if is_single_index(data["index"]):
                    match_dict = self.load_match_pairs(data, name)
                else:
                    match_dict = defaultdict(list)
                    for entry in range(get_batch_size(data["index"])):
                        match_dict_e = self.load_match_pairs(data, name, entry)
                        for key, val in match_dict_e.items():
                            match_dict[key].append(val)

                data.update(**match_dict)

    def build_sources(
        self, data: dict[str, Any], entry: int | None = None
    ) -> dict[str, Any]:
        """Construct the reference coordinate and value tensors used by
        all the representations built by the module.

        These objects should be stored along with the constructed objects
        if the objects are to be loaded later on.

        Parameters
        ----------
        data : dict
            Dictionary of input data and model outputs
        entry : int, optional
            Entry number

        Returns
        -------
        dict
            Long-form coordinate, deposition, source, and truth arrays used by
            the configured representation builders.
        """
        # Fetch the orginal sources
        sources = {}
        for key, alt_keys in self.sources.items():
            for alt in alt_keys:
                if alt in data:
                    sources[key] = data[alt]
                    if entry is not None:
                        sources[key] = data[alt][entry]
                    break

        # Build aditional information
        update = {}
        if self.mode != "truth" or "label_adapt_tensor" in sources:
            # Geometry and preferred depositions follow the calibrated view,
            # while charge follows the adapted/input view. If either optional
            # view is absent, source resolution has already selected its
            # documented fallback.
            data_tensor: TensorData = sources["data_tensor"]
            data_q_tensor: TensorData = sources.get("data_q_tensor", data_tensor)
            update["points"] = data_tensor.coordinates()
            update["depositions"] = self._primary_feature(data_tensor)
            update["depositions_q"] = self._primary_feature(data_q_tensor)

            if "sources" in sources:
                source_tensor: TensorData = sources["sources"]
                update["sources"] = source_tensor.features.astype(np.int32, copy=False)

            if "orig_index" in sources:
                update["orig_index"] = sources["orig_index"].features.astype(
                    np.int32, copy=False
                )

        if self.mode != "reco":
            label: ClusterLabelData = sources["label_tensor"]
            update["label_tensor"] = label
            update["points_label"] = label.coords
            update["depositions_label"] = label.values

            if "depositions_q_label" in sources:
                depositions_q: TensorData = sources["depositions_q_label"]
                update["depositions_q_label"] = depositions_q.values

            if "label_adapt_tensor" in sources:
                adapted: ClusterLabelData = sources["label_adapt_tensor"]
                update["label_adapt_tensor"] = adapted
                update["depositions_label_adapt"] = adapted.values

            if "label_g4_tensor" in sources:
                label_g4: ClusterLabelData = sources["label_g4_tensor"]
                update["label_g4_tensor"] = label_g4
                update["points_g4"] = label_g4.coords
                update["depositions_g4"] = label_g4.values

            if "sources_label" in sources:
                source_label: TensorData = sources["sources_label"]
                update["sources_label"] = source_label.features.astype(
                    np.int32, copy=False
                )

            if "orig_index_label" in sources:
                update["orig_index_label"] = sources[
                    "orig_index_label"
                ].features.astype(np.int32, copy=False)

        # If provided, etch the point attributes to check their units
        for obj in ["fragment", "particle"]:
            for key in [f"{obj}_start_points", f"{obj}_end_points"]:
                if key in data:
                    update[key] = data[key]
                    if entry is not None:
                        update[key] = update[key][entry]

        # Convert everything to the proper units once and for all
        if self.units != "px":
            # Fetch metadata
            if "meta" not in data:
                raise KeyError("Must provide metadata to build objects in cm.")

            meta = data["meta"][entry] if entry is not None else data["meta"]
            for key in update:
                if "points" in key and key in update:
                    if key in update:
                        update[key] = meta.to_cm(np.copy(update[key]), center=True)

            for key in ["particles", "neutrinos"]:
                if key in sources:
                    update[key] = sources[key]
                    for obj in sources[key]:
                        if obj.units != self.units:
                            obj.to_cm(meta)

        return update

    @staticmethod
    def _primary_feature(tensor: TensorData) -> np.ndarray:
        """Return the first logical feature of a point tensor.

        Parameters
        ----------
        tensor : TensorData
            Point data whose packed layout may contain multiple features.

        Returns
        -------
        np.ndarray
            One scalar value per point row.

        Notes
        -----
        Reconstruction depositions conventionally use feature zero. Accessing
        ``features`` explicitly also supports multi-feature inputs, for which
        the stricter ``values`` convenience property is intentionally invalid.
        """
        features = tensor.features
        return features if features.ndim == 1 else features[:, 0]

    @staticmethod
    def load_match_pairs(
        data: dict[str, Any], name: str, entry: int | None = None
    ) -> dict[str, list[Any]]:
        """Generate lists of matched object pairs from stored matches.

        Parameters
        ----------
        data : dict
            Dictionary of input data and model outputs
        name : str
            Object type name
        entry : int, optional
            Entry number
        """
        # Initialize the name of the match lists
        prefix = f"{name}_matches"

        # Create match pairs in both directions (true to reco and vice versa)
        result = {}
        for source, target in [("reco", "truth"), ("truth", "reco")]:
            # Fetch the lists of objects to match
            sources = data[f"{source}_{name}s"]
            targets = data[f"{target}_{name}s"]
            if entry is not None:
                sources, targets = sources[entry], targets[entry]

            # Loop
            suffix = f"{source[0]}2{target[0]}"
            match_key = f"{prefix}_{suffix}"
            match_overlap_key = f"{match_key}_overlap"
            result[match_key] = []
            result[match_overlap_key] = []
            for obj in sources:
                if not obj.is_matched:
                    # If no match is found, give an empty value to the match
                    result[match_key].append((obj, None))
                    result[match_overlap_key].append(-1.0)

                else:
                    # If a match is found, the first is always the best match
                    best_match = obj.match_ids[0]
                    result[match_key].append((obj, targets[best_match]))
                    result[match_overlap_key].append(obj.match_overlaps[0])

        return result
