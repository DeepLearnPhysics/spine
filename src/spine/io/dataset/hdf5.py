"""Dataset wrapper around :class:`spine.io.read.HDF5Reader`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from spine.config.factory import instantiate, module_dict
from spine.logging import logger
from spine.utils.conditional import TORCH_AVAILABLE

from ..parse import hdf5 as parse_hdf5
from ..read import HDF5Reader, StageHDF5Reader
from .base import BaseDataset, DataDict

__all__ = ["HDF5Dataset"]

PARSER_DICT = module_dict(parse_hdf5)


class HDF5Dataset(BaseDataset):
    """Torch dataset wrapper around flat or staged HDF5 readers.

    The dataset can operate in two modes:

    - flat HDF5 mode, backed by :class:`spine.io.read.HDF5Reader`
    - staged cache mode, backed by :class:`spine.io.read.StageHDF5Reader`

    In both cases the dataset exposes a uniform parser-driven interface to the
    DataLoader layer. Reader-produced metadata such as entry indexes and source
    provenance are forwarded automatically alongside any parsed products.
    """

    name: ClassVar[str] = "hdf5"
    parsers: dict[str, Any]
    reader: HDF5Reader | StageHDF5Reader

    def __init__(
        self,
        dtype: str | None = None,
        staged: bool = False,
        stage: str | None = None,
        stage_map: Mapping[str, str] | None = None,
        schema: Mapping[str, Mapping[str, Any]] | None = None,
        keys: Sequence[str] | None = None,
        skip_keys: Sequence[str] | None = None,
        overlay_methods: Mapping[str, str] | None = None,
        augment: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Instantiate the HDF5-backed dataset.

        Parameters
        ----------
        dtype : str, optional
            Floating-point dtype forwarded to parser factories
        staged : bool, default False
            If `True`, use :class:`StageHDF5Reader` as the backend instead of
            the flat :class:`HDF5Reader`
        stage : str, optional
            Default stage name to read when `staged=True`. Individual schema
            entries may override this with their own ``stage`` field.
        stage_map : mapping, optional
            Explicit map from raw product keys to stage names when
            ``staged=True``. Schema-derived routing is merged into this map;
            conflicting assignments are rejected.
        schema : mapping, optional
            Parser schema used to reconstruct higher-level products
        keys : sequence[str], optional
            Explicit list of raw HDF5 products to keep
        skip_keys : sequence[str], optional
            Explicit list of raw HDF5 products to drop
        overlay_methods : mapping, optional
            Explicit overlay-method overrides for raw-product mode
        augment : mapping, optional
            Augmentation applied to each loaded sample
        **kwargs : Any
            Reader-specific keyword arguments forwarded to the selected HDF5
            backend reader
        """
        # Initialize parent class
        super().__init__()

        # Validate the configuration and prepare reader arguments before
        # instantiating the backend.
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use HDF5Dataset.")
        if keys is not None and skip_keys is not None:
            raise ValueError("Provide either `keys` or `skip_keys`, not both.")
        if not staged and stage is not None:
            raise ValueError("`stage` can only be provided when `staged=True`.")
        if not staged and stage_map is not None:
            raise ValueError("`stage_map` can only be provided when `staged=True`.")

        self.keys = set(keys) if keys is not None else None
        self.skip_keys = set(skip_keys) if skip_keys is not None else set()
        self.parsers = {}
        self._overlay_methods = (
            dict(overlay_methods) if overlay_methods is not None else None
        )
        reader_stage_map = dict(stage_map or {})

        # If a parser schema is provided, instantiate the parsers and collect
        # the raw HDF5 products they require. In staged mode, also validate
        # schema-level stage assignments and build the reader key-to-stage map.
        if schema is not None:
            if dtype is None:
                raise ValueError("An explicit `dtype` is required when using `schema`.")
            inferred_keys = []
            for data_product, parser_cfg in schema.items():
                parser_cfg = dict(parser_cfg)
                parser_stage = parser_cfg.pop("stage", stage)
                parser = instantiate(
                    PARSER_DICT, parser_cfg, alt_name="parser", dtype=dtype
                )
                self.parsers[data_product] = parser
                for key in parser.tree_keys:
                    if key not in inferred_keys:
                        inferred_keys.append(key)
                    if staged and parser_stage is not None:
                        existing_stage = reader_stage_map.get(key)
                        if (
                            existing_stage is not None
                            and existing_stage != parser_stage
                        ):
                            raise ValueError(
                                f"Conflicting staged HDF5 schema for raw product '{key}': "
                                f"'{existing_stage}' vs '{parser_stage}'."
                            )
                        reader_stage_map[key] = parser_stage

            if self.keys is None:
                self.keys = set(inferred_keys)
            else:
                self.keys.update(inferred_keys)

        # Initialize the appropriate reader backend
        if staged:
            self.reader = StageHDF5Reader(
                stage=stage,
                stage_map=reader_stage_map,
                keys=tuple(self.keys) if self.keys is not None else None,
                **kwargs,
            )
        else:
            # Push the dataset's product selection into the reader. This is a
            # physical I/O projection, not merely a post-read dictionary
            # filter: neither V1 references nor V2 product groups outside the
            # requested schema are dereferenced/read.
            self.reader = HDF5Reader(
                keys=tuple(self.keys) if self.keys is not None else None,
                **kwargs,
            )

        # Initialize the augmenter
        self.build_augmenter(augment)

    def __len__(self) -> int:
        """Return the number of entries exposed by the backend reader."""
        return len(self.reader)

    def __getitem__(self, idx: int) -> DataDict:
        """Return one cached dataset entry.

        Parameters
        ----------
        idx : int
            Dataset entry index.

        Returns
        -------
        dict
            Either the raw reader output (optionally filtered to ``keys``) or
            a parsed dictionary containing standard metadata plus the products
            described by ``schema``.
        """
        result = self.reader[idx]

        # Apply raw-product projection while retaining administrative metadata
        if self.keys is not None:
            keep = self.keys.union(self._index_keys).union(self._source_keys)
            result = {key: val for key, val in result.items() if key in keep}

        for key in self.skip_keys:
            result.pop(key, None)

        if not self.parsers:
            return self.apply_augmenter(result)

        # Reconstruct configured logical products from the projected raw inputs
        parsed = self.metadata_dict(result)
        for name, parser in self.parsers.items():
            try:
                parsed[name] = parser(result)
            except Exception as err:
                logger.error("Failed to produce %s using %s", name, parser)
                raise err

        return self.apply_augmenter(parsed)

    @property
    def overlay_methods(self) -> dict[str, str]:
        """Return the overlay method for each exposed HDF5 product.

        Returns
        -------
        dict[str, str]
            Mapping from dataset output key to overlay strategy.
        """
        overlay_methods = self.index_overlay_methods()
        if self.parsers:
            for name, parser in self.parsers.items():
                overlay_methods[name] = parser.overlay_method
        if self._overlay_methods is not None:
            overlay_methods.update(self._overlay_methods)

        return overlay_methods

    @property
    def data_keys(self) -> tuple[str, ...]:
        """Return the names of all data products exposed by the dataset.

        Returns
        -------
        tuple[str, ...]
            Ordered tuple of metadata and parser-product keys.
        """
        if self.parsers:
            return (*self._index_keys, *self._source_keys, *self.parsers.keys())

        # Prefer configured projection order; otherwise discover it from a sample
        if self.keys is not None:
            selected = tuple(self.keys)
            return (*self._index_keys, *self._source_keys, *selected)
        sample = self[0] if len(self) else {}

        return tuple(sample)
