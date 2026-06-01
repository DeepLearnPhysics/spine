"""Dataset wrapper around :class:`spine.io.read.LArCVReader`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from spine.utils.factory import instantiate, module_dict
from spine.utils.logger import logger

from ..parse import larcv as parse_larcv
from ..read import LArCVReader
from .base import BaseDataset, DataDict

PARSER_DICT = module_dict(parse_larcv)

__all__ = ["LArCVDataset"]


class LArCVDataset(BaseDataset):
    """Torch dataset that parses LArCV entries into SPINE products.

    The dataset wraps :class:`spine.io.read.LArCVReader` and a parser schema.
    The schema maps output product names to parser configurations from
    :mod:`spine.io.parse.larcv`. During initialization, the dataset
    instantiates each parser, collects every LArCV tree key required by those
    parsers, and passes the union of those tree keys to the reader.

    Each loaded entry is returned as a dictionary containing standard dataset
    metadata, such as ``index`` and source-file provenance fields, plus one
    parsed product per schema entry. Optional augmentation is applied after all
    parser products are produced.
    """

    name: ClassVar[str] = "larcv"
    parsers: dict[str, Any]
    reader: LArCVReader

    def __init__(
        self,
        schema: Mapping[str, Mapping[str, Any]],
        dtype: str,
        augment: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Instantiate the LArCV-backed dataset.

        Parameters
        ----------
        schema : mapping
            Mapping from output product name to parser configuration. Each
            parser configuration must identify a parser from
            :mod:`spine.io.parse.larcv` using ``parser`` or ``name`` and
            provide any parser-specific LArCV product names.
        dtype : str
            Floating-point dtype forwarded to parser factories.
        augment : mapping, optional
            Augmentation configuration applied to each parsed sample.
        **kwargs : Any
            Reader-specific keyword arguments forwarded to
            :class:`spine.io.read.LArCVReader`, such as ``file_keys`` and
            entry-list filters.
        """
        # Initialize the parent class
        super().__init__()

        # Instantiate the configured parsers and collect the LArCV tree keys
        # needed by any parser.
        self.parsers = {}
        tree_keys: list[str] = []
        for data_product, parser_cfg in schema.items():
            parser = instantiate(
                PARSER_DICT, parser_cfg, alt_name="parser", dtype=dtype
            )
            self.parsers[data_product] = parser

            for key in parser.tree_keys:
                if key not in tree_keys:
                    tree_keys.append(key)

        # Initialize the backend reader with the collected tree keys
        self.reader = LArCVReader(tree_keys=tree_keys, **kwargs)

        # Initialize the augmenter
        self.build_augmenter(augment)

    def __len__(self) -> int:
        """Return the number of entries exposed by the LArCV reader."""
        return len(self.reader)

    def __getitem__(self, idx: int) -> DataDict:
        """Return one parsed LArCV entry.

        Parameters
        ----------
        idx : int
            Dataset entry index after any reader-level filtering.

        Returns
        -------
        dict
            Standard metadata fields plus one parser output for each product
            declared in ``schema``.
        """
        data_dict = self.reader[idx]
        result = self.metadata_dict(data_dict)

        for name, parser in self.parsers.items():
            try:
                result[name] = parser(data_dict)
            except Exception as err:
                logger.error("Failed to produce %s using %s", name, parser)
                raise err

        return self.apply_augmenter(result)

    @property
    def data_types(self) -> dict[str, str]:
        """Return the collate type for metadata and parsed products.

        Parser return types are consumed by :class:`spine.io.collate.CollateAll`
        to batch products consistently.
        """
        data_types = self.index_data_types()
        for name, parser in self.parsers.items():
            data_types[name] = parser.returns

        return data_types

    @property
    def overlay_methods(self) -> dict[str, str]:
        """Return overlay methods for metadata and parsed products.

        Parser overlay metadata is consumed by :class:`spine.io.overlay.Overlayer`
        when multiple entries are combined into one training sample.
        """
        overlay_methods = self.index_overlay_methods()
        for name, parser in self.parsers.items():
            overlay_methods[name] = parser.overlay

        return overlay_methods

    @property
    def data_keys(self) -> tuple[str, ...]:
        """Return metadata and parser-product keys exposed by this dataset."""
        return (*self._index_keys, *self._source_keys, *self.parsers.keys())

    @staticmethod
    def list_data(file_path: str) -> dict[str, list[str]]:
        """List top-level products available in an input LArCV file.

        Parameters
        ----------
        file_path : str
            Path to one LArCV input file.

        Returns
        -------
        dict
            Mapping from product category to available product names.
        """
        return LArCVReader.list_data(file_path)
