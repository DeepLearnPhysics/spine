"""Dataset wrapper around :class:`spine.io.read.LArCVReader`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from spine.utils.factory import instantiate, module_dict
from spine.utils.logger import logger

from .. import parse
from ..read import LArCVReader
from .base import BaseDataset, DataDict

PARSER_DICT = module_dict(parse)

__all__ = ["LArCVDataset"]


class LArCVDataset(BaseDataset):
    """Torch dataset that parses LArCV entries into SPINE parser products."""

    name: ClassVar[str] = "larcv"
    parsers: dict[str, Any]
    reader: LArCVReader

    def __init__(
        self,
        schema: Mapping[str, Mapping[str, Any]],
        dtype: str,
        augment: Mapping[str, Any] | None = None,
        geo: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Instantiate the LArCV-backed dataset."""
        super().__init__()

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

        self.build_augmenter(augment, geo=geo)
        self.reader = LArCVReader(tree_keys=tree_keys, **kwargs)

    def __len__(self) -> int:
        """Return the number of dataset entries."""
        return len(self.reader)

    def __getitem__(self, idx: int) -> DataDict:
        """Return one parsed dataset entry."""
        data_dict = self.reader[idx]

        entry_idx = self.reader.entry_index[idx]
        file_idx = self.reader.get_file_index(idx)
        file_entry_idx = self.reader.get_file_entry_index(idx)
        result: DataDict = {
            "index": entry_idx,
            "file_index": file_idx,
            "file_entry_index": file_entry_idx,
        }

        for name, parser in self.parsers.items():
            try:
                result[name] = parser(data_dict)
            except Exception as err:
                logger.error("Failed to produce %s using %s", name, parser)
                raise err

        return self.apply_augmenter(result)

    @property
    def data_types(self) -> dict[str, str]:
        """Return the collate type of each parsed data product."""
        data_types = self.index_data_types()
        for name, parser in self.parsers.items():
            data_types[name] = parser.returns

        return data_types

    @property
    def overlay_methods(self) -> dict[str, str]:
        """Return the overlay method of each parsed data product."""
        overlay_methods = self.index_overlay_methods()
        for name, parser in self.parsers.items():
            overlay_methods[name] = parser.overlay

        return overlay_methods

    @property
    def data_keys(self) -> tuple[str, ...]:
        """Return the names of all data products produced by this dataset."""
        return (*self._index_keys, *self.parsers.keys())

    @staticmethod
    def list_data(file_path: str) -> dict[str, list[str]]:
        """List top-level products available in an input LArCV file."""
        return LArCVReader.list_data(file_path)
