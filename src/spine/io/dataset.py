"""Contains dataset classes to be used by the model."""

from spine.utils.conditional import TORCH_AVAILABLE
from spine.utils.factory import instantiate, module_dict
from spine.utils.logger import logger

from . import parse
from .augment import AugmentManager
from .read import HDF5Reader, LArCVReader

if TORCH_AVAILABLE:
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Import-safe stand-in used when PyTorch is unavailable."""

        pass


PARSER_DICT = module_dict(parse)

__all__ = ["LArCVDataset", "HDF5Dataset"]


class LArCVDataset(Dataset):
    """A generic interface for LArCV data files.

    This Dataset is designed to produce a batch of arbitrary number of data
    chunks (e.g. input data matrix, segmentation label, point proposal target,
    clustering labels, etc.). Each data chunk is processed by parser functions
    defined in the io.parsers module. LArCVDataset object can be
    configured with arbitrary number of parser functions where each function
    can take arbitrary number of LArCV event data objects. The assumption is
    that each data chunk respects the LArCV event boundary.

    This class utilizes the :class:`LArCVReader` class. It uses it to
    load data and to push it through the parsers.
    """

    name = "larcv"

    # List of index keys produced with each entry
    _index_keys = ("index", "file_index", "file_entry_index")

    def __init__(self, schema, dtype, augment=None, **kwargs):
        """Instantiates the LArCVDataset.

        Parameters
        ----------
        schema : dict
            A dictionary of (string, dictionary) pairs. The key is a unique
            name of a data chunk in a batch and the associated dictionary
            must include:
              - parser: name of the parser
              - kwargs: (key, value) pairs that correspond to parser argument
                names and their values
        dtype : str
            Data type to cast the input data to (to match the downstream model)
        augment : dict, optional
            Augmentation strategy configuration
        **kwargs : dict, optional
            Additional arguments to pass to the LArCVReader class
        """
        # Loop over parsers
        self.parsers = {}
        tree_keys = []
        for data_product, parser_cfg in schema.items():
            # Instantiate parser
            self.parsers[data_product] = instantiate(
                PARSER_DICT, parser_cfg, alt_name="parser", dtype=dtype
            )

            # Append to the list of trees to load
            for key in self.parsers[data_product].tree_keys:
                if key not in tree_keys:
                    tree_keys.append(key)

        # Parse the augmentation configuration
        self.augmenter = None
        if augment is not None:
            self.augmenter = AugmentManager(**augment)

        # Instantiate the reader
        self.reader = LArCVReader(tree_keys=tree_keys, **kwargs)

    def __len__(self):
        """Returns the lenght of the dataset (in number of batches).

        Returns
        -------
        int
            Number of entries in the dataset
        """
        return len(self.reader)

    def __getitem__(self, idx):
        """Returns one element of the dataset.

        Parameters
        ----------
        idx : int
            Index of the dataset entry to load

        Returns
        -------
        dict
            Dictionary of data product names and their associated data
        """
        # Read in a specific entry
        data_dict = self.reader[idx]

        # Get the indexes
        entry_idx = self.reader.entry_index[idx]
        file_idx = self.reader.get_file_index(idx)
        file_entry_idx = self.reader.get_file_entry_index(idx)
        result = {
            "index": entry_idx,
            "file_index": file_idx,
            "file_entry_index": file_entry_idx,
        }

        # Loop over data products, execute parsers
        for name, parser in self.parsers.items():
            try:
                result[name] = parser(data_dict)
            except Exception as err:
                logger.error(f"Failed to produce {name} using {parser}")
                raise err

        # If requested, augment the data
        if self.augmenter is not None:
            result = self.augmenter(result)

        return result

    @property
    def data_types(self):
        """Returns the data type returned by each parser.

        Returns
        -------
        Dict[str, str]
            Dictionary of data types
        """
        data_types = {key: "scalar" for key in self._index_keys}
        for name, parser in self.parsers.items():
            data_types[name] = parser.returns

        return data_types

    @property
    def overlay_methods(self):
        """Returns a dictionary mapping data products to overlay methods.

        Returns
        -------
        Dict[str, str]
            Dictionary of overlay methods
        """
        overlay_methods = {key: "cat" for key in self._index_keys}
        for name, parser in self.parsers.items():
            overlay_methods[name] = parser.overlay

        return overlay_methods

    @property
    def data_keys(self):
        """Returns a list of data product names.

        Returns
        -------
        List[str]
            List of data product names
        """
        return (*self._index_keys, *self.parsers.keys())

    @staticmethod
    def list_data(file_path):
        """Dumps top-level information about the contents of the LArCV root
        file.

        Parameters
        ----------
        file_path : str
            Path to the file to scan

        Returns
        -------
        dict
            Dictionary which maps data types onto a list of keys
        """
        return LArCVReader.list_data(file_path)


class HDF5Dataset(Dataset):
    """Thin PyTorch dataset wrapper around :class:`HDF5Reader`.

    This dataset serves cached event-level data products directly from SPINE
    HDF5 files. Unlike :class:`LArCVDataset`, it does not parse source-format
    objects; it expects the HDF5 file to already contain products in the form
    consumed by the collate layer or by downstream code.
    """

    name = "hdf5"

    # List of index keys produced with each entry
    _index_keys = ("index", "file_index", "file_entry_index")

    def __init__(
        self,
        dtype=None,
        keys=None,
        skip_keys=None,
        data_types=None,
        overlay_methods=None,
        augment=None,
        **kwargs,
    ):
        """Instantiate the HDF5-backed dataset.

        Parameters
        ----------
        dtype : str, optional
            Accepted for factory compatibility. HDF5 products are returned in
            the dtype stored on disk.
        keys : List[str], optional
            Data product keys to keep. Index keys are always retained.
        skip_keys : List[str], optional
            Data product keys to drop after reading.
        data_types : Dict[str, str], optional
            Collate type for each HDF5 product. Defaults to ``scalar`` for
            index keys and ``list`` for cached products.
        overlay_methods : Dict[str, str], optional
            Overlay method for each HDF5 product.
        augment : dict, optional
            Augmentation strategy configuration.
        **kwargs : dict, optional
            Additional arguments passed to :class:`HDF5Reader`.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use HDF5Dataset.")
        if keys is not None and skip_keys is not None:
            raise ValueError("Provide either `keys` or `skip_keys`, not both.")

        self.keys = set(keys) if keys is not None else None
        self.skip_keys = set(skip_keys) if skip_keys is not None else set()
        self._data_types = data_types
        self._overlay_methods = overlay_methods

        self.augmenter = None
        if augment is not None:
            self.augmenter = AugmentManager(**augment)

        self.reader = HDF5Reader(**kwargs)

    def __len__(self):
        """Returns the number of entries in the dataset."""
        return len(self.reader)

    def __getitem__(self, idx):
        """Returns one cached entry."""
        result = self.reader[idx]
        if self.keys is not None:
            keep = self.keys.union(self._index_keys)
            result = {key: val for key, val in result.items() if key in keep}
        for key in self.skip_keys:
            result.pop(key, None)

        if self.augmenter is not None:
            result = self.augmenter(result)

        return result

    @property
    def data_types(self):
        """Returns the collate type for each data product."""
        data_types = {key: "scalar" for key in self._index_keys}
        if self._data_types is not None:
            data_types.update(self._data_types)
        else:
            sample = self[0] if len(self) else {}
            for key in sample:
                if key not in data_types:
                    data_types[key] = "list"
        return data_types

    @property
    def overlay_methods(self):
        """Returns the overlay method for each data product."""
        overlay_methods = {key: "cat" for key in self._index_keys}
        if self._overlay_methods is not None:
            overlay_methods.update(self._overlay_methods)
        return overlay_methods

    @property
    def data_keys(self):
        """Returns the list of data product names."""
        return tuple(self.data_types.keys())
