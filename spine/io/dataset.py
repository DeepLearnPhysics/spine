"""Contains dataset classes to be used by the model."""

from torch.utils.data import Dataset

from spine.utils.logger import logger
from spine.utils.factory import module_dict, instantiate
from spine.utils.augment import Augmenter

from . import parse
from .read import LArCVReader

PARSER_DICT  = module_dict(parse)

__all__ = ['LArCVDataset']


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
    name = 'larcv'

    def __init__(self, schema, dtype, augment=None, **kwargs):
        """Instantiates the LArCVDataset.

        Parameters
        ----------
        schema : dict
            A dictionary of (string, dictionary) pairs. The key is a unique
            name of a data chunk in a batch and the associated dictionary
            must include:
              - parser: name of the parser
              - args: (key, value) pairs that correspond to parser argument
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
                    PARSER_DICT, parser_cfg, alt_name='parser', dtype=dtype)

            # Append to the list of trees to load
            for key in self.parsers[data_product].tree_keys:
                if key not in tree_keys:
                    tree_keys.append(key)

        # Parse the augmentation configuration
        self.augmenter = None
        if augment is not None:
            self.augmenter = Augmenter(**augment)

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

        # Get the index
        entry_idx = self.reader.entry_index[idx]
        file_idx = self.reader.get_file_index(idx)
        file_entry_idx = self.reader.get_file_entry_index(idx)
        result = {'index': entry_idx, 'file_index': file_idx,
                  'file_entry_index': file_entry_idx}

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

    def data_keys(self):
        """Returns a list of data product names.

        Returns
        -------
        List[str]
            List of data product names
        """
        return list(self.parsers.keys()) + ['index']

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
