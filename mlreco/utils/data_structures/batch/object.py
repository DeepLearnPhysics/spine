"""Module with dataclass targeted at batched locally-defined objects."""

import numpy as np
from dataclasses import dataclass
from typing import Union, List

from mlreco.utils.decorators import inherit_docstring

from .base import BatchBase


@dataclass
@inherit_docstring(BatchBase)
class ObjectBatch(BatchBase):
    """Batched objects with the necessary methods to unwrap them.
    
    Attributes
    ----------
    default_object : object
        Instance of the object class the list is made of
    """
    default_object: object

    def __init__(self, data, default_object=None):
        """Initialize the attributes of the class.

        Parameters
        ----------
        data : Union[List[object], List[List[object]]]
            (O) List or list of list of objects to store
        default_object : object
            Default object instance. Only necessary to unwrap empty lists
        """
        # Turn the input into a numpy array of objects, store counts
        if isinstance(data[0], list):
            counts = [len(d) for d in data]
            data_array = np.concatenate(data, dtype=object)
        else:
            counts = np.ones(len(data), dtype=np.int64)
            data_array = np.empty(len(data), dtype=object)
            data_array[:] = data
            
        # Initialize the base class
        super().__init(data_array, is_list=True)

        # Get the number of entries from the counts
        batch_size = len(data)

        # Check that the type of objects in the list and the default match
        if len(data) and default_object is not None:
            assert type(data[0]) == type(default_object), (
                    "The default object and the data content do not match.")

        # Cast
        counts = self._as_long(counts)

        # Get the boundaries between entries in the batch
        edges = self.get_edges(counts)

        # Store the attributes
        self.data = data
        self.counts = counts
        self.edges = edges
        self.batch_size = batch_size
        self.default_object = default_object

    def __getitem__(self, batch_id):
        """Returns the subject of objects corresponding to one entry.

        Parameters
        ----------
        batch_id : int
            Entry index
        """
        # Make sure the batch_id is sensible
        if batch_id >= self.batch_size:
            raise IndexError(f"Index {batch_id} out of bound for a batch size "
                             f"of ({self.batch_size})")

        # If each entry is made up of a single object, nothing to do
        if self.default_object is None:
            return self.data[batch_id]

        # Otherwise, break up the original list into a subset for this entry
        lower, upper = self.edges[batch_id], self.edges[batch_id + 1]
        return self.data[lower:upper]

    @property
    def object_list(self):
        """Alias for the underlying object list stored.

        Returns
        -------
        List[object]
            Underlying object list
        """
        return self.data

    @property
    def batch_ids(self):
        """Returns the batch ID of each object in the list.

        Returns
        -------
        Union[np.ndarray]
            (I) Batch ID array, one per object in the list
        """
        return self._repeat(self._arange(self.batch_size), self.counts)

    def split(self):
        """Breaks up the object batch into its constituents.

        Returns
        -------
        Union[List[object], List[List[object]]]
            List of one or multiple object(s) per entry in the batch
        """
        # If each entry is made up of a single object, nothing to do
        if self.default_object is None:
            return self.data

        # Otherwise, break up the original list into one list per entry
        return self._split(self.data, self.splits)
