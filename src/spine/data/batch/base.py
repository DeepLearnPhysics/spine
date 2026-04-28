"""Module with a base class for all batched data structures."""

from dataclasses import dataclass

import numpy as np

from spine.utils.conditional import ME, torch
from spine.utils.docstring import merge_ancestor_docstrings


@dataclass(eq=False)
class BatchBase:
    """Base class for all types of batched data.

    Attributes
    ----------
    data : Union[list, np.ndarray, torch.Tensor]
        Batched data
    counts : Union[np.ndarray, torch.Tensor]
        (B) Number of data elements in each entry of the batch
    edges : Union[np.ndarray, torch.Tensor]
        (B+1) Edges separating the entries in the batch
    batch_size : int
        Number of entries that make up the batched data
    """

    data: (
        np.ndarray
        | torch.Tensor
        | ME.SparseTensor
        | list[np.ndarray]
        | list[torch.Tensor]
    )
    counts: np.ndarray | torch.Tensor
    edges: np.ndarray | torch.Tensor
    batch_size: int

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Automatically merge docstrings from parent classes.

        This hook is called whenever a class inherits from BatchBase. It
        automatically merges the Attributes sections from all parent class
        docstrings into the child class docstring.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to super().__init_subclass__
        """
        super().__init_subclass__(**kwargs)
        merge_ancestor_docstrings(cls)

    def __init__(
        self,
        data: np.ndarray | torch.Tensor | ME.SparseTensor,
        is_sparse: bool = False,
        is_list: bool = False,
    ) -> None:
        """Shared initializations across all types of batched data.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor, ME.SparseTensor]
            Batched data
        is_sparse : bool, default False
            If initializing from an ME sparse data, flip to True
        is_list : bool, default False
            Whether the underlying data is a list of tensors
        """
        # Store the datatype
        self.is_numpy = not is_sparse and not isinstance(data, torch.Tensor)
        self.is_sparse = is_sparse
        self.is_list = is_list

        # Store the datatype
        self.dtype = data.dtype

        # Store the device
        self.device = None
        if not self.is_numpy:
            if isinstance(data, torch.Tensor):
                self.device = data.device
            else:
                self.device = data.F.device

    def __len__(self) -> int:
        """Returns the number of entries that make up the batch."""
        return self.batch_size

    def __eq__(self, other: object) -> bool:
        """Checks that all attributes of two class instances are the same.

        This overloads the default dataclass `__eq__` method to include an
        appopriate check for vector (numpy) attributes.

        Parameters
        ----------
        other : BatchBase
            Other instance of the same object class

        Returns
        -------
        bool
            `True` if all attributes of both objects are identical
        """
        # Check that the two objects belong to the same class
        if self.__class__ != other.__class__:
            return False

        # Check that all attributes are identical
        for k, v in self.__dict__.items():
            v_other = getattr(other, k)
            if v is None:
                # If not filled, make sure neither are
                if v_other is not None:
                    return False

            elif np.isscalar(v) or isinstance(v, np.dtype):
                if v_other != v:
                    return False

            else:
                # For vectors, use array_equal with equal_nan=True
                if not np.array_equal(v, v_other, equal_nan=True):
                    return False

        return True

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data.

        Returns
        -------
        tuple
            Tuple of sizes in each dimension
        """
        if not self.is_list:
            return self.data.shape
        else:
            return (len(self.data),)

    @property
    def splits(self):
        """Boundaries needed to split the data into its constituents.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            (B-1) One split per batch boundary
        """
        return self.edges[1:-1]

    def get_counts(self, batch_ids, batch_size):
        """Finds the number of elements in each entry, provided a batch ID list.

        Parameters
        ----------
        batch_ids : Union[np.ndarray, torch.Tensor]
            List of batch IDs
        batch_size : int
            Number of entries that make up the batched data

        Returns
        -------
        np.ndarray
            (B) Length of each entry
        """
        # Get the count list
        device = None if self.is_numpy else batch_ids.device
        counts = self._zeros(batch_size, device)
        if len(batch_ids) > 0:
            # Find the length of each batch ID in the input index
            uni, cnts = self._unique(batch_ids)
            counts[self._as_long(uni)] = cnts

        return counts

    def get_edges(self, counts):
        """Finds the edges between successive entries in the batch.

        Parameters
        ----------
        counts : Union[np.ndarray, torch.Tensor]
            (B)Length of each entry

        Returns
        -------
        np.ndarray
            (B+1) Edges of successive entries in the batch
        """
        # Get the edge list
        device = None if self.is_numpy else counts.device
        edges = self._zeros(len(counts) + 1, device)
        cumsum = self._cumsum(counts)
        edges[1:] = cumsum

        return edges

    def _empty(self, x):
        if self.is_numpy:
            return np.empty(x, dtype=np.int64)
        else:
            return torch.empty(x, dtype=torch.long, device=self.device)

    def _zeros(self, x, device=None):
        if self.is_numpy:
            return np.zeros(x, dtype=np.int64)
        else:
            return torch.zeros(x, dtype=torch.long, device=device)

    def _ones(self, x):
        if self.is_numpy:
            return np.ones(x, dtype=np.int64)
        else:
            return torch.ones(x, dtype=torch.long, device=self.device)

    def _as_long(self, x):
        if self.is_numpy:
            return np.asarray(x, dtype=np.int64)
        else:
            # Always on CPU. This is because splits are supposed to be on
            # CPU regardless of the location of the underlying data
            return torch.as_tensor(x, dtype=torch.long, device="cpu")

    def _unique(self, x):
        if self.is_numpy:
            return np.unique(x, return_counts=True)
        else:
            return torch.unique(x, return_counts=True)

    def _transpose(self, x):
        if self.is_numpy:
            return np.transpose(x)
        else:
            return torch.transpose(x, 0, 1)

    def _sum(self, x):
        if self.is_numpy:
            return np.sum(x)
        else:
            return torch.sum(x)

    def _cumsum(self, x):
        if self.is_numpy:
            return np.cumsum(x)
        else:
            return torch.cumsum(x, dim=0)

    def _arange(self, x):
        if self.is_numpy:
            return np.arange(x)
        else:
            return torch.arange(x, device=self.device)

    def _cat(self, x):
        if self.is_numpy:
            return np.concatenate(x)
        else:
            return torch.cat(x, dim=0)

    def _split(self, *x):
        if self.is_numpy:
            return np.split(*x)
        else:
            return torch.tensor_split(*x)

    def _stack(self, x):
        if self.is_numpy:
            return np.stack(x, axis=0)
        else:
            return torch.stack(x)

    def _repeat(self, *x):
        if self.is_numpy:
            return np.repeat(*x)
        else:
            return torch.repeat_interleave(*x)

    def _to_numpy(self, x):
        return x.cpu().detach().numpy()

    def _to_tensor(self, x, dtype=None, device=None):
        return torch.as_tensor(x, dtype=dtype, device=device)
