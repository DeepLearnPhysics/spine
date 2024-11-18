"""Base class for all data representation builders."""

from abc import ABC, abstractmethod

import numpy as np

from spine.data import ObjectList


class BuilderBase(ABC):
    """Abstract base class for building all data structures

    A Builder class takes input data and full chain result dictionaries
    and processes them into human-readable data structures.
    """

    # Builder name
    name = None

    # Types of objects constructed by the builder
    _reco_type = None
    _truth_type = None

    # Necessary/optional data products to build a reconstructed object
    _build_reco_keys  = (
            ('points', True), ('depositions', True), ('sources', False)
    )

    # Necessary/optional data products to build a truth object
    _build_truth_keys = (
            ('label_tensor', True), ('label_adapt_tensor', True),
            ('label_g4_tensor', False), ('points', True),
            ('points_label', True), ('points_g4', False), ('depositions', True),
            ('depositions_label', True), ('depositions_q_label', False),
            ('depositions_g4', False), ('sources', False),
            ('sources_label', False)
    )

    # Necessary/optional data products to load a reconstructed object
    _load_reco_keys   = (
            ('points', True), ('depositions', True), ('sources', False)
    )

    # Necessary/optional data products to load a truth object
    _load_truth_keys  = (
            ('points', True), ('points_label', True), ('points_g4', False),
            ('depositions', True), ('depositions_label', True),
            ('depositions_q_label', False), ('depositions_g4', False),
            ('sources', False), ('sources_label', False)
    )

    def __init__(self, mode, units):
        """Initializes the builder.

        Parameters
        ----------
        mode : str, default 'both'
            Whether to construct reconstructed objects, true objects or both
            (one of 'reco', 'truth', 'both' or 'all')
        units : str, default 'cm'
            Units in which the position arguments of the constructed objects
            should be expressed (one of 'cm' or 'px')
        """
        # Store the mode and units
        self.mode = mode
        self.units = units

    def __call__(self, data):
        """Build representations for a batch of data.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Dispatch
        for mode, avoid in [('reco', 'truth'), ('truth', 'reco')]:
            out_key = f'{mode}_{self.name}s'
            if self.mode != avoid:
                if np.isscalar(data['index']):
                    # Single entry to process
                    data[out_key] = self.process(data, mode)

                else:
                    # Batch of data to process
                    const_list = []
                    for entry in range(len(data['index'])):
                        const_list.append(self.process(data, mode, entry))
                    data[out_key] = const_list

    def process(self, data, mode, entry=None):
        """Build representations for a single entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        mode : str
            Type of object to reconstruct ('reco' or 'truth')
        entry : int, optional
            Entry to process
        """
        # Dispatch to the appropriate function
        key = f'{mode}_{self.name}s'
        if key in data:
            func = f'load_{mode}'
        else:
            func = f'build_{mode}'

        result = self.construct(func, data, entry)

        # When loading, check that the units are as expected
        if 'load' in func:
            self.check_units(data, key, entry)

        return result

    def check_units(self, data, key, entry=None):
        """Checks that the objects in the list are expressed in the
        appropriate units. Convert them otherwise.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        key : str
            Dictionary key corresponding to the objects to convert
        entry : int, optional
            Entry to process
        """
        for obj in data[key]:
            if obj.units != self.units:
                assert 'meta' in data, (
                        "Cannot convert units without metadata information.")
                meta = data['meta'][entry] if entry is not None else data['meta']
                getattr(obj, f'to_{self.units}')(meta)

    def construct(self, func, data, entry=None):
        """Prepares the input based on the required data and runs constructor.

        Parameters
        ----------
        func : str
            Build function name
        data : dict
            Dictionary of data products
        entry : int, optional
            Entry to process

        Returns
        -------
        List[object]
            List of constructed objects
        """
        # Get the description of the fields needed by this source object
        input_data = {}
        method, dtype = func.split('_')
        keys = getattr(self, f'_{func}_keys')
        for key, req in keys:
            # If the field has no default value, must be provided
            if req and key not in data:
                raise KeyError(
                        f"Must provide `{key}` data product to {method} the "
                        f"{dtype} {self.name}s.")

            if key in data:
                if entry is not None:
                    input_data[key] = data[key][entry]
                else:
                    input_data[key] = data[key]

        obj_list = getattr(self, func)(input_data)
        default = getattr(self, f'_{dtype}_type')()

        return ObjectList(obj_list, default)

    @abstractmethod
    def build_reco(self, data):
        """Place-holder for a method used to build reconstructed objects.

        Parameters
        ----------
        data : dict
            Dictionary which contains the necessary data products
        """
        raise NotImplementedError

    @abstractmethod
    def build_truth(self, data):
        """Place-holder for a method used to build truth objects.

        Parameters
        ----------
        data : dict
            Dictionary which contains the necessary data products
        """
        raise NotImplementedError

    @abstractmethod
    def load_reco(self, data):
        """Place-holder for a method used to load reconstructed objects.

        Parameters
        ----------
        data : dict
            Dictionary which contains the necessary data products
        """
        raise NotImplementedError

    @abstractmethod
    def load_truth(self, data):
        """Place-holder for a method used to load truth objects.

        Parameters
        ----------
        data : dict
            Dictionary which contains the necessary data products
        """
        raise NotImplementedError
