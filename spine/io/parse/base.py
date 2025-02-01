"""Contains Parser class which all parsers inherit from."""

from abc import ABC, abstractmethod


class ParserBase(ABC):
    """Abstract parent class of all parser classes.

    Provides basic functionality shared by all parsers:
    1. Defines a :meth:`__call__` function shared by all classes

    Attributes
    ----------
    name : str
        Name of the parser
    aliases : List[str]
        Aliases of the parser (allowed but disfavored names)
    data_map : dict[str, str]
        Maps function parameter names onto a file data product name
    tree_keys : List[str]
        List of file data product name
    """

    # Name of the parser (as specified in the configuration)
    name = None

    # Alternative allowed names of the parser
    aliases = ()

    def __init__(self, dtype, **kwargs):
        """Loops over data product names, stores them.

        Parameters
        ----------
        dtype : str
            Data type to cast the input data to
        **kwargs : dict, optional
            Keyword arguments passed to the parser function

        Notes
        -----
        All parser argument which correspond to the name of a tree in the
        LArCV file must be contain either the `_event` or `_event_list` suffix.
        """
        # Store the type in which the parsers should return their data
        self.ftype = dtype
        self.itype = dtype.replace('float', 'int')

        # Find data keys, append them to the map
        self.data_map = {}
        self.tree_keys = []
        for key, value in kwargs.items():
            if '_event' not in key:
                class_name = self.__class__.__name__
                raise TypeError(
                        f"{class_name} got an unexpected argument: {key}.")

            if value is not None:
                self.data_map[key] = value
                if not isinstance(value, list):
                    if value not in self.tree_keys:
                        self.tree_keys.append(value)

                else:
                    for v in value:
                        if v not in self.tree_keys:
                            self.tree_keys.append(v)

    def get_input_data(self, trees):
        """Fetches the required data products from the LArCV data trees, pass
        them to the parser function.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object

        Results
        -------
        object
            Output(s) of the parser function
        """
        # Build the input to the parser function
        data_dict = {}
        for key, value in self.data_map.items():
            if isinstance(value, str):
                if value not in trees:
                    raise ValueError(
                            f"Must provide {value} for parser `{self.name}`.")
                data_dict[key] = trees[value]

            elif isinstance(value, list):
                for v in value:
                    if v not in trees:
                        raise ValueError(
                                f"Must provide {v} for parser `{self.name}`.")
                data_dict[key] = [trees[v] for v in value]

        return data_dict

    @abstractmethod
    def __call__(self, trees):
        """Parse one entry.

        This is a place-holder, must be defined in inheriting class.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        raise NotImplementedError("Must define `__call__` method.")
