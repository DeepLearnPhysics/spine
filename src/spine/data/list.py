"""Module with a class object which represent object lists."""

__all__ = ["ObjectList"]


class ObjectList(list):
    """List with a default object used to type it when it is empty.

    Attributes
    ----------
    default : object
        Default object class to use to type the list, if it is empty
    """

    def __init__(self, object_list, default):
        """Initialize the list and the default value.

        Parameters
        ----------
        object_list : List[object]
            Object list
        default : object
            Default object class to use to type the list, if it is empty
        """
        # Initialize the underlying list
        super().__init__(object_list)

        # Store the default object class
        self.default = default
