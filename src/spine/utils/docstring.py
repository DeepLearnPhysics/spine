"""Docstring inheritance utilities."""


def inherit_docstring(*parents):
    """Inherits docstring attributes of a parent class.

    Only handles numpy-style docstrings.

    Parameters
    ----------
    *parents : List[object]
        Parent class(es) to inherit attributes from

    Returns
    -------
    callable
        Class with updated docstring
    """

    def inherit(obj):
        tab = "    "
        underline = "----"
        header = f"Attributes\n{tab}----------\n"

        # If there is no attribute or method yet, add the header
        if not header in obj.__doc__:
            obj.__doc__ += f"\n\n{tab}{header}"

        # Get the parent attribute docstring block
        prestr = ""
        for parent in parents:
            docstr = parent.__doc__
            substr = docstr.split(header)[-1].rstrip() + "\n"
            if len(substr.split(underline)) > 1:
                substr = substr.split(underline)[0].split("\n")[:-1]
                substr = "".join(substr).rstrip()

            prestr += substr

        # Append it to the relevant block
        split_doc = obj.__doc__.split(header)
        obj.__doc__ = split_doc[0] + header + prestr + split_doc[1]

        return obj

    return inherit
