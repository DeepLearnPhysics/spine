"""Docstring inheritance utilities."""


def merge_ancestor_docstrings(cls):
    """Automatically merge Attributes sections from parent class docstrings.

    This function extracts the Attributes section from all direct parent
    classes and prepends them to the child class's Attributes section.
    This is designed to work with numpy-style docstrings.

    The function only looks at direct parent classes (cls.__bases__) rather
    than the full MRO to avoid duplicate attributes when multiple levels
    of inheritance are involved.

    Parameters
    ----------
    cls : type
        The class whose docstring should be updated with parent attributes

    Notes
    -----
    This function modifies the class's __doc__ attribute in-place. It is
    typically called from __init_subclass__ hooks in base classes to
    automatically merge docstrings for all subclasses.

    The function looks for the "Attributes" section in numpy-style docstrings,
    which is formatted as:

        Attributes
        ----------
        attribute_name : type
            Description

    Examples
    --------
    >>> class Parent:
    ...     '''Parent class.
    ...
    ...     Attributes
    ...     ----------
    ...     x : int
    ...         Parent attribute
    ...     '''
    ...     def __init_subclass__(cls, **kwargs):
    ...         super().__init_subclass__(**kwargs)
    ...         merge_ancestor_docstrings(cls)
    ...
    >>> class Child(Parent):
    ...     '''Child class.
    ...
    ...     Attributes
    ...     ----------
    ...     y : int
    ...         Child attribute
    ...     '''
    >>> # Child.__doc__ now contains both x and y in Attributes section
    """
    # Skip if the class has no docstring
    if cls.__doc__ is None:
        return

    # Numpy-style docstring format (no indentation on section headers)
    header = "Attributes\n----------\n"

    # Collect Attributes sections from DIRECT parent classes only
    # (indirect parents are already merged into direct parents)
    parent_attrs = []
    for base in cls.__bases__:  # Only direct parents, not full MRO
        if base is object:
            continue
        if not hasattr(base, "__doc__") or base.__doc__ is None:
            continue

        # Extract Attributes section from parent docstring
        if header not in base.__doc__:
            continue

        # Find the first occurrence of the header
        header_pos = base.__doc__.find(header)
        attr_start = header_pos + len(header)

        # Find where this section ends (next section or end of docstring)
        rest = base.__doc__[attr_start:]
        lines = rest.split("\n")
        attr_lines = []

        for line in lines:
            # Stop at next section (line with only dashes, no indentation)
            stripped = line.strip()
            if (
                stripped
                and stripped == "-" * len(stripped)
                and not line.startswith(" ")
            ):
                # This is a section divider, stop here
                break
            attr_lines.append(line)

        # Clean up and only keep non-empty content
        section_text = "\n".join(attr_lines).rstrip()

        if section_text.strip():  # Only add if there's actual content
            parent_attrs.append(section_text)

    # Now handle the child's docstring
    if header in cls.__doc__:
        # Split on FIRST occurrence only
        header_pos = cls.__doc__.find(header)
        before_header = cls.__doc__[:header_pos]
        after_header = cls.__doc__[header_pos + len(header) :]

        # Merge parent attributes before child attributes
        if parent_attrs:
            merged_parents = "\n".join(parent_attrs) + "\n"
            cls.__doc__ = before_header + header + merged_parents + after_header
    else:
        # No Attributes section in child, add one with just parent attrs
        if parent_attrs:
            merged_parents = "\n".join(parent_attrs)
            cls.__doc__ += f"\n\n{header}{merged_parents}\n"
