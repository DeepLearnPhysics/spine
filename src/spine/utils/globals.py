"""Deprecated compatibility shim for legacy global constants imports.

This module preserves the historical ``spine.utils.globals`` import path while
the codebase and downstream users migrate to :mod:`spine.constants`.

Importing this module emits a :class:`DeprecationWarning` and re-exports the
public constants package surface so older code can continue to resolve symbols
such as shape IDs, PID IDs, column selectors, and material constants.

New code should import directly from :mod:`spine.constants` or one of its
focused submodules such as :mod:`spine.constants.enums` or
:mod:`spine.constants.columns`.
"""

from warnings import warn

warn(
    "`spine.utils.globals` is deprecated. Import constants from "
    "`spine.constants` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from spine.constants import *  # noqa: F401,F403
