"""Deprecated compatibility shim for legacy global constants imports."""

from warnings import warn

warn(
    "`spine.utils.globals` is deprecated. Import constants from "
    "`spine.constants` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from spine.constants import *  # noqa: F401,F403
