"""Lazy access to optional third-party dependencies.

The symbols exported here preserve the previous
``from spine.utils.conditional import torch`` style while avoiding eager imports
and unrelated warnings. Availability flags are cheap import-spec checks. The
actual package import is deferred until an attribute on the proxy is used.
"""

import importlib
import importlib.util
import os
import sys
from typing import TYPE_CHECKING, Any

__all__ = [
    "ROOT",
    "larcv",
    "torch",
    "ME",
    "MF",
    "ROOT_AVAILABLE",
    "LARCV_AVAILABLE",
    "TORCH_AVAILABLE",
    "ME_AVAILABLE",
]


class _MissingType:
    """Placeholder type used for annotations and simple availability checks."""


class _MissingNamespace:
    """Annotation-safe namespace for unavailable optional modules.

    This allows type-hint evaluation of chained names such as
    ``torch.utils.data.Dataset`` without importing the real dependency. Any
    apparent runtime use still raises the original import error when called.
    """

    def __init__(self, error: str) -> None:
        self._error = error

    def __getattr__(self, name: str) -> Any:
        if name and name[0].isupper():
            return _MissingType
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise ImportError(self._error)


class _LazyModule:
    """Proxy that imports an optional module on first real use."""

    def __init__(self, import_name: str, display_name: str, error: str) -> None:
        self._import_name = import_name
        self._display_name = display_name
        self._error = error
        self._module = None

    def _load(self) -> Any:
        if self._module is None:
            try:
                self._module = importlib.import_module(self._import_name)
            except ModuleNotFoundError as exc:
                raise ImportError(self._error) from exc
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        if self._module is None:
            return f"<lazy module {self._display_name}>"
        return repr(self._module)


class _LazyAttribute:
    """Proxy for an attribute imported from an optional module."""

    def __init__(
        self, module_name: str, attr_name: str, display_name: str, error: str
    ) -> None:
        self._module_name = module_name
        self._attr_name = attr_name
        self._display_name = display_name
        self._error = error
        self._attr = None

    def _load(self) -> Any:
        if self._attr is None:
            try:
                module = importlib.import_module(self._module_name)
            except ModuleNotFoundError as exc:
                raise ImportError(self._error) from exc
            self._attr = getattr(module, self._attr_name)
        return self._attr

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._load()(*args, **kwargs)

    def __repr__(self) -> str:
        if self._attr is None:
            return f"<lazy attribute {self._display_name}>"
        return repr(self._attr)


class _MissingTorch(_LazyModule):
    """Torch proxy with a Tensor placeholder when torch is unavailable."""

    Tensor = _MissingType
    BoolTensor = _MissingType
    LongTensor = _MissingType
    FloatTensor = _MissingType

    def __getattr__(self, name: str) -> Any:
        if name and name[0].isupper():
            return _MissingType
        return _MissingNamespace(self._error)


class _MissingME(_LazyModule):
    """MinkowskiEngine proxy with a SparseTensor placeholder when unavailable."""

    SparseTensor = _MissingType

    def __getattr__(self, name: str) -> Any:
        if name and name[0].isupper():
            return _MissingType
        return _MissingNamespace(self._error)


def _module_available(module_name: str) -> bool:
    if os.getenv("SPINE_DOC_BUILD") and module_name in {
        "torch",
        "MinkowskiEngine",
        "MinkowskiFunctional",
        "larcv",
        "ROOT",
    }:
        return False

    if module_name in sys.modules:
        module = sys.modules[module_name]
        if module is None:
            return False

        # Sphinx autodoc mock imports can populate `sys.modules` with
        # lightweight placeholders that do not correspond to a real importable
        # package. Only treat preloaded modules as available when they carry a
        # real module spec.
        if getattr(module, "__spec__", None) is not None:
            return True

    try:
        if importlib.util.find_spec(module_name) is not None:
            return True
    except (ImportError, ValueError):
        pass

    # PyROOT/LArCV can be importable even when importlib metadata discovery
    # fails in some HPC/container/module environments. For these modules only,
    # fall back to an actual import probe before declaring them unavailable.
    if module_name not in {"ROOT", "larcv"}:
        return False

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False

    if module_name == "larcv":
        try:
            getattr(module, "larcv")
        except AttributeError:
            return False

    return True


ROOT_AVAILABLE = _module_available("ROOT")
LARCV_AVAILABLE = _module_available("larcv")
TORCH_AVAILABLE = _module_available("torch")
ME_AVAILABLE = _module_available("MinkowskiEngine")

if TYPE_CHECKING:
    import MinkowskiEngine as ME
    import MinkowskiFunctional as MF
    import ROOT
    import torch
    from larcv import larcv
else:
    ROOT = _LazyModule("ROOT", "ROOT", "ROOT is required to parse LArCV data.")
    larcv = _LazyAttribute(
        "larcv", "larcv", "larcv.larcv", "larcv is required to parse LArCV data."
    )

    if TORCH_AVAILABLE:
        torch = _LazyModule("torch", "torch", "PyTorch is required.")
    else:
        torch = _MissingTorch("torch", "torch", "PyTorch is required.")

    if ME_AVAILABLE:
        ME = _LazyModule(
            "MinkowskiEngine", "MinkowskiEngine", "MinkowskiEngine is required."
        )
        MF = _LazyModule(
            "MinkowskiFunctional",
            "MinkowskiFunctional",
            "MinkowskiFunctional is required.",
        )
    else:
        ME = _MissingME(
            "MinkowskiEngine", "MinkowskiEngine", "MinkowskiEngine is required."
        )
        MF = _LazyModule(
            "MinkowskiFunctional",
            "MinkowskiFunctional",
            "MinkowskiFunctional is required.",
        )
