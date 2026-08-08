"""Canonical data exchange between full-chain stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spine.data import TensorBatch

from .point import PointBatch

__all__ = ["ChainState", "StageResult"]


@dataclass
class StageResult:
    """Products published by one full-chain stage.

    Parameters
    ----------
    products : dict, optional
        Canonical products made available to subsequent stages.
    outputs : dict, optional
        Public model outputs retained for losses, logging and downstream
        reconstruction. These do not implicitly become stage inputs.
    """

    products: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)


class ChainState:
    """Mutable execution state with explicit product ownership.

    The state separates canonical inter-stage products from public outputs.
    Stages may only replace products declared through their ``replaces``
    contract, preventing accidental overwrites from silently changing the
    remainder of the reconstruction chain.

    Point-level tensors are exchanged by native providers through the
    canonical ``point_data`` product. Flat ``data``, ``sources``, and
    ``orig_index`` aliases remain synchronized for external providers that
    still consume the historical interface.
    """

    def __init__(self, **products: Any) -> None:
        """Initialize the state from non-null driver inputs.

        Parameters
        ----------
        **products : object
            Canonical driver inputs. Null optional products are omitted.

        Notes
        -----
        If a tensor-valued ``data`` product is supplied without an explicit
        ``point_data`` product, an aligned :class:`PointBatch` is initialized
        automatically from ``data`` and any source/index products.
        """
        self.products = {
            key: value for key, value in products.items() if value is not None
        }
        data = self.products.get("data")
        if "point_data" not in self.products and isinstance(data, TensorBatch):
            self.products["point_data"] = PointBatch.from_input(
                data,
                self.products.get("sources"),
                self.products.get("orig_index"),
            )
        self.outputs: dict[str, Any] = {}
        self.producers = {key: "input" for key in self.products}

    def __contains__(self, key: str) -> bool:
        """Return whether a canonical product is available.

        Returns
        -------
        bool
            ``True`` when ``key`` has been provided or published.
        """
        return key in self.products

    def get(self, key: str, default: Any = None) -> Any:
        """Fetch an optional canonical product.

        Parameters
        ----------
        key : str
            Canonical product name.
        default : object, optional
            Value returned when the product is unavailable.

        Returns
        -------
        object
            Stored product or ``default``.
        """
        return self.products.get(key, default)

    def require(self, key: str, stage: str | None = None) -> Any:
        """Fetch a required canonical product.

        Parameters
        ----------
        key : str
            Canonical product name.
        stage : str, optional
            Consumer name included in a missing-product error.

        Returns
        -------
        object
            Requested canonical product.

        Raises
        ------
        KeyError
            If the product is unavailable.
        """
        try:
            return self.products[key]
        except KeyError as err:
            context = "" if stage is None else f" for stage `{stage}`"
            raise KeyError(f"Missing required chain product `{key}`{context}.") from err

    def publish(
        self,
        stage: str,
        result: StageResult,
        replaces: frozenset[str] = frozenset(),
    ) -> None:
        """Merge a stage result after enforcing ownership declarations.

        Parameters
        ----------
        stage : str
            Name of the producing stage.
        result : StageResult
            Canonical products and public outputs produced by the stage.
        replaces : frozenset of str, optional
            Existing canonical products this stage is authorized to replace.

        Raises
        ------
        TypeError
            If a stage publishes a non-:class:`PointBatch` value under the
            canonical ``point_data`` key.
        ValueError
            If a stage replaces an undeclared product or duplicates a public
            output name.
        """
        # Canonical products track ownership so replacement mistakes identify
        # both the original and attempted producers.
        for key, value in result.products.items():
            if key in self.products and key not in replaces:
                producer = self.producers[key]
                raise ValueError(
                    f"Stage `{stage}` attempted to replace product `{key}` from "
                    f"`{producer}` without declaring it."
                )
            self.products[key] = value
            self.producers[key] = stage

            # Keep the historical flat aliases synchronized for external
            # providers while native stages exchange one aligned point bundle.
            if key == "point_data":
                if not isinstance(value, PointBatch):
                    raise TypeError(
                        "The canonical `point_data` product must be PointBatch."
                    )
                aliases = value.canonical_products()
                for alias in ("data", "sources", "orig_index"):
                    if alias in aliases:
                        self.products[alias] = aliases[alias]
                        self.producers[alias] = stage
                    elif alias in self.products and alias != "data":
                        del self.products[alias]
                        self.producers.pop(alias, None)

        # Public outputs are append-only: losses and reconstruction consumers
        # must never depend on stage order to resolve duplicate names.
        overlap = set(self.outputs).intersection(result.outputs)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                f"Stage `{stage}` produced duplicate public output(s): {names}."
            )
        self.outputs.update(result.outputs)
