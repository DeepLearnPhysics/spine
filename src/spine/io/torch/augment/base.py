"""Base interfaces for data augmentation modules."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from spine.data import Meta


class AugmentBase(ABC):
    """Base class for augmentation modules."""

    name = ""

    def __call__(
        self,
        data: Dict[str, Any],
        meta: Meta,
        keys: List[str],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Meta]:
        """Apply an augmentation module.

        Parameters
        ----------
        data : dict
            Dictionary of event data products to augment
        meta : Meta
            Shared image metadata
        keys : List[str]
            Keys corresponding to data products that carry coordinates
        context : dict
            Shared augmentation context built by the manager

        Returns
        -------
        Tuple[Dict[str, Any], Meta]
            Updated data dictionary and shared metadata
        """
        return self.apply(data, meta, keys, context)

    @abstractmethod
    def apply(
        self,
        data: Dict[str, Any],
        meta: Meta,
        keys: List[str],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Meta]:
        """Apply an augmentation to one event.

        Parameters
        ----------
        data : dict
            Dictionary of event data products to augment
        meta : Meta
            Shared image metadata
        keys : List[str]
            Keys corresponding to data products that carry coordinates
        context : dict
            Shared augmentation context built by the manager

        Returns
        -------
        Tuple[Dict[str, Any], Meta]
            Updated data dictionary and shared metadata
        """
