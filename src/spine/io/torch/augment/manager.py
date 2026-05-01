"""Augmentation manager."""

from typing import Any, Dict

from spine.data import Meta
from spine.io.core.parse.data import ParserTensor

from .crop import CropAugment
from .flip import FlipAugment
from .jitter import JitterAugment
from .mask import MaskAugment
from .rotate import RotateAugment
from .translate import TranslateAugment


class AugmentManager:
    """Generic class to handle ordered data augmentation modules."""

    _modules = {
        "mask": MaskAugment,
        "crop": CropAugment,
        "jitter": JitterAugment,
        "flip": FlipAugment,
        "rotate": RotateAugment,
        "translate": TranslateAugment,
    }

    def __init__(self, **augmenters: Dict[str, Any]) -> None:
        """Initialize the augmentation manager.

        Parameters
        ----------
        **augmenters : dict, optional
            Ordered dictionary of augmentation module configurations.
            If the configuration key matches a registered augmentation
            name (e.g. `crop`, `jitter`, `mask`, `rotate`, `translate`), the
            `name` entry can be omitted. If using a custom label to
            instantiate multiple augmenters of the same type, specify
            the module type explicitly through `name`.

        Returns
        -------
        None
            This method does not return anything
        """
        if not augmenters:
            raise ValueError("Must provide at least one augmentation module.")

        self.modules = []
        for key, cfg in augmenters.items():
            if cfg is None:
                continue
            if not isinstance(cfg, dict):
                raise ValueError(
                    f"Augmentation configuration for `{key}` must be a dictionary."
                )

            config = dict(cfg)
            name = config.pop("name", key)
            if name not in self._modules:
                raise ValueError(
                    f"Augmentation module not recognized: {name}. "
                    f"Must be one of {tuple(self._modules)}."
                )

            self.modules.append(self._modules[name](**config))

        if not self.modules:
            raise ValueError("Must enable at least one augmentation module.")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the data products in one event.

        Parameters
        ----------
        data : dict
            Dictionary of event data products

        Returns
        -------
        dict
            Updated dictionary of augmented data products
        """
        augment_keys = []
        meta = None
        for key, value in data.items():
            if isinstance(value, ParserTensor) and value.coords is not None:
                augment_keys.append(key)
                if meta is None:
                    meta = value.meta
                elif meta != value.meta:
                    raise ValueError("Metadata should be shared by all data products.")

            elif isinstance(value, Meta):
                augment_keys.append(key)
                meta = value

        if meta is None:
            return data

        context = {"original_meta": self.copy_meta(meta)}
        for module in self.modules:
            data, meta = module(data, meta, augment_keys, context)
            context["meta"] = meta

        return data

    @staticmethod
    def copy_meta(meta: Meta) -> Meta:
        """Return a detached copy of the metadata.

        Parameters
        ----------
        meta : Meta
            Metadata object to copy

        Returns
        -------
        Meta
            Detached metadata copy
        """
        return Meta(
            lower=meta.lower.copy(),
            upper=meta.upper.copy(),
            size=meta.size.copy(),
            count=meta.count.copy(),
        )
