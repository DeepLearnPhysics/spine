"""Deprecated compatibility shim for legacy enum imports."""

from warnings import warn

warn(
    "`spine.utils.enums` is deprecated. Import enums from "
    "`spine.constants.enums` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from spine.constants import ClusterLabelCol as ClusterLabelEnum  # noqa: F401
from spine.constants import NuCurrentType as NuCurrentTypeEnum
from spine.constants import ParticlePID as PIDEnum
from spine.constants import ParticleShape as ParticleShapeEnum
from spine.constants.factory import enum_factory

__all__ = [
    "ClusterLabelEnum",
    "ParticleShapeEnum",
    "PIDEnum",
    "NuCurrentTypeEnum",
    "enum_factory",
]
