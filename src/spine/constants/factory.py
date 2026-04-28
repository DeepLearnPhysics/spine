"""Helpers for parsing SPINE enum-like configuration values.

This module is intentionally separate from :mod:`spine.constants.enums`.
The enum module should define canonical enumerated types; this module contains
parsing helpers that translate user/config strings into those enum values.
"""

from spine.constants.columns import ClusterLabelCol
from spine.constants.enums import NuInteractionScheme, ParticlePID, ParticleShape

__all__ = ["enum_factory"]


def enum_factory(enum: str, value):
    """Parse canonical SPINE enum values from config strings.

    This is a small compatibility/helper layer used by config-driven code that
    still refers to groups like ``"shape"`` or ``"pid"`` rather than importing
    enum classes directly.

    Parameters
    ----------
    enum : str
        Name of the supported enum group to parse. Supported groups are
        ``"cluster"``, ``"shape"``, ``"pid"``, and
        ``"interaction_scheme"``.
    value : Union[str, Sequence[str]]
        One enum-member name or a sequence of enum-member names. Member names
        are matched case-insensitively through their uppercase representation.

    Returns
    -------
    Union[int, List[int]]
        Integer value of the parsed enum member, or a list of integer values
        if a sequence of names is provided.

    Raises
    ------
    AssertionError
        If the requested enum group is not supported.
    ValueError
        If one of the provided member names does not exist on the target enum.
    """
    enum_dict = {
        "cluster": ClusterLabelCol,
        "shape": ParticleShape,
        "pid": ParticlePID,
        "interaction_scheme": NuInteractionScheme,
    }
    assert enum in enum_dict, (
        f"Enumerated type not recognized: {enum}. "
        f"Must be one of {list(enum_dict.keys())}."
    )
    enum_type = enum_dict[enum]

    def parse_one(name: str) -> int:
        member_name = name.upper()
        if not hasattr(enum_type, member_name):
            raise ValueError(
                f"Enumerated object not recognized: {name}. "
                f"Must be one of {[e.name for e in enum_type]}."
            )
        return getattr(enum_type, member_name).value

    if isinstance(value, str):
        return parse_one(value)

    return [parse_one(v) for v in value]
