"""Optional PDG particle-name resolution."""

from functools import lru_cache
from operator import index
from typing import Any

from spine.utils.conditional import PARTICLE_AVAILABLE, particle

__all__ = ["pdg_name"]


def pdg_name(code: Any) -> str | None:
    """Resolve a PDG identifier to its canonical particle name.

    The canonical SPINE sentinel ``0`` resolves to ``"UNKNOWN"``. Missing
    dependencies and unrecognized identifiers return ``None`` so callers can
    preserve the raw numeric value.

    Parameters
    ----------
    code : Any
        Integer-like PDG identifier to resolve.

    Returns
    -------
    str or None
        Canonical particle name, ``"UNKNOWN"`` for the SPINE sentinel, or
        ``None`` when the identifier cannot be resolved.
    """
    try:
        code = index(code)
    except (TypeError, ValueError):
        return None

    if code == 0:
        return "UNKNOWN"

    if not PARTICLE_AVAILABLE:
        return None

    return _pdg_name(code)


@lru_cache(maxsize=1024)
def _pdg_name(code: int) -> str | None:
    """Perform and cache one validated Scikit-HEP particle lookup."""
    try:
        return particle.Particle.from_pdgid(code).name
    except (particle.InvalidParticle, particle.ParticleNotFound):
        return None
