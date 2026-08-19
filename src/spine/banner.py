"""SPINE command-line identity banner."""

from .version import __version__

__all__ = ["ASCII_LOGO", "BANNER_SEPARATOR", "format_banner"]


BANNER_SEPARATOR = "=" * 60

ASCII_LOGO = (
    " ██████████   ██████████    ███   ███       ██   ███████████\n"
    "███        █  ██       ███   █    █████     ██   ██         \n"
    "  ████████    ██       ███  ███   ██  ████  ██   ██████████ \n"
    "█        ███  ██████████     █    ██     █████   ██         \n"
    " ██████████   ██            ███   ██       ███   ███████████\n"
)


def format_banner(version: str = __version__) -> str:
    """Build the complete identity banner shown by the SPINE CLI.

    Parameters
    ----------
    version : str, default ``spine.__version__``
        Package version to display.

    Returns
    -------
    str
        Multi-line project identity terminated by a strong separator.
    """
    return (
        f"\n{ASCII_LOGO}\n"
        f"SPINE {version}\n"
        "Scalable Particle Imaging with Neural Embeddings\n"
        "DeepLearnPhysics Collaboration\n"
        "https://github.com/DeepLearnPhysics/spine\n\n"
        f"{BANNER_SEPARATOR}\n"
    )
