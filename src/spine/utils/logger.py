"""SPINE logging setup and rank-aware output filtering."""

import logging
import sys
import warnings


class MainProcessFilter(logging.Filter):
    """Suppress low-priority log records from non-main distributed ranks."""

    def __init__(self, rank: int | None = None) -> None:
        """Initialize the rank filter.

        Parameters
        ----------
        rank : int, optional
            Current process rank. ``None`` and ``0`` are treated as main
            process ranks.
        """
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        """Return ``True`` when a record should be emitted."""
        if getattr(record, "all_ranks", False):
            return True
        if record.levelno >= logging.WARNING:
            return True
        return self.rank is None or self.rank == 0


def configure_rank_logging(rank: int | None = None) -> None:
    """Configure the global SPINE logger to emit INFO logs on the main rank.

    Parameters
    ----------
    rank : int, optional
        Current process rank. ``None`` and ``0`` emit all records. Positive
        ranks suppress records below ``WARNING`` unless the log call provides
        ``extra={"all_ranks": True}``.
    """
    for target in [logger, *logger.handlers]:
        target.filters = [
            filter_obj
            for filter_obj in target.filters
            if not isinstance(filter_obj, MainProcessFilter)
        ]
        target.addFilter(MainProcessFilter(rank))


# Configure the formatting of the logger
logging.basicConfig(format="%(message)s", stream=sys.stdout)
# logging.basicConfig(format='[%(levelname)s] %(message)s')
# logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s')

# Capture warning messages and redirect them through the logger
logging.captureWarnings(True)

# Initialize logger
logger = logging.getLogger("spine")
configure_rank_logging()

# Configure the warnings package to only issue warnings once
warnings.simplefilter("ignore")

# Suppress MinkowskiEngine internal problems (nothing we can do about it)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="<class 'Minkowski"
)
