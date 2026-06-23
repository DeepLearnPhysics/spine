"""Tests for SPINE logger helpers."""

import io
import logging

from spine.utils.logger import MainProcessFilter, configure_rank_logging, logger


def test_main_process_filter_suppresses_low_priority_non_main_records():
    """Non-main ranks should suppress routine logs but keep warnings."""
    filter_obj = MainProcessFilter(rank=1)

    info = logging.LogRecord("spine", logging.INFO, __file__, 1, "info", (), None)
    warning = logging.LogRecord(
        "spine", logging.WARNING, __file__, 1, "warning", (), None
    )
    forced = logging.LogRecord("spine", logging.INFO, __file__, 1, "forced", (), None)
    forced.all_ranks = True

    assert not filter_obj.filter(info)
    assert filter_obj.filter(warning)
    assert filter_obj.filter(forced)


def test_configure_rank_logging_replaces_existing_rank_filters():
    """Rank logging configuration should be idempotent."""
    configure_rank_logging(rank=1)
    configure_rank_logging(rank=0)

    rank_filters = [
        filter_obj
        for filter_obj in logger.filters
        if isinstance(filter_obj, MainProcessFilter)
    ]
    assert len(rank_filters) == 1
    assert rank_filters[0].rank == 0

    configure_rank_logging()


def test_logger_filter_applies_to_global_spine_logger():
    """The global logger should suppress info messages on non-main ranks."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.INFO)

    try:
        configure_rank_logging(rank=1)
        logger.info("hidden")
        logger.warning("visible")
        logger.info("forced", extra={"all_ranks": True})
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        configure_rank_logging()

    output = stream.getvalue()
    assert "hidden" not in output
    assert "visible" in output
    assert "forced" in output
