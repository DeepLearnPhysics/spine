from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from spine.post.trigger.trigger import TriggerProcessor


def test_trigger_processor_validates_flash_keys(tmp_path):
    path = tmp_path / "triggers.csv"
    pd.DataFrame(
        [
            {
                "run_number": 1,
                "event_no": 2,
                "wr_seconds": 10,
                "wr_nanoseconds": 20,
                "beam_seconds": 9,
                "beam_nanoseconds": 10,
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="flash keys"):
        TriggerProcessor(str(path), correct_flash_times=True, flash_keys=None)

    with pytest.raises(FileNotFoundError):
        TriggerProcessor(str(tmp_path / "missing.csv"), correct_flash_times=False)


def test_trigger_processor_builds_trigger(tmp_path):
    path = tmp_path / "triggers.csv"
    pd.DataFrame(
        [
            {
                "run_number": 1,
                "event_no": 2,
                "wr_seconds": 10,
                "wr_nanoseconds": 20,
                "beam_seconds": 9,
                "beam_nanoseconds": 10,
                "trigger_type": 3,
            }
        ]
    ).to_csv(path, index=False)

    processor = TriggerProcessor(str(path), correct_flash_times=False)
    run_info = type("RunInfo", (), {"run": 1, "event": 2})()

    result = processor.process({"run_info": run_info})

    assert result["trigger"].type == 3


def test_trigger_processor_rejects_missing_or_duplicate_triggers(tmp_path):
    path = tmp_path / "triggers.csv"
    pd.DataFrame(
        [
            {
                "run_number": 1,
                "event_no": 2,
                "wr_seconds": 10,
                "wr_nanoseconds": 20,
                "beam_seconds": 9,
                "beam_nanoseconds": 10,
            },
            {
                "run_number": 1,
                "event_no": 2,
                "wr_seconds": 11,
                "wr_nanoseconds": 20,
                "beam_seconds": 9,
                "beam_nanoseconds": 10,
            },
        ]
    ).to_csv(path, index=False)
    processor = TriggerProcessor(str(path), correct_flash_times=False)

    with pytest.raises(KeyError, match="Could not find"):
        processor.process({"run_info": SimpleNamespace(run=9, event=9)})

    with pytest.raises(KeyError, match="more than one"):
        processor.process({"run_info": SimpleNamespace(run=1, event=2)})


def test_trigger_processor_corrects_flash_times(tmp_path):
    path = tmp_path / "triggers.csv"
    pd.DataFrame(
        [
            {
                "run_number": 1,
                "event_no": 2,
                "wr_seconds": 10,
                "wr_nanoseconds": 2000,
                "beam_seconds": 9,
                "beam_nanoseconds": 1000,
            }
        ]
    ).to_csv(path, index=False)
    processor = TriggerProcessor(
        str(path), flash_keys=["flashes"], flash_time_corr_us=4
    )
    flash = SimpleNamespace(time=1.0)

    processor.process({"run_info": SimpleNamespace(run=1, event=2), "flashes": [flash]})

    assert flash.time == pytest.approx(999998.0)
