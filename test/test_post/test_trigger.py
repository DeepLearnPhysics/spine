from __future__ import annotations

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
