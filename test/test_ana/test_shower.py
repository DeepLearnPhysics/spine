from __future__ import annotations

import pytest

from spine.ana.diag.shower import ShowerStartDEdxAna


def test_shower_start_dedx_ana_process_is_not_implemented(monkeypatch):
    monkeypatch.setattr(
        ShowerStartDEdxAna, "initialize_writer", lambda self, name: None
    )
    ana = ShowerStartDEdxAna(radius=2.0)

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        ana.process({})
