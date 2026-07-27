from __future__ import annotations

import numpy as np

from spine.ana.template import TemplateAna


def test_template_ana_uses_current_base_constructor(monkeypatch):
    writers = []
    monkeypatch.setattr(
        TemplateAna, "initialize_writer", lambda self, name: writers.append(name)
    )

    ana = TemplateAna("a", "b", obj_type="particle", run_mode="reco")

    assert ana.arg0 == "a"
    assert ana.arg1 == "b"
    assert ana.obj_keys == ["reco_particles"]
    assert writers == ["template"]
    assert ana.keys["prod"] is True


class FakeObject:
    start_point = np.array([1.0, 2.0, 3.0])
    end_point = np.array([4.0, 6.0, 8.0])


def test_template_ana_process_writes_displacements(monkeypatch):
    rows = []
    monkeypatch.setattr(TemplateAna, "initialize_writer", lambda self, name: None)
    monkeypatch.setattr(
        TemplateAna, "append", lambda self, name, **kwargs: rows.append((name, kwargs))
    )
    ana = TemplateAna("a", "b", obj_type="particle", run_mode="reco")

    ana.process({"prod": {"reco_particles": [FakeObject()]}})

    assert rows == [("template", {"disp_x": 3.0, "disp_y": 4.0, "disp_z": 5.0})]


def test_template_ana_processes_projected_columns(monkeypatch):
    monkeypatch.setattr(TemplateAna, "initialize_writer", lambda self, name: None)
    ana = TemplateAna("a", "b")
    starts = np.asarray([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    ends = np.asarray([[4.0, 6.0, 8.0], [3.0, 3.0, 3.0]])
    offsets = np.asarray([0, 1, 2], dtype=np.int64)

    result = ana.process_columnar(
        {
            "prod": {
                "start_point": starts,
                "end_point": ends,
                "event_offsets": offsets,
            }
        }
    )

    assert ana.supports_columnar
    assert np.array_equal(
        result["displacements"]["values"],
        np.asarray([[3.0, 4.0, 5.0], [3.0, 2.0, 1.0]]),
    )
    assert result["displacements"]["event_offsets"] is offsets
