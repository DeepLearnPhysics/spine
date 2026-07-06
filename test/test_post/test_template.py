from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spine.post.template import TemplateProcessor


def test_template_processor_registers_prod_and_processes_objects():
    processor = TemplateProcessor("a", "b", obj_type="particle", run_mode="reco")
    obj = SimpleNamespace(
        is_truth=False,
        points=np.zeros((1, 3), dtype=np.float32),
        sources=np.zeros((1, 2), dtype=np.int32),
    )

    result = processor({"prod": {"reco_particles": [obj]}, "reco_particles": [obj]})

    assert processor.keys["prod"] is True
    assert result == {}


def test_template_processor_loops_over_all_object_key_groups():
    processor = TemplateProcessor(
        "a", "b", obj_type=("fragment", "particle", "interaction"), run_mode="reco"
    )
    obj = SimpleNamespace(
        is_truth=False,
        points=np.zeros((1, 3), dtype=np.float32),
        sources=np.zeros((1, 2), dtype=np.int32),
    )

    result = processor(
        {
            "prod": {
                "reco_fragments": [obj],
                "reco_particles": [obj],
                "reco_interactions": [obj],
            },
            "reco_fragments": [obj],
            "reco_particles": [obj],
            "reco_interactions": [obj],
        }
    )

    assert result == {}
