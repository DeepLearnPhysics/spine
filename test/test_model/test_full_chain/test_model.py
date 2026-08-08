"""Unit tests for the provider-driven full reconstruction chain."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from spine.constants import GHOST_SHP, SHOWR_SHP, TRACK_SHP
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.full_chain import (
    FullChain,
    FullChainLoss,
    PointBatch,
    process_chain_config,
)
from spine.model.full_chain.config import build_chain_plan
from spine.model.full_chain.ops import AggregationOperations
from spine.model.full_chain.providers.aggregation import (
    GrapPALossStage,
    InteractionAggregationStage,
    ParticleAggregationStage,
    build_interaction_aggregation_loss,
    build_interaction_aggregation_stage,
    build_particle_aggregation_loss,
    build_particle_aggregation_stage,
)
from spine.model.full_chain.providers.fragmentation import (
    FragmentationStage,
    GraphSPICELossStage,
    build_fragmentation_loss,
    build_fragmentation_stage,
)
from spine.model.full_chain.providers.image import (
    ParticleImageLossStage,
    ParticleImageStage,
    build_particle_image_loss,
    build_particle_image_stage,
)
from spine.model.full_chain.providers.segmentation import (
    SegmentationLossStage,
    SegmentationStage,
    build_segmentation_loss,
    build_segmentation_stage,
)
from spine.model.full_chain.registry import (
    ProviderSpec,
    provider_spec,
    register_provider,
)
from spine.model.full_chain.stage import ChainStage
from spine.model.full_chain.state import ChainState, StageResult
from spine.utils.cluster.label import ClusterLabelAdapter


class ExternalStage(ChainStage):
    """Minimal external provider used to exercise import-path discovery."""

    def forward(self, state: ChainState) -> StageResult:
        return StageResult()


def _build_external(name, _config, _owner):
    """Build the import-path test provider."""
    return ExternalStage(name)


EXTERNAL_SPEC = ProviderSpec("test_external_import_provider", _build_external)


def make_data(size: int = 4) -> TensorBatch:
    """Build one sparse tensor batch with canonical coordinates and values."""
    rows = torch.zeros((size, 5), dtype=torch.float32)
    rows[:, 1] = torch.arange(size)
    rows[:, 4] = 1.0
    return TensorBatch(
        rows,
        counts=[size],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )


def make_cluster_label() -> ClusterLabelBatch:
    """Build compact cluster labels for two particles in one event."""
    rows = np.array(
        [
            [0, 0, 0, 0, 1, 10, 0],
            [0, 1, 0, 0, 1, 10, 0],
            [0, 2, 0, 0, 1, 20, 1],
            [0, 3, 0, 0, 1, 20, 1],
        ],
        dtype=np.float32,
    )
    data = TensorBatch(
        rows,
        counts=[4],
        has_batch_col=True,
        coord_cols=np.arange(1, 4),
    )
    particles = {
        "shape": TensorBatch(np.array([SHOWR_SHP, TRACK_SHP]), counts=[2]),
        "group": TensorBatch(np.array([7, 9]), counts=[2]),
        "group_primary": TensorBatch(np.array([1, 0]), counts=[2]),
    }
    return ClusterLabelBatch(data, particles)


def make_clusters() -> tuple[IndexBatch, TensorBatch]:
    """Build two voxel clusters and their semantic shapes."""
    clusts = IndexBatch(
        [np.array([0, 1]), np.array([2, 3])],
        spans=[4],
        counts=[2],
        single_counts=[2, 2],
    )
    shapes = TensorBatch(np.array([SHOWR_SHP, TRACK_SHP]), counts=[2])
    return clusts, shapes


def test_chain_state_enforces_declared_replacements() -> None:
    """Canonical products cannot be silently overwritten by a provider."""
    state = ChainState(data=object())
    with pytest.raises(ValueError, match="without declaring"):
        state.publish("bad", StageResult({"data": object()}))

    replacement = object()
    state.publish(
        "good",
        StageResult({"data": replacement}, {"data_adapt": replacement}),
        frozenset({"data"}),
    )
    assert state.require("data") is replacement


def test_chain_state_optional_required_and_output_contracts() -> None:
    """State access reports context and public outputs remain append-only."""
    state = ChainState(data=1, optional=None)
    assert "data" in state
    assert "optional" not in state
    assert state.get("missing", 3) == 3
    with pytest.raises(KeyError, match="for stage `consumer`"):
        state.require("missing", "consumer")

    state.publish("first", StageResult(outputs={"score": 1}))
    with pytest.raises(ValueError, match="duplicate public output.*score"):
        state.publish("second", StageResult(outputs={"score": 2}))


def test_chain_state_validates_point_data_and_removes_stale_aliases() -> None:
    """Point-family publication should validate type and synchronize aliases."""
    invalid = ChainState()
    with pytest.raises(TypeError, match="must be PointBatch"):
        invalid.publish("bad", StageResult({"point_data": object()}))

    data = make_data()
    sources = TensorBatch(torch.arange(8).reshape(4, 2), counts=[4])
    orig_index = IndexBatch(np.arange(4), spans=[4], counts=[4])
    state = ChainState(data=data, sources=sources, orig_index=orig_index)

    # Publishing a family without optional aligned products must also remove
    # their historical flat aliases, rather than leaving stale row domains.
    replacement = PointBatch.from_input(data)
    state.publish(
        "replace",
        StageResult({"point_data": replacement}),
        frozenset({"point_data"}),
    )

    assert state.require("data") is data
    assert "sources" not in state
    assert "orig_index" not in state


def test_stage_validation_reports_missing_inputs_and_collisions() -> None:
    """Provider contracts reject missing dependencies and undeclared writes."""

    class TestStage(ExternalStage):
        requires = frozenset({"input"})
        provides = frozenset({"output"})

    stage = TestStage("test")
    with pytest.raises(ValueError, match="unavailable products: input"):
        stage.validate(set())
    with pytest.raises(ValueError, match="replace undeclared products: output"):
        stage.validate({"input", "output"})
    assert stage.validate({"input"}) == {"input", "output"}
    with pytest.raises(KeyError, match="for stage `test`"):
        stage(ChainState())


def test_provider_registry_rejects_duplicates_and_resolves_import_paths() -> None:
    """Provider discovery supports extensions without allowing collisions."""
    first = ProviderSpec("test_duplicate_provider", _build_external)
    assert register_provider(first) is first
    assert register_provider(first) is first
    with pytest.raises(ValueError, match="already registered"):
        register_provider(ProviderSpec("test_duplicate_provider", _build_external))

    imported = provider_spec(f"{__name__}:EXTERNAL_SPEC")
    assert imported is EXTERNAL_SPEC
    with pytest.raises(TypeError, match="not a ProviderSpec"):
        provider_spec("pytest:mark")
    with pytest.raises(ValueError, match="Unknown full-chain provider"):
        provider_spec("missing_provider")


def test_provider_registry_lazily_imports_builtin(monkeypatch) -> None:
    """A missing builtin registration triggers its provider-module import."""
    import spine.model.full_chain.registry as registry

    saved = registry._PROVIDERS.pop("segmentation")

    def fake_import(_module):
        registry._PROVIDERS["segmentation"] = saved

    monkeypatch.setattr(registry, "import_module", fake_import)
    try:
        assert provider_spec("segmentation") is saved
    finally:
        registry._PROVIDERS["segmentation"] = saved


def test_native_plan_resolves_used_and_loss_blocks() -> None:
    """Ordered configs resolve named model blocks without flattening them."""
    chain = {
        "stages": [
            {
                "name": "semantic",
                "provider": "segmentation",
                "uses": "backbone",
                "loss": "objective",
                "config": {"mode": "label"},
            }
        ]
    }
    plan = build_chain_plan(
        chain,
        {"backbone": {"depth": 5}, "objective": {"loss": "ce"}},
    )
    assert plan[0].config == {"backbone": {"depth": 5}, "mode": "label"}
    assert plan[0].loss_config == {"loss": "ce"}


@pytest.mark.parametrize(
    ("stages", "error", "message"),
    [
        (None, ValueError, "nonempty list"),
        (["bad"], TypeError, "must be a mapping"),
        ([{"name": "stage"}], ValueError, "requires `name` and `provider`"),
        ([{"name": "", "provider": "test"}], ValueError, "stage names"),
        ([{"name": "stage", "provider": None}], ValueError, "provider names"),
        (
            [
                {"name": "stage", "provider": "test"},
                {"name": "stage", "provider": "test"},
            ],
            ValueError,
            "Duplicate",
        ),
        (
            [{"name": "stage", "provider": "test", "config": []}],
            TypeError,
            "config.*mapping",
        ),
        (
            [{"name": "stage", "provider": "test", "uses": [1]}],
            TypeError,
            "uses.*block names",
        ),
        (
            [{"name": "stage", "provider": "test", "loss": []}],
            TypeError,
            "loss must",
        ),
    ],
)
def test_native_plan_rejects_malformed_stage_descriptors(stages, error, message):
    """The ordered schema rejects malformed descriptors at its boundary."""
    with pytest.raises(error, match=message):
        build_chain_plan({"stages": stages}, {})


def test_native_plan_validates_model_and_loss_references() -> None:
    """Referenced blocks must exist when required and must be mappings."""
    descriptor = {"name": "stage", "provider": "test", "uses": "model"}
    with pytest.raises(ValueError, match="missing block `model`"):
        build_chain_plan({"stages": [descriptor]}, {})

    descriptor = {"name": "stage", "provider": "test", "loss": "objective"}
    with pytest.raises(ValueError, match="missing loss `objective`"):
        build_chain_plan({"stages": [descriptor]}, {}, require_losses=True)
    assert build_chain_plan({"stages": [descriptor]}, {})[0].loss_config is None
    with pytest.raises(TypeError, match="Loss block `objective`"):
        build_chain_plan({"stages": [descriptor]}, {"objective": []})

    descriptor["loss"] = {"node": "objective"}
    with pytest.raises(TypeError, match="must map names"):
        build_chain_plan(
            {"stages": [descriptor | {"loss": {1: "objective"}}]},
            {"objective": {}},
        )
    with pytest.raises(ValueError, match="missing loss `objective`"):
        build_chain_plan({"stages": [descriptor]}, {}, require_losses=True)
    with pytest.raises(TypeError, match="Loss block `objective`"):
        build_chain_plan({"stages": [descriptor]}, {"objective": []})
    assert build_chain_plan({"stages": [descriptor]}, {})[0].loss_config is None


def test_chain_plan_rejects_invalid_top_level_schemas() -> None:
    """Native and legacy top-level configuration contracts are unambiguous."""
    with pytest.raises(TypeError, match="must be a mapping"):
        build_chain_plan([], {})
    with pytest.raises(ValueError, match="unknown keys: segmentation"):
        build_chain_plan({"stages": [], "segmentation": "label"}, {})
    with pytest.raises(ValueError, match="at least one stage"):
        build_chain_plan({}, {})


@pytest.mark.parametrize(
    ("calibration", "message"),
    [
        (None, "requires a `calibration` configuration block"),
        ({}, "requires `stage`"),
        ({"stage": "missing"}, "target stage `missing` is not enabled"),
    ],
)
def test_legacy_calibration_validates_placement(calibration, message) -> None:
    """The unordered legacy schema needs a valid calibration insertion target."""
    modules = {} if calibration is None else {"calibration": calibration}
    with pytest.raises(ValueError, match=message):
        build_chain_plan(
            {"segmentation": "label", "calibration": "label"},
            modules,
        )


def test_legacy_calibration_resolves_alias_and_precedes_target() -> None:
    """Historical task aliases place calibration immediately before its stage."""
    plan = build_chain_plan(
        {
            "fragmentation": "label",
            "particle_aggregation": "label",
            "particle_identification": "image",
            "calibration": "label",
        },
        {
            "image_particle": {"encoder": {}, "heads": {"pid": 5}},
            "calibration": {"stage": "particle_identification", "gain": 2},
        },
    )
    assert [stage.name for stage in plan] == [
        "fragmentation",
        "particle_aggregation",
        "calibration_before_particle_image",
        "particle_image",
    ]
    assert plan[2].config == {"mode": "label", "calibration": {"gain": 2}}


def test_legacy_plan_translates_particle_image_tasks() -> None:
    """Historical image task modes become an explicit image provider."""
    plan = build_chain_plan(
        {
            "fragmentation": "label",
            "particle_aggregation": "label",
            "particle_identification": "image",
        },
        {"image_particle": {"encoder": {}, "heads": {"type": 5}}},
    )
    assert [stage.provider for stage in plan] == [
        "fragmentation",
        "particle_aggregation",
        "particle_image",
    ]


def test_legacy_plan_translates_voxel_and_interaction_stages() -> None:
    """Legacy preprocessing, semantics, and interaction modes retain order."""
    plan = build_chain_plan(
        {
            "deghosting": "label",
            "charge_rescaling": "label",
            "segmentation": "label",
            "point_proposal": "label",
            "inter_aggregation": "label",
        },
        {
            "uresnet_deghost_loss": {"balance_loss": False},
            "uresnet_loss": {"balance_loss": False},
            "grappa_inter_loss": {"edge_loss": {}},
        },
    )
    assert [stage.provider for stage in plan] == [
        "deghost",
        "segmentation",
        "interaction_aggregation",
    ]
    assert plan[0].config["charge_rescaling"] == "label"
    assert plan[1].config["point_proposal"] == "label"
    assert plan[2].config["task_modes"] == {
        "type": None,
        "primary": None,
        "orient": None,
    }


def test_external_provider_can_supply_multiple_capabilities() -> None:
    """A combined semantic/fragment provider needs no FullChain changes."""

    class CombinedStage(ChainStage):
        requires = frozenset({"data"})
        provides = frozenset({"seg_pred", "fragment_clusts", "fragment_shapes"})

        def forward(self, state: ChainState) -> StageResult:
            state.require("data", self.name)
            clusts = IndexBatch([np.arange(4)], [4], [1], [4])
            products = {
                "seg_pred": TensorBatch(torch.zeros(4, dtype=torch.long), [4]),
                "fragment_clusts": clusts,
                "fragment_shapes": TensorBatch(np.array([SHOWR_SHP]), [1]),
            }
            return StageResult(products, dict(products))

    def build(name: str, _config: dict[str, Any], _owner: Any) -> ChainStage:
        return CombinedStage(name)

    provider_name = "test_combined_semantic_fragments"
    register_provider(ProviderSpec(provider_name, build))
    chain = FullChain(
        chain={
            "stages": [
                {"name": "combined", "provider": provider_name},
            ]
        }
    )
    result = chain(make_data())
    assert set(result) == {"seg_pred", "fragment_clusts", "fragment_shapes"}


def test_full_chain_publishes_final_data_adaptation_once() -> None:
    """Deghosted charge and calibrated energy remain separate and aligned."""
    chain = FullChain(
        chain={
            "stages": [
                {
                    "name": "deghost",
                    "provider": "deghost",
                    "mode": "label",
                },
                {
                    "name": "calibration",
                    "provider": "calibration",
                    "mode": "label",
                },
            ]
        }
    )
    seg_label = TensorBatch(torch.tensor([0, GHOST_SHP, 1, GHOST_SHP]), [4])
    energy_label = TensorBatch(torch.tensor([10.0, 20.0, 30.0, 40.0]), [4])

    result = chain(
        data=make_data(),
        seg_label=seg_label,
        energy_label=energy_label,
    )

    assert result["data_adapt"].values.torch_tensor().tolist() == [1.0, 1.0]
    assert result["data_calib"].values.torch_tensor().tolist() == [10.0, 30.0]
    assert result["orig_index"].index.tolist() == [0, 2]


def test_segmentation_deghosts_only_row_aligned_ppn_outputs() -> None:
    """Internal sparse PPN products retain their independently pruned rows."""

    class Model:
        def __call__(self, _data: TensorBatch) -> dict[str, Any]:
            return {
                "segmentation": TensorBatch(torch.zeros((3, 5)), [3]),
                "ghost": TensorBatch(
                    torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]),
                    [3],
                ),
                "ppn_points": TensorBatch(torch.arange(30).reshape(3, 10), [3]),
                "ppn_coords": [TensorBatch(torch.zeros((1, 4)), [1])],
            }

    stage = SegmentationStage(
        "semantic",
        "uresnet",
        Model(),  # type: ignore[arg-type]
        ClusterLabelAdapter(),
    )
    state = ChainState(data=make_data(3))
    result = stage(state)
    assert result.outputs["ppn_points"].counts.tolist() == [2]
    assert result.outputs["ppn_coords"][0].counts.tolist() == [1]


def test_segmentation_loss_aligns_cached_deghosted_rows() -> None:
    """Cached deghosting indexes align semantic labels and logits."""
    seg_label = TensorBatch(
        torch.tensor([0, GHOST_SHP, 1, 2, GHOST_SHP]).float(),
        [5],
    )
    logits = TensorBatch(torch.zeros((3, 3)), [3])
    orig_index = IndexBatch(torch.tensor([0, 2, 4]), spans=[5], counts=[3])
    aligned_label, aligned_logits = SegmentationLossStage._align(
        seg_label,
        logits,
        orig_index,
    )
    assert aligned_label.values.torch_tensor().tolist() == [0.0, 1.0]
    assert aligned_logits.shape[0] == 2


def test_segmentation_stages_validate_modes_and_required_inputs() -> None:
    """Semantic adapters reject unknown modes and unavailable implementations."""
    adapter = ClusterLabelAdapter()
    with pytest.raises(ValueError, match="mode must"):
        SegmentationStage("semantic", "bad", None, adapter)

    label_stage = SegmentationStage("semantic", "label", None, adapter)
    with pytest.raises(ValueError, match="requires `seg_label`"):
        label_stage(ChainState(data=make_data()))
    label_result = label_stage(
        ChainState(
            data=make_data(),
            seg_label=TensorBatch(torch.tensor([0, 1, 1, 0]), [4]),
        )
    )
    assert label_result.products["seg_pred"].counts.tolist() == [4]

    learned_stage = SegmentationStage("semantic", "uresnet", None, adapter)
    with pytest.raises(RuntimeError, match="not initialized"):
        learned_stage(ChainState(data=make_data()))


def test_segmentation_loss_stage_validates_and_routes_inputs() -> None:
    """Semantic loss routing requires truth/logits and prefers adapted labels."""

    class Loss:
        def __call__(self, **inputs):
            return inputs

    stage = SegmentationLossStage("semantic", Loss())
    with pytest.raises(ValueError, match="requires labels and logits"):
        stage({})

    labels = make_data(2)
    logits = TensorBatch(torch.zeros((2, 3)), [2])
    adapted = object()
    result = stage(
        {
            "seg_label": labels,
            "segmentation": logits,
            "clust_label": object(),
            "clust_label_adapt": adapted,
        }
    )
    assert result["seg_label"] is labels
    assert result["segmentation"] is logits
    assert result["clust_label"] is adapted


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ({}, ValueError, "string `mode`"),
        ({"mode": "uresnet"}, ValueError, "exactly one"),
        (
            {"mode": "uresnet", "uresnet": {}, "uresnet_ppn": {}},
            ValueError,
            "exactly one",
        ),
        (
            {"mode": "uresnet", "uresnet": {}, "point_proposal": "ppn"},
            ValueError,
            "requires `uresnet_ppn`",
        ),
        ({"mode": "uresnet", "uresnet": []}, TypeError, "must be a mapping"),
        ({"mode": "uresnet", "uresnet_ppn": []}, TypeError, "must be a mapping"),
        ({"mode": "label", "adapt_labels": "bad"}, TypeError, "must be a mapping"),
    ],
)
def test_segmentation_builder_validates_configuration(config, error, message):
    """Semantic model selection is unambiguous at the provider boundary."""
    with pytest.raises(error, match=message):
        build_segmentation_stage("semantic", config, torch.nn.Module())


def test_segmentation_loss_builder_validates_configuration() -> None:
    """Semantic objectives require matching model and mapping loss blocks."""
    owner = torch.nn.Module()
    assert build_segmentation_loss("semantic", {}, owner) is None
    with pytest.raises(TypeError, match="loss configuration"):
        build_segmentation_loss("semantic", {"loss": []}, owner)
    with pytest.raises(TypeError, match="uresnet_ppn.*mapping"):
        build_segmentation_loss("semantic", {"loss": {}, "uresnet_ppn": []}, owner)
    with pytest.raises(ValueError, match="requires a `uresnet`"):
        build_segmentation_loss("semantic", {"loss": {}}, owner)


def test_segmentation_builders_register_standalone_modules(monkeypatch) -> None:
    """Valid standalone segmentation blocks are attached to their owners."""

    class FakeModel(torch.nn.Module):
        def __init__(self, _config):
            super().__init__()

    class FakeLoss(torch.nn.Module):
        def __init__(self, _model, _loss):
            super().__init__()

    monkeypatch.setattr(
        "spine.model.full_chain.providers.segmentation.UResNetSegmentation",
        FakeModel,
    )
    monkeypatch.setattr(
        "spine.model.full_chain.providers.segmentation.SegmentationLoss", FakeLoss
    )
    model_owner = torch.nn.Module()
    stage = build_segmentation_stage(
        "semantic", {"mode": "uresnet", "uresnet": {}}, model_owner
    )
    loss_owner = torch.nn.Module()
    loss_stage = build_segmentation_loss(
        "semantic", {"uresnet": {}, "loss": {}}, loss_owner
    )

    assert model_owner.uresnet is stage.model
    assert loss_stage is not None
    assert loss_owner.uresnet_loss is loss_stage.loss


def test_label_fragmentation_uses_structured_shape_field() -> None:
    """Truth fragmentation reads shape through ClusterLabelBatch aliases."""
    stage = FragmentationStage("fragments", "label", None, None)
    state = ChainState(
        data=make_data(),
        seg_pred=TensorBatch(torch.zeros(4, dtype=torch.long), [4]),
        clust_label=make_cluster_label(),
    )
    result = stage(state)
    assert [
        index.tolist() for index in result.products["fragment_clusts"].index_list
    ] == [
        [0, 1],
        [2, 3],
    ]
    assert result.products["fragment_shapes"].numpy_tensor().tolist() == [
        SHOWR_SHP,
        TRACK_SHP,
    ]


def test_fragmentation_stage_validates_mode_and_label_truth() -> None:
    """Fragmentation rejects unknown implementations and absent label truth."""
    with pytest.raises(ValueError, match="Unknown fragmentation mode"):
        FragmentationStage("fragments", "bad", None, None)
    stage = FragmentationStage("fragments", "label", None, None)
    with pytest.raises(ValueError, match="requires `clust_label`"):
        stage(
            ChainState(
                data=make_data(),
                seg_pred=TensorBatch(torch.zeros(4, dtype=torch.long), [4]),
            )
        )


def test_graph_spice_loss_adapter_validates_and_denamespaces() -> None:
    """The fragment loss adapter requires truth/semantics and strips prefixes."""

    class Loss:
        def __call__(self, **inputs):
            return inputs

    stage = GraphSPICELossStage("fragments", Loss())
    with pytest.raises(ValueError, match="requires `clust_label`"):
        stage({})
    with pytest.raises(ValueError, match="requires `seg_pred`"):
        stage({"clust_label": make_cluster_label()})

    result = stage(
        {
            "clust_label": make_cluster_label(),
            "seg_pred": TensorBatch(torch.tensor([0, 0, 1, 1]), [4]),
            "graph_spice_edge_attr": "edges",
        }
    )
    assert result["edge_attr"] == "edges"
    assert result["seg_label"].shape == (4, 1)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "string `mode`"),
        ({"mode": "dbscan"}, "requires a `dbscan` block"),
        ({"mode": "graph_spice"}, "requires a `graph_spice` block"),
    ],
)
def test_fragmentation_builder_validates_required_blocks(config, message):
    """Selected fragment implementations require their native configurations."""
    with pytest.raises(ValueError, match=message):
        build_fragmentation_stage("fragments", config, torch.nn.Module())


def test_fragmentation_loss_builder_validates_blocks() -> None:
    """Graph-SPICE supervision is optional but mapping-typed when enabled."""
    owner = torch.nn.Module()
    assert build_fragmentation_loss("fragments", {}, owner) is None
    assert build_fragmentation_loss("fragments", {"graph_spice": {}}, owner) is None
    with pytest.raises(TypeError, match="must be mappings"):
        build_fragmentation_loss(
            "fragments",
            {"graph_spice": [], "loss": {}},
            owner,
        )


def test_dbscan_fragmentation_executes_and_builds(monkeypatch) -> None:
    """DBSCAN outputs merge into the canonical empty fragment collection."""

    class FakeDBSCAN(torch.nn.Module):
        shapes = [0, 1, 2, 3]

        def __init__(self, **_config):
            super().__init__()

        def forward(self, data, seg_pred, coord_label=None, **outputs):
            assert outputs == {"ppn_points": "points"}
            clusts, shapes = make_clusters()
            return clusts, shapes

    monkeypatch.setattr(
        "spine.model.full_chain.providers.fragmentation.DBSCAN", FakeDBSCAN
    )
    owner = torch.nn.Module()
    stage = build_fragmentation_stage(
        "fragments", {"mode": "dbscan", "dbscan": {}}, owner
    )
    state = ChainState(
        data=make_data(),
        seg_pred=TensorBatch(torch.zeros(4, dtype=torch.long), [4]),
    )
    state.outputs["ppn_points"] = "points"
    result = stage(state)

    assert owner.dbscan is stage.dbscan
    assert len(result.products["fragment_clusts"].index_list) == 2


def test_fragmentation_builder_rejects_incomplete_shape_ownership(monkeypatch) -> None:
    """Configured fragmenters must collectively cover each supported shape."""

    class PartialDBSCAN(torch.nn.Module):
        shapes = [0]

        def __init__(self, **_config):
            super().__init__()

    monkeypatch.setattr(
        "spine.model.full_chain.providers.fragmentation.DBSCAN", PartialDBSCAN
    )
    with pytest.raises(ValueError, match="collectively own"):
        build_fragmentation_stage(
            "fragments", {"mode": "dbscan", "dbscan": {}}, torch.nn.Module()
        )


def test_group_builder_falls_back_when_primary_is_missing() -> None:
    """Primary-aware grouping falls back to the full group when necessary."""
    clusts, shapes = make_clusters()
    assignments = TensorBatch(np.array([5, 5]), [2])
    primary_mask = TensorBatch(np.array([False, False]), [2])
    groups, group_shapes, primaries = AggregationOperations.build_groups(
        clusts,
        shapes,
        assignments,
        primary_mask,
        aggregate_shapes=True,
        shape_use_primary=True,
        retain_primaries=True,
    )
    assert [group.tolist() for group in groups.index_list] == [[0, 1, 2, 3]]
    assert group_shapes.numpy_tensor().tolist() == [SHOWR_SHP]
    assert primaries.index_list[0].tolist() == [0, 1, 2, 3]


def test_grappa_builder_validates_and_registers_group_model(monkeypatch) -> None:
    """Aggregation's native-model helper owns only group-producing GrapPAs."""
    from spine.model.full_chain.providers.aggregation import _build_grappa

    with pytest.raises(ValueError, match="requires a `particle_grappa` block"):
        _build_grappa("particle_grappa", None, torch.nn.Module())

    class FakeGrapPA(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.make_groups = config["make_groups"]

    monkeypatch.setattr(
        "spine.model.full_chain.providers.aggregation.GrapPA", FakeGrapPA
    )
    with pytest.raises(ValueError, match="make_groups: true"):
        _build_grappa("particle_grappa", {"make_groups": False}, torch.nn.Module())

    owner = torch.nn.Module()
    model = _build_grappa("particle_grappa", {"make_groups": True}, owner)
    assert owner.particle_grappa is model


def test_group_builder_uses_primary_shape_and_voxels() -> None:
    """Primary-aware grouping selects the marked node's identity and indexes."""
    clusts, shapes = make_clusters()
    assignments = TensorBatch(np.array([5, 5]), [2])
    primary_mask = TensorBatch(np.array([False, True]), [2])

    groups, group_shapes, primaries = AggregationOperations.build_groups(
        clusts,
        shapes,
        assignments,
        primary_mask,
        aggregate_shapes=True,
        shape_use_primary=True,
        retain_primaries=True,
    )

    assert groups.index_list[0].tolist() == [0, 1, 2, 3]
    assert group_shapes.numpy_tensor().tolist() == [TRACK_SHP]
    assert primaries.index_list[0].tolist() == [2, 3]


def test_aggregation_input_preparation_covers_optional_features() -> None:
    """GrapPA adapters build point, charge, shape, and truth-coordinate inputs."""
    operations = AggregationOperations()
    data = make_data()
    data = TensorBatch(
        torch.cat((data.data, torch.full((len(data.data), 1), 1000.0)), dim=1),
        data.counts,
        has_batch_col=data.has_batch_col,
        coord_cols=data.coord_cols,
    )
    clusts, shapes = make_clusters()

    encoder = SimpleNamespace(
        add_points=False,
        add_value=True,
        add_shape=True,
    )
    coord_label = TensorBatch(torch.zeros((2, 6)), [2])
    result = operations.prepare_grappa_input(
        SimpleNamespace(node_encoder=encoder),
        {},
        data,
        clusts,
        shapes,
        coord_label=coord_label,
    )
    assert result["coord_label"] is coord_label
    assert result["extra"].shape == (2, 3)
    assert result["extra"].data[:, 0].tolist() == [1.0, 1.0]

    encoder.add_points = True
    with pytest.raises(ValueError, match="require `primaries`"):
        operations.prepare_grappa_input(
            SimpleNamespace(node_encoder=encoder),
            {},
            data,
            clusts,
            shapes,
            point_use_primaries=True,
        )
    with pytest.raises(ValueError, match="require `ppn_points`"):
        operations.prepare_grappa_input(
            SimpleNamespace(node_encoder=encoder),
            {},
            data,
            clusts,
            shapes,
        )

    expected_points = TensorBatch(torch.zeros((2, 6)), [2])
    operations.point_predictor = lambda *_args: expected_points
    result = operations.prepare_grappa_input(
        SimpleNamespace(node_encoder=encoder),
        {"ppn_points": object()},
        data,
        clusts,
        shapes,
    )
    assert result["points"] is expected_points


def test_aggregation_input_derives_truth_points(monkeypatch) -> None:
    """A truth-only GrapPA path derives endpoints from the two label products."""
    operations = AggregationOperations()
    clusts, shapes = make_clusters()
    expected = TensorBatch(torch.zeros((2, 6)), [2])
    monkeypatch.setattr(
        "spine.model.full_chain.ops.get_cluster_points_label_batch",
        lambda *_args, **_kwargs: expected,
    )
    encoder = SimpleNamespace(
        add_points=True,
        add_value=False,
        add_shape=False,
        random_order=True,
    )
    result = operations.prepare_grappa_input(
        SimpleNamespace(node_encoder=encoder),
        {},
        make_data(),
        clusts,
        shapes,
        clust_label=make_cluster_label(),
        coord_label=TensorBatch(torch.zeros((2, 6)), [2]),
    )
    assert result["points"] is expected


def test_grappa_execution_requires_logits_for_primary_grouping() -> None:
    """Primary-aware group building requires a node-classification head."""
    operations = AggregationOperations()
    operations.prepare_grappa_input = lambda *_args, **_kwargs: {}
    clusts, shapes = make_clusters()

    class Model:
        def __call__(self, **_inputs):
            return {"group_pred": TensorBatch(np.array([0, 0]), [2])}

    with pytest.raises(ValueError, match="requires `node_pred`"):
        operations.run_grappa(
            Model(),
            {},
            make_data(),
            clusts,
            shapes,
            [SHOWR_SHP, TRACK_SHP],
            shape_use_primary=True,
        )


def test_truth_grouping_restricts_shapes_and_retains_primaries() -> None:
    """Truth aggregation shares semantic restriction and primary logic."""
    clusts, shapes = make_clusters()
    groups, group_shapes, primaries, selected = AggregationOperations.group_labels(
        make_cluster_label(),
        clusts,
        shapes,
        accepted_shapes=[SHOWR_SHP],
        aggregate_shapes=True,
        shape_use_primary=True,
        retain_primaries=True,
    )
    assert selected.tolist() == [0]
    assert groups.index_list[0].tolist() == [0, 1]
    assert group_shapes.numpy_tensor().tolist() == [SHOWR_SHP]
    assert primaries.index_list[0].tolist() == [0, 1]


def test_particle_aggregation_skip_and_label_paths() -> None:
    """Particle aggregation promotes fragments or groups them from truth."""
    data = make_data()
    clusts, shapes = make_clusters()
    operations = AggregationOperations()
    state = ChainState(data=data, fragment_clusts=clusts, fragment_shapes=shapes)
    stage = ParticleAggregationStage(
        "particles",
        {"shower": "skip", "track": "skip", "particle": None},
        {},
        operations,
    )

    result = stage(state)

    assert len(result.products["particle_clusts"].index_list) == 2
    assert result.products["particle_shapes"].numpy_tensor().tolist() == [
        SHOWR_SHP,
        TRACK_SHP,
    ]

    label_stage = ParticleAggregationStage(
        "particles",
        {"shower": None, "track": None, "particle": "label"},
        {},
        operations,
    )
    with pytest.raises(ValueError, match="requires `clust_label`"):
        label_stage(state)

    label_state = ChainState(
        data=data,
        fragment_clusts=clusts,
        fragment_shapes=shapes,
        clust_label=make_cluster_label(),
    )
    assert len(label_stage(label_state).products["particle_clusts"].index_list) == 2

    disabled = ParticleAggregationStage(
        "particles",
        {"shower": None, "track": None, "particle": None},
        {},
        operations,
    )
    with pytest.raises(RuntimeError, match="disabled"):
        disabled._run_path("particle", [SHOWR_SHP], False, state)


def test_particle_aggregation_merges_grappa_diagnostics() -> None:
    """Separate learned paths restore fragment-aligned native diagnostics."""
    data = make_data()
    clusts, shapes = make_clusters()

    class Operations:
        def run_grappa(self, *args, **kwargs):
            selected, selected_shapes, selected_index = (
                AggregationOperations.restrict_clusters(
                    clusts,
                    shapes,
                    [SHOWR_SHP],
                )
            )
            native = {
                "start_points": TensorBatch(torch.ones((1, 3)), [1]),
                "end_points": TensorBatch(torch.ones((1, 3)), [1]),
                "node_pred": TensorBatch(torch.ones((1, 2)), [1]),
                "group_pred": TensorBatch(np.zeros(1, dtype=np.int64), [1]),
            }
            return selected, selected_shapes, selected, selected_index, native

    stage = ParticleAggregationStage(
        "particles",
        {"shower": "grappa", "track": None, "particle": None},
        {"shower": object()},
        Operations(),
    )
    result = stage(
        ChainState(data=data, fragment_clusts=clusts, fragment_shapes=shapes)
    )

    assert result.outputs["fragment_start_points"].shape == (2, 3)
    assert result.outputs["fragment_group_pred"].numpy_tensor().tolist() == [1, 0]


def test_interaction_aggregation_modes_and_task_ownership() -> None:
    """Interaction grouping validates context and suppresses image-owned heads."""
    with pytest.raises(ValueError, match="mode must be"):
        InteractionAggregationStage("inter", "skip", None, AggregationOperations())

    data = make_data()
    clusts, shapes = make_clusters()
    state = ChainState(
        data=data,
        particle_clusts=clusts,
        particle_shapes=shapes,
        particle_primaries=clusts,
    )
    label_stage = InteractionAggregationStage(
        "inter", "label", None, AggregationOperations()
    )
    with pytest.raises(ValueError, match="requires `clust_label`"):
        label_stage(state)

    label_state = ChainState(
        data=data,
        particle_clusts=clusts,
        particle_shapes=shapes,
        particle_primaries=clusts,
        clust_label=make_cluster_label(),
    )
    assert len(label_stage(label_state).products["interaction_clusts"].index_list) == 2

    no_model = InteractionAggregationStage(
        "inter", "grappa", None, AggregationOperations()
    )
    with pytest.raises(RuntimeError, match="not initialized"):
        no_model(state)

    class Model:
        node_type = [SHOWR_SHP, TRACK_SHP]

    class Operations:
        def run_grappa(self, *args, **kwargs):
            native = {
                "clusts": clusts,
                "node_type_pred": TensorBatch(torch.zeros((2, 2)), [2]),
                "node_primary_pred": TensorBatch(torch.zeros((2, 2)), [2]),
                "edge_pred": TensorBatch(torch.zeros((1, 2)), [1]),
            }
            return clusts, shapes, clusts, None, native

    stage = InteractionAggregationStage(
        "inter",
        "grappa",
        Model(),
        Operations(),
        {"type": "image"},
    )
    result = stage(state)
    assert "particle_node_type_pred" not in result.outputs
    assert "particle_node_primary_pred" in result.outputs
    assert "particle_edge_pred" in result.outputs


def test_grappa_loss_stage_restores_native_namespace() -> None:
    """The full-chain loss adapter strips its public aggregation prefix."""
    seen = {}

    class Loss:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return {"loss": torch.tensor(0.0)}

    stage = GrapPALossStage("loss", "fragment_", Loss())
    with pytest.raises(ValueError, match="requires `clust_label`"):
        stage({})

    label = make_cluster_label()
    result = stage(
        {
            "clust_label": label,
            "fragment_node_pred": object(),
            "unrelated": object(),
        }
    )
    assert result["loss"] == 0.0
    assert seen["clust_label"] is label
    assert "node_pred" in seen
    assert "unrelated" not in seen


def test_aggregation_builders_validate_modes_and_loss_mappings(monkeypatch) -> None:
    """Aggregation builders reject contradictory and malformed configurations."""
    owner = torch.nn.Module()
    with pytest.raises(ValueError, match="Unknown shower"):
        build_particle_aggregation_stage(
            "particles", {"shower_aggregation": "mystery"}, owner
        )
    with pytest.raises(ValueError, match="requires GrapPA shower"):
        build_particle_aggregation_stage(
            "particles",
            {"shower_aggregation": "skip", "shower_primary": "grappa"},
            owner,
        )
    with pytest.raises(ValueError, match="not both"):
        build_particle_aggregation_stage(
            "particles",
            {"particle_aggregation": "skip", "track_aggregation": "skip"},
            owner,
        )
    with pytest.raises(ValueError, match="string `mode`"):
        build_interaction_aggregation_stage("inter", {}, owner)
    with pytest.raises(TypeError, match="task_modes"):
        build_interaction_aggregation_stage(
            "inter", {"mode": "label", "task_modes": []}, owner
        )

    with pytest.raises(TypeError, match="configuration must be a mapping"):
        build_particle_aggregation_loss("loss", {"loss": [{}]}, owner)
    assert build_particle_aggregation_loss("loss", {}, owner) is None
    assert (
        build_particle_aggregation_loss("loss", {"loss": {"shower": None}}, owner)
        is None
    )
    with pytest.raises(TypeError, match="must be a mapping"):
        build_particle_aggregation_loss("loss", {"loss": {"shower": []}}, owner)

    assert build_interaction_aggregation_loss("loss", {}, owner) is None
    with pytest.raises(TypeError, match="must be a mapping"):
        build_interaction_aggregation_loss("loss", {"loss": []}, owner)
    with pytest.raises(ValueError, match="ambiguous"):
        build_interaction_aggregation_loss(
            "loss",
            {
                "task_modes": {"type": "image"},
                "loss": {"node_loss": {"name": "class", "target": "pid"}},
            },
            owner,
        )


def test_full_chain_loss_aggregates_and_validates_stage_results() -> None:
    """Chain-wide summaries weight accuracy and namespace diagnostics."""

    class LossStage:
        def __init__(self, name, result):
            self.name = name
            self.result = result

        def __call__(self, _data):
            return self.result

    loss = object.__new__(FullChainLoss)
    torch.nn.Module.__init__(loss)
    loss.stages = [
        LossStage("first", {"loss": torch.tensor(2.0), "accuracy": 0.5}),
        LossStage(
            "second",
            {
                "loss": torch.tensor(3.0),
                "accuracy": 1.0,
                "num_losses": 2,
                "detail": 7,
            },
        ),
    ]
    result = loss()
    torch.testing.assert_close(result["loss"], torch.tensor(5.0))
    assert result["accuracy"] == pytest.approx(5 / 6)
    assert result["num_losses"] == 3
    assert result["second_detail"] == 7

    loss.stages = [LossStage("empty", {"loss": 0.0, "num_losses": 0})]
    with pytest.raises(ValueError, match="reported no objectives"):
        loss()


def test_full_chain_loss_skips_provider_without_objective() -> None:
    """Inference-only providers contribute no loss stage."""
    name = "test_no_loss_provider"
    register_provider(ProviderSpec(name, _build_external))
    loss = FullChainLoss(chain={"stages": [{"name": "external", "provider": name}]})
    assert loss.stages == []
    assert loss() == {"loss": 0.0, "accuracy": 1.0, "num_losses": 0}


def test_legacy_process_chain_config_sets_owner_attributes() -> None:
    """The compatibility helper returns a plan and mirrors legacy modes."""
    owner = SimpleNamespace()
    plan = process_chain_config(owner, segmentation="label", dump_config=True)
    assert owner.segmentation == "label"
    assert plan[0].provider == "segmentation"


def test_particle_image_publishes_grappa_compatible_keys() -> None:
    """Image task predictions satisfy existing particle-builder interfaces."""

    class Model:
        def __call__(self, data: TensorBatch, objects: IndexBatch) -> dict[str, Any]:
            assert len(objects.index_list) == 2
            return {
                "type_pred": TensorBatch(torch.zeros((2, 5)), objects.counts),
                "energy_pred": TensorBatch(torch.ones((2, 1)), objects.counts),
            }

    particles, _ = make_clusters()
    stage = ParticleImageStage(
        "particle_tasks",
        Model(),  # type: ignore[arg-type]
        {"type": "type", "energy": "energy"},
    )
    result = stage(ChainState(data=make_data(), particle_clusts=particles))
    assert set(result.outputs) == {
        "particle_node_type_pred",
        "particle_node_energy_pred",
    }
    assert set(result.products) == {"particle_type_pred", "particle_energy_pred"}


def test_particle_image_publishes_optional_shared_features() -> None:
    """Shared image features remain diagnostic outputs, not chain products."""

    class Model:
        def __call__(self, _data, objects):
            return {
                "pid_pred": TensorBatch(torch.zeros((2, 5)), objects.counts),
                "features": TensorBatch(torch.ones((2, 3)), objects.counts),
            }

    particles, _ = make_clusters()
    result = ParticleImageStage(
        "particle_tasks",
        Model(),  # type: ignore[arg-type]
        {"pid": "type"},
    )(ChainState(data=make_data(), particle_clusts=particles))
    assert result.outputs["particle_image_features"].shape == (2, 3)
    assert "particle_image_features" not in result.products


def test_particle_image_loss_restores_native_prediction_names() -> None:
    """The loss adapter reverses canonical task aliases before evaluation."""

    class Loss:
        def __init__(self):
            self.data = None

        def __call__(self, objects, **data):
            self.data = (objects, data)
            return {"loss": torch.tensor(0.0)}

    particles, _ = make_clusters()
    prediction = TensorBatch(torch.zeros((2, 5)), particles.counts)
    loss = Loss()
    stage = ParticleImageLossStage(
        "particle_tasks",
        loss,  # type: ignore[arg-type]
        {"pid": "type"},
    )
    result = stage(
        {
            "particle_clusts": particles,
            "particle_node_type_pred": prediction,
        }
    )
    assert result["loss"].item() == 0.0
    assert loss.data[1]["pid_pred"] is prediction

    with pytest.raises(ValueError, match="requires `particle_clusts`"):
        stage({})
    with pytest.raises(ValueError, match="requires `particle_node_type_pred`"):
        stage({"particle_clusts": particles})


def test_particle_image_builder_accepts_native_module_name() -> None:
    """Native `uses: image_particle` blocks build explicit object models."""
    owner = torch.nn.Module()
    stage = build_particle_image_stage(
        "particle_tasks",
        {
            "image_particle": {
                "objects": {"source": "explicit"},
                "encoder": {
                    "name": "cnn",
                    "num_input": 1,
                    "spatial_size": 768,
                    "filters": 4,
                    "depth": 2,
                    "reps": 1,
                },
                "heads": {"pid": 5, "primary": 2},
            }
        },
        owner,
    )
    assert stage.provides == {"particle_type_pred", "particle_primary_pred"}
    assert owner.image_particle is stage.model


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ({}, ValueError, "require.*`image` block"),
        ({"image": {"objects": []}}, TypeError, "objects.*mapping"),
        (
            {
                "image": {
                    "objects": {"source": "cluster"},
                    "encoder": {},
                    "heads": {"pid": 5},
                }
            },
            ValueError,
            "source: explicit",
        ),
        (
            {"image": {"encoder": {}, "heads": []}},
            ValueError,
            "heads.*mapping",
        ),
        (
            {
                "image": {
                    "encoder": {
                        "name": "cnn",
                        "num_input": 1,
                        "spatial_size": 4,
                        "filters": 2,
                        "depth": 1,
                        "reps": 1,
                    },
                    "heads": {"": 2},
                }
            },
            ValueError,
            "nonempty strings",
        ),
    ],
)
def test_particle_image_builder_validates_provider_contract(config, error, message):
    """Particle-image providers enforce explicit objects and named heads."""
    with pytest.raises(error, match=message):
        build_particle_image_stage("particle_tasks", config, torch.nn.Module())


def test_image_head_name_helper_rejects_malformed_heads() -> None:
    """The canonical task mapper validates direct callers as well as builders."""
    from spine.model.full_chain.providers.image import _head_names

    with pytest.raises(ValueError, match="must be a mapping"):
        _head_names({"heads": []})
    with pytest.raises(ValueError, match="nonempty strings"):
        _head_names({"heads": {1: 2}})


def test_particle_image_builder_validates_legacy_task_ownership() -> None:
    """Every legacy task delegated to images needs a corresponding head."""
    with pytest.raises(ValueError, match="missing heads: primary"):
        build_particle_image_stage(
            "particle_tasks",
            {
                "image": {
                    "encoder": {
                        "name": "cnn",
                        "num_input": 1,
                        "spatial_size": 4,
                        "filters": 2,
                        "depth": 1,
                        "reps": 1,
                    },
                    "heads": {"pid": 5},
                },
                "primary_identification": "image",
            },
            torch.nn.Module(),
        )


def test_particle_image_loss_builder_handles_optional_and_invalid_loss() -> None:
    """Particle-image supervision is optional but must be a task mapping."""
    assert build_particle_image_loss("particle_tasks", {}, torch.nn.Module()) is None
    with pytest.raises(TypeError, match="loss configuration must be a mapping"):
        build_particle_image_loss(
            "particle_tasks",
            {"loss": [], "image": {}},
            torch.nn.Module(),
        )


def test_particle_image_loss_builder_constructs_native_loss() -> None:
    """A valid image task configuration registers its native objective."""
    owner = torch.nn.Module()
    image = {
        "objects": {"source": "explicit"},
        "encoder": {
            "name": "cnn",
            "num_input": 1,
            "spatial_size": 4,
            "filters": 2,
            "depth": 1,
            "reps": 1,
        },
        "heads": {"pid": 5},
    }
    stage = build_particle_image_loss(
        "particle_tasks",
        {
            "image": image,
            "loss": {"pid": {"name": "classification"}},
        },
        owner,
    )
    assert stage is not None
    assert owner.image_particle_loss is stage.loss


def test_image_owned_task_rejects_duplicate_grappa_loss() -> None:
    """Delegated image tasks cannot retain unreachable GrapPA objectives."""
    with pytest.raises(ValueError, match="image-owned particle task.*type"):
        build_interaction_aggregation_loss(
            "interaction",
            {
                "task_modes": {"type": "image", "primary": "grappa"},
                "loss": {
                    "node_loss": {
                        "type": {"name": "class", "target": "pid"},
                        "primary": {
                            "name": "class",
                            "target": "interaction_primary",
                        },
                    }
                },
            },
            torch.nn.Module(),
        )
