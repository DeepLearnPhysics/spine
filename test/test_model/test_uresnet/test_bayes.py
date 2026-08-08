"""Contracts for Bayesian UResNet segmentation and uncertainty outputs."""

import pytest

torch = pytest.importorskip("torch")

from spine.data import TensorBatch
from spine.model.uresnet.bayes import BayesianSegmentationLoss, BayesianUResNet


def model_config(mode="standard", **kwargs):
    """Return a small Bayesian UResNet configuration for focused tests."""
    return {
        "num_input": 1,
        "num_classes": 3,
        "mode": mode,
        "num_samples": 2,
        "dropout_p": 0.25,
        "filters": 4,
        "depth": 2,
        "reps": 1,
        **kwargs,
    }


def input_batch():
    """Build two sparse events with one scalar feature per voxel."""
    data = torch.tensor(
        [
            [0, 0, 0, 0, 1.0],
            [0, 1, 0, 0, 2.0],
            [1, 0, 0, 0, 3.0],
            [1, 1, 0, 0, 4.0],
        ],
        dtype=torch.float32,
    )
    return TensorBatch(
        data,
        counts=torch.tensor([2, 2]),
        has_batch_col=True,
        coord_cols=(1, 2, 3),
    )


def label_batch(values=(0, 1, 2, 1)):
    """Build semantic labels aligned with :func:`input_batch`."""
    labels = torch.tensor(values, dtype=torch.float32)
    return TensorBatch(labels, counts=torch.tensor([2, 2]))


@pytest.mark.model
@pytest.mark.parametrize("mode", ["standard", "mc_dropout", "evidential"])
def test_bayesian_uresnet_modes_return_aligned_batches(mode):
    """Each uncertainty mode must preserve voxel and batch alignment."""
    model = BayesianUResNet(model_config(mode))
    model.eval()

    result = model(input_batch())

    assert result["segmentation"].shape == (4, 3)
    assert result["segmentation"].counts.tolist() == [2, 2]
    if mode == "mc_dropout":
        assert result["softmax"].shape == (4, 3)
        assert torch.allclose(
            result["softmax"].torch_tensor().sum(dim=1),
            torch.ones(4),
        )
    elif mode == "evidential":
        evidence = result["evidence"].torch_tensor()
        concentration = result["concentration"].torch_tensor()
        assert torch.all(evidence >= 0)
        assert torch.allclose(concentration, evidence + 1.0)
        assert torch.allclose(
            result["expected_probability"].torch_tensor().sum(dim=1),
            torch.ones(4),
        )


def test_evidential_loss_is_finite_and_differentiable():
    """The evidential objective must consume evidence without shifting twice."""
    evidence = torch.rand((4, 3), requires_grad=True)
    predictions = TensorBatch(evidence, counts=torch.tensor([2, 2]))
    loss_fn = BayesianSegmentationLoss(
        model_config("evidential"),
        {"loss": "edl_sumsq", "annealing_steps": 10},
    )

    result = loss_fn(label_batch(), predictions, iteration=5)

    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert evidence.grad is not None


@pytest.mark.parametrize(
    ("mode", "loss"),
    [("standard", "edl_sumsq"), ("evidential", "ce")],
)
def test_loss_rejects_mode_mismatch(mode, loss):
    """Model output semantics and loss families cannot be mixed silently."""
    with pytest.raises(ValueError, match="requires"):
        BayesianSegmentationLoss(model_config(mode), {"loss": loss})


def test_bayesian_model_rejects_unknown_mode():
    """Unknown uncertainty behavior must fail during construction."""
    with pytest.raises(ValueError, match="Unknown Bayesian UResNet mode"):
        BayesianUResNet(model_config("mystery"))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {
                key: value
                for key, value in model_config().items()
                if key != "num_classes"
            },
            "requires `num_classes`",
        ),
        (model_config(num_classes=1), "at least two classes"),
        (model_config(num_samples=0), "num_samples"),
    ],
)
def test_bayesian_model_validates_prediction_contract(config, message):
    """Class and stochastic-sampling requirements fail during construction."""
    with pytest.raises(ValueError, match=message):
        BayesianUResNet(config)


def test_bayesian_evd_alias_and_feature_wrapper():
    """The historical evidential alias and sparse feature adapter remain valid."""
    model = BayesianUResNet(model_config("evd"))
    assert model.mode == "evidential"

    class Output:
        def aligned_features(self):
            return torch.ones((4, 2))

    wrapped = model._feature_batch(Output(), input_batch())
    assert wrapped.shape == (4, 2)
    assert wrapped.counts.tolist() == [2, 2]


def test_mc_dropout_restores_module_training_states():
    """Stochastic inference does not leak dropout training state to callers."""
    from spine.model import sparse

    model = BayesianUResNet(model_config("mc_dropout"))
    model.eval()
    dropouts = [
        module for module in model.modules() if isinstance(module, sparse.Dropout)
    ]
    assert dropouts and not any(module.training for module in dropouts)

    model(input_batch())

    assert not any(module.training for module in dropouts)


def test_bayesian_loss_rejects_unknown_conventional_options():
    """Only evidential objectives accept additional annealing configuration."""
    with pytest.raises(ValueError, match="Unexpected loss options"):
        BayesianSegmentationLoss(model_config(), {"loss": "ce", "extra": 1})


def test_bayesian_loss_validates_rows_labels_and_weights():
    """Predictions, labels, and optional weights obey one aligned contract."""
    loss_fn = BayesianSegmentationLoss(model_config(), {"loss": "ce"})
    predictions = TensorBatch(torch.zeros((4, 3)), [2, 2])

    with pytest.raises(ValueError, match="lengths do not match"):
        loss_fn(TensorBatch(torch.tensor([0.0, 1.0]), [1, 1]), predictions)
    with pytest.raises(ValueError, match="nonempty labels"):
        loss_fn(
            TensorBatch(torch.empty(0), [0]),
            TensorBatch(torch.empty((0, 3)), [0]),
        )
    with pytest.raises(ValueError, match="must lie"):
        loss_fn(label_batch((0, 1, 2, 3)), predictions)
    with pytest.raises(ValueError, match="weight and label lengths"):
        loss_fn(label_batch(), predictions, TensorBatch(torch.ones(3), [1, 2]))
    with pytest.raises(ValueError, match="must be nonnegative"):
        loss_fn(label_batch(), predictions, TensorBatch(-torch.ones(4), [2, 2]))
    with pytest.raises(ValueError, match="sum.*must be positive"):
        loss_fn(label_batch(), predictions, TensorBatch(torch.zeros(4), [2, 2]))


def test_bayesian_loss_combines_balance_and_explicit_weights():
    """Class balancing composes with caller-provided voxel weights."""
    logits = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [0.0, 2.0, 0.0]],
        requires_grad=True,
    )
    loss_fn = BayesianSegmentationLoss(
        model_config(),
        {"loss": "ce", "balance_loss": True},
    )
    result = loss_fn(
        label_batch(),
        TensorBatch(logits, [2, 2]),
        TensorBatch(torch.tensor([1.0, 2.0, 3.0, 4.0]), [2, 2]),
    )

    assert result["weights"].shape == (4,)
    assert result["accuracy"] == 1.0
    result["loss"].backward()


def test_bayesian_loss_reduces_multicomponent_voxel_losses():
    """Custom per-voxel objectives may return multiple components per row."""

    class ComponentLoss(torch.nn.Module):
        def forward(self, predictions, labels):
            return torch.ones((len(labels), 2))

    loss_fn = BayesianSegmentationLoss(model_config(), {"loss": "ce"})
    loss_fn.loss_fn = ComponentLoss()
    result = loss_fn(label_batch(), TensorBatch(torch.zeros((4, 3)), [2, 2]))
    torch.testing.assert_close(result["loss"], torch.tensor(1.0))
