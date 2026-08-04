"""Focused tests for GrapPA component factory routing."""

from spine.model.grappa.encode import EmptyClusterGlobalEncoder
from spine.model.grappa.factories import global_encoder_factory, global_loss_factory


def test_global_encoder_factory_discovers_empty_encoder() -> None:
    """Global encoders are discovered through the global class-name suffix."""
    encoder = global_encoder_factory({"name": "empty"})
    assert isinstance(encoder, EmptyClusterGlobalEncoder)


def test_global_loss_factory_routes_discovered_class(monkeypatch) -> None:
    """The reserved global-loss factory uses the common discovery mechanism."""
    import spine.model.grappa.factories as factories

    sentinel = object()
    monkeypatch.setattr(factories, "module_dict", lambda *_args, **_kwargs: {"x": 1})
    monkeypatch.setattr(
        factories,
        "instantiate",
        lambda choices, config: sentinel if choices == {"x": 1} else None,
    )
    assert global_loss_factory({"name": "future"}) is sentinel
