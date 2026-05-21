"""Tests for training visualization helpers."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from spine.vis.train import TrainDrawer


def test_train_drawer_finds_keys_and_loads_logs(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    pd.DataFrame({"iter": [0, 1], "loss": [3.0, 2.0]}).to_csv(
        log_dir / "train-0.csv", index=False
    )
    pd.DataFrame({"iter": [2], "loss": [1.0]}).to_csv(
        log_dir / "train-2.csv", index=False
    )
    pd.DataFrame({"acc": [0.5, 0.7]}).to_csv(log_dir / "inference-2.csv", index=False)

    drawer = TrainDrawer(str(tmp_path), interactive=True)
    key, key_name = drawer.find_key({"fallback": []}, "loss:fallback")
    train_df = drawer.get_training_df(str(log_dir), ["loss"])
    val_df = drawer.get_validation_df(str(log_dir), ["accuracy:acc"])
    drawer.set_log_dir(str(log_dir))

    assert (key, key_name) == ("fallback", "loss")
    assert train_df["loss"].tolist() == [3.0, 2.0, 1.0]
    assert val_df["accuracy_mean"].tolist() == [0.6]
    assert drawer.log_dir == str(log_dir)
    with pytest.raises(KeyError, match="Could not find"):
        drawer.find_key({"loss": []}, "missing")
    with pytest.raises(FileNotFoundError, match="Found no train log"):
        drawer.get_training_df(str(tmp_path / "missing"), ["loss"])


def test_train_drawer_draws_interactive_and_matplotlib_logs(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    pd.DataFrame(
        {"iter": [0, 1], "epoch": [0.0, 0.5], "loss": [2.0, 1.0], "acc": [0.4, 0.6]}
    ).to_csv(model_dir / "train-0.csv", index=False)
    pd.DataFrame({"loss": [1.5, 1.7], "acc": [0.5, 0.7]}).to_csv(
        model_dir / "inference-2.csv", index=False
    )
    monkeypatch.setattr("spine.vis.train.iplot", lambda fig: None)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    interactive = TrainDrawer(str(tmp_path), interactive=True)
    interactive.draw("model", ["loss", "acc"], same_plot=False, iter_per_epoch=2)

    matplotlib_drawer = TrainDrawer(str(tmp_path), interactive=False)
    matplotlib_drawer.draw(
        "model",
        "loss",
        smoothing=1,
        print_min=True,
        print_max=True,
        figure_name=str(tmp_path / "training"),
    )

    assert (tmp_path / "training.png").exists()


def test_train_drawer_draws_multi_model_branches(tmp_path, monkeypatch):
    for model, offset in [("model_a", 0.0), ("model_b", 1.0)]:
        model_dir = tmp_path / model
        model_dir.mkdir()
        pd.DataFrame(
            {
                "iter": [0, 1, 2, 3],
                "epoch": [0.0, 0.5, 1.0, 1.5],
                "loss": [3.0 + offset, 2.0 + offset, 1.5 + offset, 1.0 + offset],
                "acc": [0.1 + offset, 0.2 + offset, 0.3 + offset, 0.4 + offset],
            }
        ).to_csv(model_dir / "train-0.csv", index=False)
        pd.DataFrame({"loss": [1.5, 1.7], "acc": [0.5, 0.7]}).to_csv(
            model_dir / "inference-2.csv", index=False
        )

    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    monkeypatch.setattr("spine.vis.train.iplot", lambda fig: None)

    drawer = TrainDrawer(str(tmp_path), interactive=False)
    drawer.draw(
        ["model_a", "model_b"],
        ["loss", "acc"],
        limits={"loss": [0.0, 4.0], "acc": [0.0, 2.0]},
        model_name={"model_a": "A", "model_b": "B"},
        metric_name={"loss": "Loss", "acc": "Accuracy"},
        same_plot=False,
        max_iter=3,
        step=2,
        smoothing=2,
    )

    interactive = TrainDrawer(str(tmp_path), interactive=True)
    interactive.draw(
        "model_a",
        ["loss", "acc"],
        limits={"loss": [0.0, 5.0], "acc": [0.0, 1.0]},
        same_plot=False,
        iter_per_epoch=2,
    )
    interactive.draw(
        ["model_a", "model_b"],
        ["loss", "acc"],
        same_plot=True,
        iter_per_epoch=2,
    )
    interactive.draw(
        ["model_a", "model_b"],
        "loss",
        same_plot=True,
        iter_per_epoch=2,
    )


def test_train_drawer_validates_display_names_and_empty_chunks(tmp_path, monkeypatch):
    drawer = TrainDrawer(str(tmp_path), interactive=True)

    with pytest.raises(ValueError, match="model_name"):
        drawer.draw(["a", "b"], "loss", model_name="model")
    with pytest.raises(ValueError, match="metric_name"):
        drawer.draw("a", ["loss", "acc"], metric_name="metric")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    pd.DataFrame({"iter": [0, 1], "epoch": [0.0, 1.0], "loss": [2.0, 1.0]}).to_csv(
        model_dir / "train-0.csv", index=False
    )
    pd.DataFrame(columns=["iter", "epoch", "loss"]).to_csv(
        model_dir / "train-1.csv", index=False
    )
    monkeypatch.setattr("spine.vis.train.iplot", lambda fig: None)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    interactive = TrainDrawer(str(tmp_path), interactive=True)
    interactive.draw(
        "model",
        "loss",
        limits=[0.0, 3.0],
        model_name="Model",
        metric_name="Loss",
        same_plot=True,
    )

    matplotlib_drawer = TrainDrawer(str(tmp_path), interactive=False)
    matplotlib_drawer.draw(
        "model",
        "loss",
        limits=[0.0, 3.0],
        model_name="Model",
        metric_name="Loss",
        same_plot=True,
    )
    with matplotlib.rc_context():
        paper_drawer = TrainDrawer(str(tmp_path), interactive=False, paper=True)
        assert paper_drawer.linewidth == 0.5
    plt.close("all")
