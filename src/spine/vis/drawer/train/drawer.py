"""Drawer class for training and validation metric histories."""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly import colors as pcolors
from plotly import graph_objs as go
from plotly import subplots as psubplots
from plotly.offline import iplot

from ...layout import PLOTLY_COLORS_TUPLE, color_rgba
from .io import find_key, get_training_df, get_validation_df
from .style import initialize_matplotlib_style, initialize_plotly_layout

__all__ = ["TrainDrawer"]


class TrainDrawer:
    """Centralize training and validation log loading and visualization."""

    def __init__(
        self,
        log_dir: str,
        interactive: bool = True,
        paper: bool = False,
        alpha: float = 0.5,
        train_prefix: str = "train",
        val_prefix: str = "inference",
        separator: str = ":",
    ) -> None:
        """Initialize the training-history drawer.

        Parameters
        ----------
        log_dir : str
            Path to the parent directory containing per-model log folders.
        interactive : bool, default True
            If ``True``, draw with Plotly. Otherwise use Matplotlib.
        paper : bool, default False
            If ``True`` and ``interactive`` is ``False``, configure a compact
            LaTeX-style Matplotlib theme.
        alpha : float, default 0.5
            Opacity of the plotted traces.
        train_prefix : str, default ``"train"``
            Filename prefix identifying training logs.
        val_prefix : str, default ``"inference"``
            Filename prefix identifying validation logs.
        separator : str, default ``":"``
            Character used to separate acceptable metric aliases.
        """
        self.log_dir = log_dir
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self.separator = separator
        self.layout: go.Layout | dict[str, Any]
        self.colors: list[Any]

        # Initialize backend-specific layout and line-break formatting.
        self.interactive = interactive
        if interactive:
            self.layout = initialize_plotly_layout()
            self.cr_char = "<br>"
        else:
            self.layout = {}
            self.cr_char = "\n"
            self.linewidth, self.markersize = initialize_matplotlib_style(paper)

        self.alpha = alpha
        if interactive:
            parsed_colors = (
                pcolors.color_parser(
                    PLOTLY_COLORS_TUPLE, partial(color_rgba, alpha=alpha)
                )
                or []
            )
            self.colors = [str(color) for color in parsed_colors]
        else:
            self.colors = list(PLOTLY_COLORS_TUPLE)

    def initialize_plotly(self) -> None:
        """Reset the Plotly layout to the default training-curve style."""
        self.layout = initialize_plotly_layout()

    def initialize_matplotlib(self, paper: bool) -> None:
        """Configure matplotlib styling and cache line and marker sizes.

        Parameters
        ----------
        paper : bool
            If ``True``, configure the compact LaTeX-style Matplotlib theme.
        """
        self.linewidth, self.markersize = initialize_matplotlib_style(paper)

    def set_log_dir(self, log_dir: str) -> None:
        """Reset the base log directory.

        Parameters
        ----------
        log_dir : str
            New parent directory containing per-model log folders.
        """
        self.log_dir = log_dir

    def get_training_df(self, log_dir: str, keys: list[str]) -> pd.DataFrame:
        """Load and concatenate segmented training logs.

        Parameters
        ----------
        log_dir : str
            Directory containing one model's training logs.
        keys : List[str]
            Metrics to extract from the logs.

        Returns
        -------
        pd.DataFrame
            Concatenated training logs with canonicalized metric columns.
        """
        return get_training_df(log_dir, keys, self.train_prefix, self.separator)

    def get_validation_df(self, log_dir: str, keys: list[str]) -> pd.DataFrame:
        """Summarize validation logs into means and errors per checkpoint.

        Parameters
        ----------
        log_dir : str
            Directory containing one model's validation logs.
        keys : List[str]
            Metrics to extract from the logs.

        Returns
        -------
        pd.DataFrame
            Validation summary with one row per checkpoint iteration.
        """
        return get_validation_df(log_dir, keys, self.val_prefix, self.separator)

    def find_key(
        self, df: pd.DataFrame | dict[str, Any], key_list: str
    ) -> tuple[str, str]:
        """Resolve the first available metric name from a separator list.

        Parameters
        ----------
        df : Union[pd.DataFrame, Dict[str, Any]]
            Data container whose keys are searched.
        key_list : str
            Separator-delimited list of acceptable metric names.

        Returns
        -------
        Tuple[str, str]
            Matched key and canonical display key name.
        """
        return find_key(df, key_list, self.separator)

    def draw(
        self,
        model: str | list[str],
        metric: str | list[str],
        limits: list[float] | dict[str, list[float]] | None = None,
        model_name: str | dict[str, str] | None = None,
        metric_name: str | dict[str, str] | None = None,
        max_iter: int | None = None,
        step: int | None = None,
        smoothing: int | None = None,
        iter_per_epoch: float | None = None,
        print_min: bool = False,
        print_max: bool = False,
        same_plot: bool = True,
        leg_ncols: int = 1,
        figure_name: str | None = None,
    ) -> None:
        """Draw training and validation metric histories for one or more models.

        Parameters
        ----------
        model : Union[str, List[str]]
            Model folder name or names under ``self.log_dir``.
        metric : Union[str, List[str]]
            Metric name or names to display.
        limits : Union[List[float], Dict[str, List[float]]], optional
            Y-axis limits shared across metrics or provided per metric.
        model_name : Union[str, Dict[str, str]], optional
            Display name or names for the requested models.
        metric_name : Union[str, Dict[str, str]], optional
            Display name or names for the requested metrics.
        max_iter : int, optional
            Maximum training iteration to include.
        step : int, optional
            Subsampling step for training iterations.
        smoothing : int, optional
            Rolling-average window size for training metrics.
        iter_per_epoch : float, optional
            Conversion factor from iterations to epochs for validation points.
        print_min : bool, default False
            If ``True``, append the validation minimum iteration to legend labels.
        print_max : bool, default False
            If ``True``, append the validation maximum iteration to legend labels.
        same_plot : bool, default True
            If ``True``, draw all curves on one plot.
        leg_ncols : int, default 1
            Number of legend columns for Matplotlib figures.
        figure_name : str, optional
            Base filename used to save Matplotlib figures.
        """
        if isinstance(model, str):
            model = [model]
        if isinstance(metric, str):
            metric = [metric]
        interactive = self.interactive

        # Normalize the model and metric display-name mappings so the plotting
        # code can always rely on dictionary lookups.
        model_name = model_name or {}
        if isinstance(model_name, str):
            if len(model) != 1:
                raise ValueError(
                    "Should provide a single `model_name` if there is a "
                    "single `model` to be represented."
                )
            model_name = {model[0]: model_name}
        else:
            for model_key in model:
                if model_key not in model_name:
                    model_name[model_key] = model_key

        metric_name = metric_name or {}
        if isinstance(metric_name, str):
            if len(metric) != 1:
                raise ValueError(
                    "Should provide a single `metric_name` if there is a "
                    "single `metric` to be represented."
                )
            metric_name = {metric[0]: metric_name}
        else:
            for metric_key in metric:
                if metric_key not in metric_name:
                    metric_name[metric_key] = metric_key.split(self.separator)[0]

        if limits is not None and not isinstance(limits, dict):
            limits = {metric_key: limits for metric_key in metric}

        traces: list[go.Scatter] = []
        trace_rows: list[int] = []
        axes: list[Any] = []
        fig: go.Figure | None = None
        uses_subplots = not same_plot and len(metric) > 1

        if interactive:
            self.initialize_plotly()

        # Build the plotting canvas. Multi-metric non-shared plots require a
        # subplot container, while the other cases can use a single figure.
        if uses_subplots:
            if not interactive:
                figure, axes_raw = plt.subplots(len(metric), sharex=True)
                figure.subplots_adjust(hspace=0)
                axes = list(np.atleast_1d(axes_raw))
                for axis in axes:
                    axis.set_facecolor("white")
            else:
                fig = psubplots.make_subplots(
                    rows=len(metric), shared_xaxes=True, vertical_spacing=0
                )
                fig.update_layout(self.layout)
                for i, metric_key in enumerate(metric, start=1):
                    fig.update_xaxes(
                        title_text="Epochs" if i == len(metric) else None,
                        row=i,
                        col=1,
                    )
                    axis_kwargs: dict[str, Any] = {
                        "title_text": metric_name[metric_key]
                    }
                    if limits is not None and metric_key in limits:
                        axis_kwargs["range"] = tuple(limits[metric_key])
                    fig.update_yaxes(row=i, col=1, **axis_kwargs)

        elif interactive:
            fig = go.Figure(layout=cast(go.Layout, self.layout))
            if limits is not None and metric[0] in limits:
                fig.update_yaxes(range=tuple(limits[metric[0]]))

            if len(metric) == 1:
                fig.update_yaxes(title_text=metric_name[metric[0]])

            if same_plot and len(model) == 1:
                fig.update_layout(legend_title_text=model_name[model[0]])

        dfs: dict[str, pd.DataFrame] = {}
        val_dfs: dict[str, pd.DataFrame] = {}
        colors: dict[str, Any] = {}
        draw_val: dict[str, bool] = {}
        for i, key in enumerate(model):
            log_subdir = f"{self.log_dir}/{key}"
            dfs[key] = self.get_training_df(log_subdir, metric)
            val_dfs[key] = self.get_validation_df(log_subdir, metric)
            draw_val[key] = bool(len(val_dfs[key]["iter"]))
            colors[key] = self.colors[i % len(self.colors)]

        # Loop over metric/model pairs and add the corresponding training and
        # validation traces to the active backend.
        for i, metric_list in enumerate(metric):
            for j, key in enumerate(dfs.keys()):
                iter_t, epoch_t = dfs[key]["iter"], dfs[key]["epoch"]
                metric_key, metric_label = self.find_key(dfs[key], metric_list)
                metric_t = dfs[key][metric_key]
                iter_v: np.ndarray[Any, Any] | None = None
                epoch_v: np.ndarray[Any, Any] | None = None
                metric_v_mean: np.ndarray[Any, Any] | None = None
                metric_v_err: np.ndarray[Any, Any] | None = None

                if draw_val[key]:
                    iter_v = val_dfs[key]["iter"].to_numpy()
                    metric_v_mean = val_dfs[key][f"{metric_label}_mean"].to_numpy()
                    metric_v_err = val_dfs[key][f"{metric_label}_err"].to_numpy()

                    if iter_per_epoch is None:
                        epoch_matches = [epoch_t[iter_t == it] for it in iter_v]
                        mask = np.where(
                            np.array([len(epoch) for epoch in epoch_matches]) == 1
                        )[0]
                        epoch_v = np.asarray(
                            [float(epoch_matches[idx].iloc[0]) for idx in mask]
                        )
                        iter_v = iter_v[mask]
                        metric_v_mean = metric_v_mean[mask]
                        metric_v_err = metric_v_err[mask]
                    else:
                        epoch_v = iter_v / iter_per_epoch

                    # Apply the same iteration cut to the validation points so
                    # training and validation stay visually aligned.
                    if max_iter is not None:
                        mask_val = np.where(iter_v < max_iter)[0]
                        iter_v = iter_v[mask_val]
                        epoch_v = epoch_v[mask_val]
                        metric_v_mean = metric_v_mean[mask_val]
                        metric_v_err = metric_v_err[mask_val]

                has_validation = (
                    iter_v is not None
                    and epoch_v is not None
                    and metric_v_mean is not None
                    and metric_v_err is not None
                    and len(iter_v) > 0
                )

                if max_iter is not None:
                    epoch_t = epoch_t[:max_iter]
                    metric_t = metric_t[:max_iter]

                if smoothing is not None and smoothing > 1:
                    metric_t = metric_t.rolling(
                        smoothing, min_periods=1, center=True
                    ).mean()

                if step is not None and step > 1:
                    epoch_t = epoch_t[::step]
                    metric_t = metric_t[::step]

                # Resolve the legend label based on whether curves are grouped
                # by model, metric, or both on the active plot.
                if not same_plot:
                    label = model_name[key]
                else:
                    if len(model) == 1:
                        label = metric_name[metric_list]
                    elif len(metric) == 1:
                        label = model_name[key]
                    else:
                        label = f"{metric_name[metric_list]} ({model_name[key]})"
                    if print_min and has_validation:
                        assert iter_v is not None and metric_v_mean is not None
                        min_it = iter_v[np.argmin(metric_v_mean)]
                        label += f"{self.cr_char}Min: {min_it:d}"
                    if print_max and has_validation:
                        assert iter_v is not None and metric_v_mean is not None
                        max_it = iter_v[np.argmax(metric_v_mean)]
                        label += f"{self.cr_char}Max: {max_it:d}"

                idx = i * len(model) + j
                color = (
                    self.colors[idx % len(self.colors)] if same_plot else colors[key]
                )

                if not interactive:
                    axis = plt if not uses_subplots else axes[i]
                    axis.plot(
                        epoch_t,
                        metric_t,
                        label=label,
                        color=color,
                        alpha=self.alpha,
                        linewidth=self.linewidth,
                    )

                    if has_validation:
                        assert (
                            epoch_v is not None
                            and metric_v_mean is not None
                            and metric_v_err is not None
                        )
                        axis.errorbar(
                            epoch_v,
                            metric_v_mean,
                            yerr=metric_v_err,
                            fmt=".",
                            color=color,
                            linewidth=self.linewidth,
                            markersize=self.markersize,
                        )

                else:
                    legendgroup = f"group{idx}"
                    showlegend = same_plot or not i
                    traces.append(
                        go.Scatter(
                            x=epoch_t,
                            y=metric_t,
                            name=label,
                            line={"color": color},
                            legendgroup=legendgroup,
                            showlegend=showlegend,
                        )
                    )
                    if uses_subplots:
                        trace_rows.append(i + 1)

                    if has_validation:
                        assert (
                            iter_v is not None
                            and epoch_v is not None
                            and metric_v_mean is not None
                            and metric_v_err is not None
                        )
                        hovertext = [f"(Iteration: {it:d})" for it in iter_v]
                        traces.append(
                            go.Scatter(
                                x=epoch_v,
                                y=metric_v_mean,
                                error_y={"array": metric_v_err},
                                mode="markers",
                                hovertext=hovertext,
                                marker={"color": color},
                                legendgroup=legendgroup,
                                showlegend=False,
                            )
                        )
                        if uses_subplots:
                            trace_rows.append(i + 1)

        if not interactive:
            if uses_subplots:
                for i, metric_key in enumerate(metric):
                    axes[i].set_xlabel("Epochs")
                    axes[i].set_ylabel(metric_name[metric_key])
                    if limits is not None and metric_key in limits:
                        axes[i].set_ylim(*limits[metric_key])
                axes[0].legend(ncol=leg_ncols)
            else:
                plt.xlabel("Epochs")
                ylabel = metric_name[metric[0]]
                plt.ylabel(ylabel if len(metric) == 1 else "Metric")
                if limits is not None and metric[0] in limits:
                    plt.gca().set_ylim(*limits[metric[0]])
                legend_title = model_name[model[0]] if len(model) == 1 else None
                plt.legend(
                    ncol=leg_ncols,
                    title=legend_title,
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                )

            if figure_name:
                plt.savefig(f"{figure_name}.png", bbox_inches="tight")
                plt.savefig(f"{figure_name}.pdf", bbox_inches="tight")

            plt.show()
            return

        # Plotly subplot insertion needs explicit row and column indices when
        # multiple metric panels are drawn.
        assert fig is not None
        if uses_subplots:
            for trace, row in zip(traces, trace_rows):
                fig.add_trace(trace, row=row, col=1)
        else:
            for trace in traces:
                fig.add_trace(trace)

        iplot(fig)
