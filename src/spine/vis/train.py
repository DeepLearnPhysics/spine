"""Tools to monitor training/validation processes."""

import glob
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import colors as pcolors
from plotly import graph_objs as go
from plotly import subplots as psubplots
from plotly.offline import iplot

from .layout import PLOTLY_COLORS_TUPLE, apply_latex_style, color_rgba

__all__ = ["TrainDrawer"]


class TrainDrawer:
    """Class which centralizes function used to monitor a training process."""

    def __init__(
        self,
        log_dir,
        interactive=True,
        paper=False,
        alpha=0.5,
        train_prefix="train",
        val_prefix="inference",
        separator=":",
    ):
        """Initialize the drawer.

        Parameters
        ----------
        log_dir, str
            Path to the parent directory of all the log files
        interactive : bool, default True
            If `True`, use plotly to draw the training/validation curve
        paper : bool, default False
            If `True`, format the figure for a paper, using latext style
        alpha : float, default 0.5
            Opacity of the traces
        train_prefix : str, default 'train'
            Log name prefix shared between training logs
        val_prefix : str, default 'inference'
            Log name prefix shared between validation logs
        separator : str, default ':'
            Character used to separate the acceptable metric names in the
            metric parameter
        """
        # Store the path to the main log directory
        self.log_dir = log_dir
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self.separator = separator

        # Initialize the style
        self.interactive = interactive
        if interactive:
            self.layout = None
            self.cr_char = "<br>"
            self.initialize_plotly()
        else:
            self.linwidth = None
            self.markersize = None
            self.cr_char = "\n"
            self.initialize_matplotlib(paper)

        # Initialize the colors
        self.alpha = alpha
        if interactive:
            self.colors = pcolors.color_parser(
                PLOTLY_COLORS_TUPLE, partial(color_rgba, alpha=alpha)
            )
        else:
            self.colors = PLOTLY_COLORS_TUPLE

    def initialize_plotly(self):
        """Initialize the style parameters for plotly."""
        font = {"size": 20}
        axis_base = {"tickfont": font, "linecolor": "black", "mirror": True}
        self.layout = go.Layout(
            template="plotly_white",
            width=1000,
            height=500,
            margin={"t": 20, "b": 20, "l": 20, "r": 20},
            xaxis={"title": {"text": "Epochs", "font": font}, **axis_base},
            yaxis={"title": {"text": "Metric", "font": font}, **axis_base},
            legend={"font": font, "tracegroupgap": 1},
        )

    def initialize_matplotlib(self, paper):
        """Initialize the style parameters for matplotlib.

        paper : bool, default False
            If `True`, format the figure for a paper, using latext style
        """
        # Define a matplotlib layout using seaborn
        if paper:
            # Define the style for latex
            apply_latex_style()
            self.linewidth = 0.5
            self.markersize = 1

        else:
            # Define the style to be displayed in a notebook
            sns.set(rc={"figure.figsize": (9, 6)}, context="notebook", font_scale=2)
            sns.set_style("white")
            sns.set_style(rc={"axes.grid": True})
            self.linewidth = 2
            self.markersize = 10

    def set_log_dir(self, log_dir):
        """Simply reset the base log directory to another one.

        Parameters
        ----------
        log_dir, str
            Path to the parent directory of all the log files
        """
        self.log_dir = log_dir

    def draw(
        self,
        model,
        metric,
        limits=None,
        model_name=None,
        metric_name=None,
        max_iter=None,
        step=None,
        smoothing=None,
        iter_per_epoch=None,
        print_min=False,
        print_max=False,
        same_plot=True,
        leg_ncols=1,
        figure_name=None,
    ):
        """Finds all training and validation log files inside the specified
        directory and draws an evolution plot of the requested quantities.

        Parameters
        ----------
        model : Union[str, List[str]]
            Model (folder) / list of models names under the main directory
        metric : Union[str, List[str]]
            Metric / list of metrics to draw
        limits : Union[List[float], Dict[str, List[float]]], optional
            List of y boundaries for the plot. If specified as a dictionary,
            can be used to specify different boundaries for each metric.
        model_name : Union[str, Dict[str, str]], optional
            Name of the model as displayed in the legend. If there are multiple
            models, provide a dictionary which maps each model onto a name
        metric_name : Union[str, Dict[str, str]], optional
            Name of the metric as displayed in the legend. If there are multiple
            metrics, provide a dictionary which maps each metric onto a name
        max_iter : int, optional
            Maximum number of iterations to include in the plot
        step : int, optional
            Step between two consecutive iterations that are represented
        smoothing : int, optional
            Number of iteration over which to average the metric values
        iter_per_epoch : float, optional
            Number of iterations to complete an dataset epoch
        same_plot : bool, default True
            If `True`, draw all metrics on the same plot
        leg_ncols : int, default 1
            Number of columns in the legend
        figure_name : str, optional
            Name of the figure. If specified, figure is saved
        """
        # If the model/metric is a single string, nest it
        if isinstance(model, str):
            model = [model]
        if isinstance(metric, str):
            metric = [metric]

        # Make sure each model is given a name
        model_name = model_name or {}
        if isinstance(model_name, str):
            assert len(model) == 1, (
                "Should provide a single `model_name` if there is a "
                "single `model` to be represented."
            )
            model_name = {model[0]: model_name}
        else:
            for m in model:
                if m not in model_name:
                    model_name[m] = m

        # Make sure each metric is given a name
        metric_name = metric_name or {}
        if isinstance(metric_name, str):
            assert len(metric) == 1, (
                "Should provide a single `metric_name` if there is a "
                "single `metric` to be represented."
            )
            metric_name = {metric[0]: metric_name}
        else:
            for m in metric:
                if m not in metric_name:
                    metric_name[m] = m.split(self.separator)[0]

        # Make sure that each metric is given a range, if specified
        if limits is not None and not isinstance(limits, dict):
            limits = {m: limits for m in metric}

        # If using plotly, reset the layout and the trace list
        if self.interactive:
            self.initialize_plotly()
            traces = []

        # Initialize the figure to draw the training/validation metrics on
        if not same_plot and len(metric) > 1:
            if not self.interactive:
                # Prepare the matplotlib sublots
                fig, axes = plt.subplots(len(metric), sharex=True)
                fig.subplots_adjust(hspace=0)

                # Set all the axes with the same color
                for axis in axes:
                    axis.set_facecolor("white")

            else:
                # Prepare the plotly subplots
                fig = psubplots.make_subplots(
                    rows=len(metric), shared_xaxes=True, vertical_spacing=0
                )

                # Loop over the metrics
                for i, m in enumerate(metric):
                    # Set all axes formatting the same
                    if i > 0:
                        self.layout[f"xaxis{i+1}"] = self.layout["xaxis"]
                        self.layout[f"yaxis{i+1}"] = self.layout["yaxis"]

                    # Set all x axis empty apart from the bottom one
                    xtitle = None if i != len(metric) - 1 else "Epochs"
                    self.layout[f"xaxis{i+1}"]["title"]["text"] = xtitle

                    # Set all y axis the same
                    self.layout[f"yaxis{i+1}"]["title"]["text"] = metric_name[m]

                    # If there are limits specified,
                    if limits is not None and m in limits:
                        self.layout[f"yaxis{i+1}"]["range"] = limits[m]

                fig.update_layout(self.layout)

        elif self.interactive:
            # Set the limit for the shared y axis
            if limits is not None and metric[0] in limits:
                self.layout["yaxis"]["range"] = limits[metric[0]]

            # If there is a single metric, rename the y axis with its name
            if len(metric) == 1:
                self.layout["yaxis"]["title"]["text"] = metric_name[metric[0]]

            # If all the metrics are on a shared plot, give the legend a title
            if same_plot and len(model) == 1:
                self.layout["legend"]["title"] = model_name[model[0]]

            # Initialize the plotly figure
            fig = go.Figure(layout=self.layout)

        # Get the DataFrames for the requested models/metrics
        dfs, val_dfs, colors, draw_val = {}, {}, {}, {}
        for i, key in enumerate(model):
            log_subdir = self.log_dir + key
            dfs[key] = self.get_training_df(log_subdir, metric)
            val_dfs[key] = self.get_validation_df(log_subdir, metric)
            draw_val[key] = bool(len(val_dfs[key]["iter"]))
            colors[key] = self.colors[i % len(self.colors)]

        # Loop over the requested metrics, append a trace
        for i, metric_list in enumerate(metric):
            # Get a trace per training campaign
            for j, key in enumerate(dfs.keys()):
                # Get the necessary data
                iter_t, epoch_t = dfs[key]["iter"], dfs[key]["epoch"]
                metric_key, metric_label = self.find_key(dfs[key], metric_list)
                metric_t = dfs[key][metric_key]

                # If validation points are available, fetch the validation data
                if draw_val[key]:
                    # Get the necessary data
                    iter_v = val_dfs[key]["iter"]
                    metric_v_mean = val_dfs[key][metric_label + "_mean"]
                    metric_v_err = val_dfs[key][metric_label + "_err"]

                    # Convert the iteration number of each file to an epoch value
                    if iter_per_epoch is None:
                        epoch_v = [epoch_t[iter_t == it] for it in iter_v]
                        mask = np.where(np.array([len(e) for e in epoch_v]) == 1)[0]
                        epoch_v = [float(epoch_v[i].iloc[0]) for i in mask]
                        iter_v = iter_v[mask]
                        metric_v_mean = metric_v_mean[mask]
                        metric_v_err = metric_v_err[mask]
                    else:
                        epoch_v = iter_v / iter_per_epoch

                    # If requested, restrict the number of points to a range
                    if max_iter is not None:
                        mask_val = np.where(iter_v < max_iter)[0]
                        iter_v = iter_v[mask_val]
                        epoch_v = epoch_v[mask_val]
                        metric_v_mean = metric_v_mean[mask_val]
                        metric_v_err = metric_v_err[mask_val]

                # If requested, restrict the data to a certain range
                if max_iter is not None:
                    epoch_t = epoch_t[:max_iter]
                    metric_t = metric_t[:max_iter]

                # If requested, smooth out the curve by taking a rolling average
                if smoothing is not None and smoothing > 1:
                    metric_t = metric_t.rolling(
                        smoothing, min_periods=1, center=True
                    ).mean()

                # If requested, do not display all the iterations
                if step is not None and step > 1:
                    epoch_t = epoch_t[::step]
                    metric_t = metric_t[::step]

                # Pick a label for this specific model/metric pair
                if not same_plot:
                    label = model_name[key]
                else:
                    if len(model) == 1:
                        label = metric_name[metric_list]
                    elif len(metric) == 1:
                        label = model_name[key]
                    else:
                        label = f"{metric_name[metric_list]} ({model_name[key]})"
                    if print_min and draw_val[key]:
                        min_it = iter_v[np.argmin(metric_v_mean)]
                        label += f"{self.cr_char}Min: {min_it:d}"
                    if print_max and draw_val[key]:
                        max_it = iter_v[np.argmax(metric_v_mean)]
                        label += f"{self.cr_char}Max: {max_it:d}"

                # Fetch the appropriate color
                idx = i * len(model) + j
                if same_plot:
                    color = self.colors[idx % len(self.colors)]
                else:
                    color = colors[key]

                # Prepare the traces
                if not self.interactive:
                    # Add a trace for the training curve
                    axis = plt if same_plot else axes[i]
                    axis.plot(
                        epoch_t,
                        metric_t,
                        label=label,
                        color=color,
                        alpha=self.alpha,
                        linewidth=self.linewidth,
                    )

                    # Add a trace for the validation points
                    if draw_val[key]:
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
                    # Add a trace for the training curve
                    legendgroup = f"group{idx}"
                    showlegend = same_plot or not i
                    traces += [
                        go.Scatter(
                            x=epoch_t,
                            y=metric_t,
                            name=label,
                            line={"color": color},
                            legendgroup=legendgroup,
                            showlegend=showlegend,
                        )
                    ]

                    # Add a trace for the validation points
                    if draw_val[key]:
                        hovertext = [f"(Iteration: {it:d})" for it in iter_v]
                        traces += [
                            go.Scatter(
                                x=epoch_v,
                                y=metric_v_mean,
                                error_y_array=metric_v_err,
                                mode="markers",
                                hovertext=hovertext,
                                marker={"color": color},
                                legendgroup=legendgroup,
                                showlegend=False,
                            )
                        ]

        if not self.interactive:
            if not same_plot:
                for i, metric_key in enumerate(metric):
                    axes[i].set_xlabel("Epochs")
                    axes[i].set_ylabel(metric_name[metric_key])
                    if limits is not None and metric_key in limits:
                        axes[i].set_ylim(limits[metric_key])
                axes[0].legend(ncol=leg_ncols)

            else:
                plt.xlabel("Epochs")
                ylabel = metric_name[metric[0]]
                plt.ylabel(ylabel if len(metric) == 1 else "Metric")
                plt.gca().set_ylim(limits[metric[0]])
                legend_title = model_name[model[0]] if len(model) == 1 else None
                plt.legend(ncol=leg_ncols, title=legend_title)

            if figure_name:
                plt.savefig(f"{figure_name}.png", bbox_inches="tight")
                plt.savefig(f"{figure_name}.pdf", bbox_inches="tight")

            plt.show()

        else:
            if not same_plot:
                n_mod, n_met = len(model), len(metric)
                mult = np.sum([2**val for val in draw_val.values()])
                step = 1.0 / mult
                rows = list(np.arange(n_met, step=step).astype(int) + 1)
                cols = list(np.ones(mult * n_met, dtype=int))
                fig.add_traces(traces, rows=rows, cols=cols)

            else:
                fig.add_traces(traces)

            iplot(fig)

    def get_training_df(self, log_dir, keys):
        """Finds all training log files inside the specified directory and
        concatenates them. If the range of iterations overlap, keep only that
        from the file started further in the training.

        Assumes that the formatting of the log file names is of the form
        `self.train_prefix-x.csv`, with `x` the number of iterations.

        Parameters
        ----------
        log_dir : str
            Path to the directory that contains the training log files
        keys : List[str]
            List of quantities to extract from the log files

        Returns
        -------
        pd.DataFrame
            Combined training log data
        """
        # Get all the log files that fit the pattern
        log_files = glob.glob(f"{log_dir}/{self.train_prefix}*")
        if not log_files:
            raise FileNotFoundError(
                f"Found no train log with prefix '{self.train_prefix}' "
                f"under {log_dir}."
            )

        # Find the iteration at which each successsive log file picks up to
        # avoid duplicate values at a given iteration
        start_iter = np.zeros(len(log_files), dtype=np.int64)
        for i, f in enumerate(log_files):
            start_iter[i] = int(f.split("-")[-1].split(".csv")[0])
        order = np.argsort(start_iter)
        end_points = np.append(start_iter[order], 1e12)

        # Loop over the log files, concatenate them
        log_dfs = []
        for i, f in enumerate(np.array(log_files)[order]):
            df = pd.read_csv(f, nrows=end_points[i + 1] - end_points[i])
            if len(df) == 0:
                continue
            for key_list in keys:
                key, key_name = self.find_key(df, key_list)
                df[key_name] = df[key]
            log_dfs.append(df)

        return pd.concat(log_dfs, sort=True)

    def get_validation_df(self, log_dir, keys):
        """Finds all validation log files inside the specified directory and
        build a single dataframe out of them. It returns the mean and std of
        the requested keys for each file.

        Assumes that the formatting of the log file names is of the form
        `self.val_prefix-x.csv`, with `x` the number of iterations.

        The key list allows for `:`-separated names, in case separate files
        use different names for the same quantity.

        Parameters
        ----------
        log_dir : str
            Path to the directory that contains the validation log files
        keys : List[str]
            List of quantities to extract from the log files

        Returns
        -------
        pd.DataFrame
            Combined validation log data
        """
        # Initialize a dictionary
        val_data = {"iter": []}
        for key in keys:
            key_name = key.split(":")[0]
            val_data[key_name + "_mean"] = []
            val_data[key_name + "_err"] = []

        # Loop over validation log files
        log_files = np.array(glob.glob(f"{log_dir}/{self.val_prefix}*"))
        for log_file in log_files:
            df = pd.read_csv(log_file)
            it = int(log_file.split("/")[-1].split("-")[-1].split(".")[0])
            val_data["iter"].append(it - 1)
            for key_list in keys:
                key, key_name = self.find_key(df, key_list)
                mean, err = df[key].mean(), df[key].std() / np.sqrt(len(df[key]))
                val_data[f"{key_name}_mean"].append(mean)
                val_data[f"{key_name}_err"].append(err)

        args = np.argsort(val_data["iter"])
        for key, val in val_data.items():
            val_data[key] = np.array(val)[args]

        return pd.DataFrame(val_data)

    def find_key(self, df, key_list):
        """Checks if a :class:`pd.DataFrame` contains any of the keys listed
        in a character-separated string.

        If multiple keys match, pick the first one.

        Parameters
        ----------
        df : Union[pd.DataFrame, dict]
            Pandas dataframe or dictionary containing data
        key_list : str
            Character-separated list of acceptable names for an attribute

        Returns
        -------
        key : str
            First data key which matches one of the keys in the list
        key_name : str
            First key in the list of acceptale names
        """
        key_list = key_list.split(self.separator)
        key_name = key_list[0]
        key_found = np.array([k in df.keys() for k in key_list])
        if not np.any(key_found):
            raise KeyError("Could not find any of the keys provided:", key_list)
        key = key_list[np.where(key_found)[0][0]]

        return key, key_name
