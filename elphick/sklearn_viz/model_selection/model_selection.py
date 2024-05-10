import logging
from typing import Union, Optional, Dict, List, Callable, Tuple

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
from plotly import colors
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from elphick.sklearn_viz.model_selection.cross_validation import CrossValidatorBase


def subplot_index(idx: int, col_wrap: int) -> Tuple[int, int]:
    col: int = int(idx % col_wrap + 1)
    row: int = int(np.floor(idx / col_wrap) + 1)
    return row, col


def plot_model_selection(estimators: Union[sklearn.base.RegressorMixin, sklearn.base.ClassifierMixin, Dict],
                         datasets: Union[pd.DataFrame, Dict],
                         target: str,
                         pre_processor: Optional[Pipeline] = None,
                         k_folds: int = 10,
                         title: Optional[str] = None) -> go.Figure:
    """

    Args:
            estimators: sklearn estimator or a Dict of algorithms to cross-validate, keyed by string name/code.
            datasets: pandas DataFrame or a dict of DataFrames, keyed by string name/code.
            target: target column
            pre_processor: Optional pipeline used to pre-process the datasets.
            k_folds: The number of cross validation folds.
            title: Optional plot title

    Returns:
        a plotly GraphObjects.Figure

    """

    return ModelSelection(estimators=estimators, datasets=datasets, target=target, pre_processor=pre_processor,
                          k_folds=k_folds).plot(title=title)


class ModelSelection(CrossValidatorBase):
    def __init__(self,
                 estimators: Union[BaseEstimator, Dict[str, BaseEstimator]],
                 datasets: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 target: str,
                 pre_processor: Optional[Pipeline] = None,
                 k_folds: int = 10,
                 scorer: Optional[Union[str, Callable]] = None,
                 metrics: Optional[Dict[str, Callable]] = None,
                 group: Optional[pd.Series] = None,
                 random_state: Optional[int] = None):
        """

        Args:
            estimators: sklearn estimator or a Dict of algorithms to cross-validate, keyed by string name/code.
            datasets: pandas DataFrame or a dict of DataFrames, keyed by string name/code.
            target: target column
            pre_processor: Optional pipeline used to pre-process the datasets.
            k_folds: The number of cross validation folds.
            scorer: Optional callable scorers which the model will be fitted using
            metrics: Optional Dict of callable metrics to calculate post-fitting
            group: Optional group variable by which to partition/group metrics.  The same group applies across all
             datasets, so is more useful when testing different algorithms.
            random_state: Optional random seed
        """

        self._logger = logging.getLogger(name=__class__.__name__)

        super().__init__(estimators=estimators, datasets=datasets, target=target, pre_processor=pre_processor,
                         cv=k_folds, scorer=scorer, metrics=metrics, group=group, random_state=random_state)

    def plot(self,
             metrics: Optional[Union[str, List[str]]] = None,
             show_group: bool = False,
             title: Optional[str] = None,
             col_wrap: Optional[int] = None) -> go.Figure:
        """Create the plot

        The plot will show the cross-validation scores for each algorithm and dataset.  The first panel is used to show
        the scorer, that is the metric used to fit the model.  If multiple metrics are supplied, each metric will be
        shown in a separate panel.  If a show_group is true, the metrics will be grouped by the group variable.
        col_wrap allows the width of the plot to be controlled by wrapping the columns to new rows.

        KUDOS: https://towardsdatascience.com/applying-a-custom-colormap-with-plotly-boxplots-5d3acf59e193

        Args:
            metrics: The metric or metrics to plot in addition to the scorer.  Each metric will be plotted in a
             separate panel.
            show_group: If True (and a group variable has been set), plot by group.
            title: Title of the plot
            col_wrap: If plotting multiple metrics, col_wrap will wrap columns to new rows, resulting in
             col-wrap columns, and multiple rows.

        Returns:
            a plotly GraphObjects.Figure

        """

        # Access the attributes of the CrossValidationResult dataclass
        data: pd.DataFrame = self.get_cv_scores()
        data = data.droplevel(level=0, axis=1) if self._num_datasets == 1 else data.droplevel(level=1, axis=1)

        metric_data: pd.DataFrame = pd.DataFrame()

        if metrics is not None:
            if isinstance(metrics, str):
                metrics = [metrics]
            metric_data = self.get_cv_metrics(metrics, show_group)
        else:
            metrics = []

        if self._num_algorithms > 1:
            x_index = 'algo_key'
        else:
            x_index = 'data_key'

        # define the color map for the scorer
        vmin, vmax = data.min().min(), data.max().max()
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.cm.get_cmap('RdYlGn')

        subtitle: str = f'Cross Validation folds={str(self.cv)}'
        if title is None:
            title = subtitle
        else:
            title = title + '<br>' + subtitle

        # create the plot, managing the shape.
        num_plots: int = len(metrics) + 1 if len(metrics) > 0 else 1
        num_cols: int = num_plots if col_wrap is None else col_wrap
        num_rows, _ = subplot_index(len(metrics), col_wrap=num_cols)
        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[f'{self.scorer} (scorer)'] + metrics)

        # Add the scorer subplot
        for col in data.columns:
            # For the scorer build the plot by column to color individually based on score
            median = np.median(data[col])  # find the median
            color = 'rgb' + str(cmap(norm(median))[0:3])  # normalize
            fig.add_trace(go.Box(y=data[col], name=col, boxpoints='all', notched=True, fillcolor=color,
                                 line={"color": "grey"}, marker={"color": "grey"}, showlegend=False,
                                 offsetgroup='A'), row=1, col=1)

        # add the metric subplots
        for i, metric in enumerate(metrics):
            row, col = subplot_index(i + 1, col_wrap=num_cols)
            if show_group:
                colorscale = colors.qualitative.Plotly + colors.qualitative.Dark24
                add_to_legend = True if i == 0 else False
                df_metric: pd.DataFrame = metric_data.query('metric==@metric').drop(columns=['metric'])
                x = df_metric.index.get_level_values(x_index)
                for g, grp in enumerate(df_metric.columns):
                    if len(df_metric.columns) > len(colorscale):
                        raise ValueError("Too many groups to plot")
                    fig.add_trace(go.Box(x=x, y=df_metric[grp], name=grp, boxpoints='all', notched=True,
                                         legendgroup=self.group.name,
                                         showlegend=add_to_legend,
                                         line={"color": colorscale[g]}, marker={"color": colorscale[g]},
                                         offsetgroup=str(g)), row=row, col=col)
            else:
                df_metric: pd.DataFrame = metric_data.query('metric==@metric').drop(columns=['metric'])
                x = list(df_metric.index.get_level_values(x_index))
                fig.add_trace(go.Box(x=x, y=df_metric.values.ravel(), name=metric, boxpoints='all', notched=True,
                                     line={"color": "grey"}, marker={"color": "grey"}), row=row, col=col)

        # finalise some display elements
        fig.update_layout(title=title, showlegend=False)
        if show_group:
            fig.update_layout(boxmode='group', showlegend=True, legend_title=self.group.name,
                              boxgroupgap=0.5, boxgap=0
                              )

        return fig

    def plot_category_analysis(self,
                               algorithm: Optional[str] = None,
                               dataset: Optional[str] = None,
                               metrics: Optional[Union[str, List[str]]] = None,
                               title: Optional[str] = None,
                               col_wrap: Optional[int] = None) -> go.Figure:
        """Plot the category feature analysis

        Args:
            algorithm: If supplied, this will be the name of the algorithm tested.  If None the first algorithm is used.
            dataset: If supplied, this will be the name of the dataset tested.  If None the first dataset is used.
            metrics: The metric or metrics to show.  Each metric will be plotted in a
             separate panel.
            title: Title of the plot
            col_wrap: If plotting multiple metrics, col_wrap will wrap columns to new rows, resulting in
             col-wrap columns, and multiple rows.

        Returns:
            a plotly GraphObjects.Figure

        """
        algorithms: list[str] = list(self.estimators.keys())
        algorithm: str = algorithms[0] if algorithm is None else algorithm
        if algorithm not in algorithms:
            raise KeyError(f"Algorithm {algorithm} is not in the list of available algorithms: {algorithms}")

        datasets: list[str] = list(self.datasets.keys())
        dataset: str = datasets[0] if dataset is None else dataset
        if dataset not in datasets:
            raise KeyError(f"Dataset {dataset} is not in the list of available datasets: {datasets}")

        metrics: list[str] = [list(self.metrics.keys())[0]] if metrics is None else metrics

        baseline_metrics: pd.DataFrame = self.get_cv_metrics(metrics, by_group=True).loc[
            (slice(None), dataset, algorithm)]
        baseline_metrics = baseline_metrics.melt(id_vars=['metric'], value_vars=self.group.unique().tolist(),
                                                 var_name='group',
                                                 ignore_index=False).assign(model='baseline')

        # cross-validate the individual models
        by_group_metrics, by_group_scores = self.get_model_by_group_data(algorithm, dataset)
        by_group_metrics = by_group_metrics.melt(id_vars=['group'], value_vars=metrics, var_name='metric',
                                                 ignore_index=False).assign(model='by_group')
        metric_data: pd.DataFrame = pd.concat([baseline_metrics, by_group_metrics]).sort_values(['model', 'metric'])
        metric_data = metric_data.set_index(['metric', 'group'], append=True).pivot(columns='model',
                                                                                    values='value').reset_index(
            'metric')

        if title is None:
            title = f'Model by Group Test on {algorithm} with cv = {str(self.cv)}'

        num_plots: int = len(metrics) if len(metrics) > 0 else 1
        num_cols: int = num_plots if col_wrap is None else col_wrap
        num_rows, _ = subplot_index(len(metrics) - 1, col_wrap=num_cols)
        fig = make_subplots(rows=num_rows, cols=num_cols,
                            subplot_titles=metrics)

        # metrics
        for i, metric in enumerate(metrics):
            row, col = subplot_index(i, col_wrap=num_cols)
            colorscale = colors.qualitative.Plotly
            add_to_legend = True if i == 0 else False
            df_metric: pd.DataFrame = metric_data.query('metric==@metric').drop(columns=['metric'])
            x = df_metric.index.get_level_values('group')
            for g, grp in enumerate(df_metric.columns):
                fig.add_trace(go.Box(x=x, y=df_metric[grp], name=grp, boxpoints='all', notched=True,
                                     legendgroup=self.group.name,
                                     showlegend=add_to_legend,
                                     line={"color": colorscale[g]}, marker={"color": colorscale[g]},
                                     offsetgroup=str(g)), row=row, col=col)

        fig.update_layout(title=title, showlegend=False)
        fig.update_layout(boxmode='group', showlegend=True, legend_title='model',
                          boxgroupgap=0.5, boxgap=0
                          )

        return fig

    def get_model_by_group_data(self, estimator, dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
        return super().get_model_by_group_data(estimator, dataset)

    def get_cv_scores(self) -> pd.DataFrame:
        return super().get_cv_scores()

    def get_cv_metrics(self, metrics, by_group: bool = False) -> pd.DataFrame:
        return super().get_cv_metrics(metrics, by_group)

    def calculate_metrics(self, x, y, estimators, indices, group) -> Tuple[Dict, Dict]:
        return super().calculate_metrics(x, y, estimators, indices, group)
