import logging
from typing import Union, Optional, Dict, List, Callable, Tuple

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
from plotly import colors
from plotly.subplots import make_subplots
from sklearn import model_selection
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from elphick.sklearn_viz.model_selection.metrics import regression_metrics, classification_metrics
from elphick.sklearn_viz.model_selection.scorers import classification_scorers, regression_scorers


def subplot_index(idx: int, col_wrap: int) -> Tuple[int, int]:
    col: int = int(idx % col_wrap + 1)
    row: int = int(np.floor(idx / col_wrap) + 1)
    return row, col


def plot_model_selection(algorithms: Union[sklearn.base.RegressorMixin, sklearn.base.ClassifierMixin, Dict],
                         datasets: Union[pd.DataFrame, Dict],
                         target: str,
                         pre_processor: Optional[Pipeline] = None,
                         k_folds: int = 10,
                         title: Optional[str] = None) -> go.Figure:
    """

    Args:
            algorithms: sklearn estimator or a Dict of algorithms to cross-validate, keyed by string name/code.
            datasets: pandas DataFrame or a dict of DataFrames, keyed by string name/code.
            target: target column
            pre_processor: Optional pipeline used to pre-process the datasets.
            k_folds: The number of cross validation folds.
            title: Optional plot title

    Returns:
        a plotly GraphObjects.Figure

    """

    return ModelSelection(algorithms=algorithms, datasets=datasets, target=target, pre_processor=pre_processor,
                          k_folds=k_folds).plot(title=title)


class ModelSelection:
    def __init__(self,
                 algorithms: Union[
                     sklearn.base.RegressorMixin, sklearn.base.ClassifierMixin, Dict[str, sklearn.base.BaseEstimator]],
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
            algorithms: sklearn estimator or a Dict of algorithms to cross-validate, keyed by string name/code.
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
        self.pre_processor: Pipeline = pre_processor
        if isinstance(algorithms, sklearn.base.BaseEstimator):
            self.algorithms = {algorithms.__class__.__name__: algorithms}
        else:
            self.algorithms = algorithms
        if isinstance(datasets, pd.DataFrame):
            self.datasets = {'Dataset': datasets}
        else:
            self.datasets = datasets
        self.target = target
        self.k_folds: int = k_folds

        self.is_classifier: bool = is_classifier(list(self.algorithms.values())[0])
        self.is_regressor: bool = is_regressor(list(self.algorithms.values())[0])
        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = classification_scorers[list(classification_scorers.keys())[0]] if self.is_classifier else \
                regression_scorers[list(regression_scorers.keys())[0]]

        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = classification_metrics if self.is_classifier else regression_metrics

        self.group: pd.Series = group
        self.random_state: Optional[int] = random_state

        self.features_in: List[str] = [col for col in self.datasets[list(self.datasets.keys())[0]] if
                                       col != self.target]

        self._data: Optional[Dict] = None
        self._num_algorithms: int = len(list(self.algorithms.keys()))
        self._num_datasets: int = len(list(self.datasets.keys()))

        if self._num_algorithms > 1 and self._num_datasets > 1:
            raise NotImplementedError("Cannot have multiple algorithms and multiple datasets.")

    @property
    def data(self) -> Optional[Dict]:
        if self.metrics is None:
            cv_kwargs: Dict = dict()
        else:
            cv_kwargs: Dict = dict(return_estimator=True, return_indices=True, error_score=np.nan)

        if self._data is not None:
            results = self._data
        else:
            results: Dict = {}
            for data_key, data in self.datasets.items():
                self._logger.info(f"Commencing Cross Validation for dataset {data_key}")
                results[data_key] = {}
                x: pd.DataFrame = data[self.features_in]
                y: pd.DataFrame = data[self.target]
                if self.pre_processor:
                    x = self.pre_processor.set_output(transform="pandas").fit_transform(X=x)

                for algo_key, algo in self.algorithms.items():
                    kfold = model_selection.KFold(n_splits=self.k_folds, random_state=self.random_state, shuffle=True)
                    res = cross_validate(algo, x, y, cv=kfold, scoring=self.scorer, **cv_kwargs)
                    if self.metrics is not None:
                        res['metrics'], res['metrics_group'] = self.calculate_metrics(x=x, y=y,
                                                                                      estimators=res['estimator'],
                                                                                      indices=res['indices'],
                                                                                      group=self.group)
                    results[data_key][algo_key] = res
                    res_mean = res[f"test_score"].mean()
                    res_std = res[f"test_score"].std()
                    self._logger.info(f"CV Results for {algo_key}: Mean = {res_mean}, SD = {res_std}")

            self._data = results

        return results

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

        subtitle: str = f'Cross Validation folds={self.k_folds}'
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
        algorithms: list[str] = list(self.algorithms.keys())
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
            title = f'Model by Group Test on {algorithm} with {self.k_folds} folds'

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

    def get_model_by_group_data(self, algorithm, dataset):
        results: dict = {}
        by_group_score_chunks: list = []
        by_group_metric_chunks: list = []
        for grp in self.group.unique():
            grp_index: pd.Index = self.group.loc[self.group == grp].index
            x: pd.DataFrame = self.datasets[dataset][self.features_in].loc[grp_index]
            y: pd.DataFrame = self.datasets[dataset][self.target].loc[grp_index]
            if self.pre_processor:
                x = self.pre_processor.set_output(transform="pandas").fit_transform(X=x)
            kfold = model_selection.KFold(n_splits=self.k_folds, random_state=self.random_state, shuffle=True)
            res = cross_validate(self.algorithms[algorithm], x, y, cv=kfold, scoring=self.scorer, return_estimator=True,
                                 return_indices=True)
            by_group_score_chunks.append(pd.Series(res['test_score'], name=grp))
            if self.metrics is not None:
                res['metrics'], _ = self.calculate_metrics(x=x, y=y,
                                                           estimators=res['estimator'],
                                                           indices=res['indices'],
                                                           group=None)
            results[grp] = res
            by_group_metric_chunks.append(
                pd.DataFrame(res['metrics']).assign(group=grp).rename_axis('fold', axis='index'))
        by_group_metrics: pd.DataFrame = pd.concat(by_group_metric_chunks)
        #                                   .reset_index().melt(id_vars=['fold', 'group'],
        #                                                                                       value_vars=list(
        #                                                                                           self.metrics.keys()),
        #                                                                                       var_name='metric'))
        # by_group_metrics = by_group_metrics.set_index(['fold', 'metric', 'group']).unstack(level=-1)
        # by_group_metrics = by_group_metrics.droplevel(level=0, axis=1).reset_index(-1)
        by_group_scores = pd.concat(by_group_score_chunks, axis=1)

        return by_group_metrics, by_group_scores

    def get_cv_scores(self) -> pd.DataFrame:
        chunks: List = []
        for data_key, data in self.datasets.items():
            for algo_key, algo in self.algorithms.items():
                chunks.append(pd.Series(self.data[data_key][algo_key][f"test_score"], name=(data_key, algo_key)))
        return pd.concat(chunks, axis=1)

    def get_cv_metrics(self, metrics, by_group: bool = False) -> pd.DataFrame:
        chunks: List = []
        metric_key = "metrics_group" if by_group else "metrics"
        for data_key, data in self.datasets.items():
            for algo_key, algo in self.algorithms.items():
                for metric in metrics:
                    chunks.append(pd.DataFrame(self.data[data_key][algo_key][metric_key][metric]).assign(
                        **dict(data_key=data_key, algo_key=algo_key, metric=metric)))
        res: pd.DataFrame = pd.concat(chunks, axis=0).set_index(['data_key', 'algo_key'], append=True).rename(
            columns={0: 'value'})
        res.index.names = ['fold', 'data_key', 'algo_key']
        return res

    def calculate_metrics(self, x, y, estimators, indices, group) -> Tuple[Dict, Dict]:
        metric_results: Dict = {}
        metric_results_group: Dict = {}

        for k, fn_metric in self.metrics.items():
            metric_values: List = []
            metric_groups: Dict = {}
            for estimator, test_indexes in zip(estimators, indices['test']):
                y_true = y[y.index[test_indexes]]
                y_est = estimator.predict(x.loc[x.index[test_indexes], :])
                if isinstance(y_est, pd.DataFrame) and y_est.shape[1] == 1:
                    y_est = y_est.iloc[:, 0]
                metric_values.append(fn_metric(y_true, y_est))
                if group is not None:
                    # calculate the metric by each group in the group series.
                    y_est = pd.merge(left=pd.Series(y_est, name='y_est', index=x.index[test_indexes]),
                                     right=group, left_index=True, right_index=True)
                    y_est_grouped = y_est.groupby([group.name])
                    grouped_results = [y_est_grouped.get_group(x) for x in y_est_grouped.groups]
                    for grp_res in grouped_results:
                        group_value = str(grp_res[group.name].iloc[0])
                        group_metric_results = fn_metric(y_true[grp_res.index], grp_res['y_est'].values)
                        if group_value not in metric_groups.keys():
                            metric_groups[group_value] = [group_metric_results]
                        else:
                            metric_groups[group_value].append(group_metric_results)
            metric_results[k] = metric_values
            if group is not None:
                metric_results_group[k] = metric_groups

        return metric_results, metric_results_group

# if __name__ == '__main__':
#
#     for i in range(0, 7):
#         print(subplot_index(i, col_wrap=3))
#
#     print('done')
