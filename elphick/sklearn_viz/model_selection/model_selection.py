import logging
from typing import Union, Optional, Dict, List, Callable, Any

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
from plotly.subplots import make_subplots
from sklearn import model_selection
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from elphick.sklearn_viz.model_selection.metrics import regression_metrics, classification_metrics
from elphick.sklearn_viz.model_selection.scorers import classification_scorers, regression_scorers
from elphick.sklearn_viz.utils import log_timer


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
                 algorithms: Union[sklearn.base.RegressorMixin, sklearn.base.ClassifierMixin, Dict],
                 datasets: Union[pd.DataFrame, Dict],
                 target: str,
                 pre_processor: Optional[Pipeline] = None,
                 k_folds: int = 10,
                 scorer: Optional[Union[str, Callable]] = None,
                 metrics: Optional[Dict[str, Callable]] = None):
        """

        Args:
            algorithms: sklearn estimator or a Dict of algorithms to cross-validate, keyed by string name/code.
            datasets: pandas DataFrame or a dict of DataFrames, keyed by string name/code.
            target: target column
            pre_processor: Optional pipeline used to pre-process the datasets.
            k_folds: The number of cross validation folds.
            scorer: Optional callable scorers which the model will be fitted using
            metrics: Optional Dict of callable metrics to calculate post-fitting
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

        self.features_in: List[str] = [col for col in self.datasets[list(self.datasets.keys())[0]] if
                                       col != self.target]

        self._data: Optional[Dict] = None
        self._num_algorithms: int = len(list(self.algorithms.keys()))
        self._num_datasets: int = len(list(self.datasets.keys()))

        if self._num_algorithms > 1 and self._num_datasets > 1:
            raise NotImplementedError("Cannot have multiple algorithms and multiple datasets.")

    @property
    @log_timer
    def data(self) -> Optional[Dict]:
        if self.metrics is None:
            cv_kwargs: Dict = dict()
        else:
            cv_kwargs: Dict = dict(return_estimator=True, return_indices=True)

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
                    kfold = model_selection.KFold(n_splits=self.k_folds)
                    res = cross_validate(algo, x, y, cv=kfold, scoring=self.scorer, **cv_kwargs)
                    if self.metrics is not None:
                        res['metrics'] = self.calculate_metrics(x=x, y=y, estimators=res['estimator'],
                                                                indices=res['indices'])
                    results[data_key][algo_key] = res
                    res_mean = res[f"test_score"].mean()
                    res_std = res[f"test_score"].std()
                    self._logger.info(f"CV Results for {algo_key}: Mean = {res_mean}, SD = {res_std}")

            self._data = results

        return results

    def plot(self,
             metrics: Optional[Union[str, Dict[str, Any]]] = None,
             color_group: Optional[Union[str, pd.Series]] = None,
             title: Optional[str] = None) -> go.Figure:
        """Create the plot

        KUDOS: https://towardsdatascience.com/applying-a-custom-colormap-with-plotly-boxplots-5d3acf59e193

        Args:
            metrics: The metric or metrics to plot in addition to the scorer.  Each metric will be plotted in a
             separate panel.
            color_group: An optional column name or Series to create a grouped boxplot.  Column must be a category or
             object.
            title: Title of the plot

        Returns:
            a plotly GraphObjects.Figure

        """

        data: pd.DataFrame = self.get_cv_scores()
        data = data.droplevel(level=0, axis=1) if self._num_datasets == 1 else data.droplevel(level=1, axis=1)

        num_plots = 1
        metric_data: pd.DataFrame = pd.DataFrame()
        if metrics is not None:
            if isinstance(metrics, str):
                metrics = {metrics: metrics}
            metric_keys: List[str] = list(metrics.keys())
            metric_data = self.get_cv_metrics(metrics.keys())
            num_plots += len(metrics)
        else:
            metrics = []
            metric_keys = []

        if isinstance(color_group, str):
            color_group = data[color_group]

        if self._num_algorithms > 1:
            xaxis_title = 'Algorithm'
            unstack_index = 'algo_key'
        else:
            xaxis_title = 'Dataset'
            unstack_index = 'data_key'

        vmin, vmax = data.min().min(), data.max().max()
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.cm.get_cmap('RdYlGn')

        subtitle: str = f'Cross Validation folds={self.k_folds}'
        if title is None:
            title = subtitle
        else:
            title = title + '<br>' + subtitle

        fig = make_subplots(rows=1, cols=num_plots, subplot_titles=[f'{self.scorer} (scorer)'] + metric_keys)
        for col in data.columns:
            median = np.median(data[col])  # find the median
            color = 'rgb' + str(cmap(norm(median))[0:3])  # normalize
            fig.add_trace(go.Box(y=data[col], name=col, boxpoints='all', notched=True, fillcolor=color,
                                 line={"color": "grey"}, marker={"color": "grey"}), row=1, col=1)

            for i, metric in enumerate(metrics):
                df_metric: pd.DataFrame = metric_data.query('metric==@metric').drop(columns=['metric']).unstack(
                    unstack_index).droplevel(0, axis=1)
                fig.add_trace(go.Box(y=df_metric[col], name=col, boxpoints='all', notched=True,
                                     line={"color": "grey"}, marker={"color": "grey"}), row=1, col=2 + i)

        fig.update_layout(title=title, showlegend=False)
        return fig

    def get_cv_scores(self) -> pd.DataFrame:
        chunks: List = []
        for data_key, data in self.datasets.items():
            for algo_key, algo in self.algorithms.items():
                chunks.append(pd.Series(self.data[data_key][algo_key][f"test_score"], name=(data_key, algo_key)))
        return pd.concat(chunks, axis=1)

    def get_cv_metrics(self, metrics) -> pd.DataFrame:
        chunks: List = []
        for data_key, data in self.datasets.items():
            for algo_key, algo in self.algorithms.items():
                for metric in metrics:
                    chunks.append(pd.DataFrame(self.data[data_key][algo_key]["metrics"][metric]).assign(
                        **dict(data_key=data_key, algo_key=algo_key, metric=metric)))
        res: pd.DataFrame = pd.concat(chunks, axis=0).set_index(['data_key', 'algo_key'], append=True).rename(
            columns={0: 'value'})
        res.index.names = ['cv', 'data_key', 'algo_key']
        return res

    def calculate_metrics(self, x, y, estimators, indices) -> Dict:
        metric_results: Dict = {}
        for k, fn_metric in self.metrics.items():
            metric_values: List = []
            for estimator, test_indexes in zip(estimators, indices['test']):
                y_true = y[y.index[test_indexes]]
                y_est = estimator.predict(x.loc[x.index[test_indexes], :])
                metric_values.append(fn_metric(y_true, y_est))
            metric_results[k] = metric_values

        return metric_results
