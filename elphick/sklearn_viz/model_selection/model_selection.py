import logging
from typing import Union, Optional, Dict, List

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
from sklearn import model_selection
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

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
                 k_folds: int = 10):
        """

        Args:
            algorithms: sklearn estimator or a Dict of algorithms to cross-validate, keyed by string name/code.
            datasets: pandas DataFrame or a dict of DataFrames, keyed by string name/code.
            target: target column
            pre_processor: Optional pipeline used to pre-process the datasets.
            k_folds: The number of cross validation folds.
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

        self.is_classifier: bool = is_classifier(list(self.algorithms.items())[0][0])
        self.is_regressor: bool = is_regressor(list(self.algorithms.items())[0][0])
        self.scorers = classification_scorers if self.is_classifier else regression_scorers

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
        if self._data is not None:
            results = self._data
        else:
            first_scorer: str = list(self.scorers.items())[0][0]
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
                    res = cross_validate(algo, x, y, cv=kfold, scoring=self.scorers)
                    results[data_key][algo_key] = res
                    res_mean = res[f"test_{first_scorer}"].mean()
                    res_std = res[f"test_{first_scorer}"].std()
                    self._logger.info(f"CV Results for {algo_key}: Mean = {res_mean}, SD = {res_std}")

            self._data = results

        return results

    def plot(self,
             scorer: Optional[str] = None,
             title: Optional[str] = None) -> go.Figure:
        """Create the plot

        KUDOS: https://towardsdatascience.com/applying-a-custom-colormap-with-plotly-boxplots-5d3acf59e193

        Args:
            scorer: the scorer metric to plot
            title: title for the plot

        Returns:
            a plotly GraphObjects.Figure

        """

        if scorer is None:
            scorer: str = list(self.scorers.items())[0][0]

        data: pd.DataFrame = self.get_cv_scores(scorer)
        data = data.droplevel(level=0, axis=1) if self._num_datasets == 1 else data.droplevel(level=1, axis=1)
        xaxis_title = 'Algorithm' if self._num_algorithms > 1 else 'Dataset'

        vmin, vmax = data.min().min(), data.max().max()
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.cm.get_cmap('RdYlGn')

        subtitle: str = f'Cross Validation folds={self.k_folds}'
        if title is None:
            title = subtitle
        else:
            title = title + '<br>' + subtitle

        fig = go.Figure()
        for col in data.columns:
            median = np.median(data[col])  # find the median
            color = 'rgb' + str(cmap(norm(median))[0:3])  # normalize
            fig.add_trace(go.Box(y=data[col], name=col, boxpoints='all', notched=True, fillcolor=color,
                                 line={"color": "grey"}, marker={"color": "grey"}))
        fig.update_layout(title=title, showlegend=False, yaxis_title=scorer, xaxis_title=xaxis_title)
        return fig

    def get_cv_scores(self, scorer) -> pd.DataFrame:
        chunks: List = []
        for data_key, data in self.datasets.items():
            for algo_key, algo in self.algorithms.items():
                chunks.append(pd.Series(self.data[data_key][algo_key][f"test_{scorer}"], name=(data_key, algo_key)))
        return pd.concat(chunks, axis=1)
