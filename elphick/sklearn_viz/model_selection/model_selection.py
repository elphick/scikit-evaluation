import logging
from copy import deepcopy
from typing import Union, Optional, Dict, List

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn import model_selection
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.pipeline import Pipeline

from elphick.sklearn_viz.utils import log_timer


def plot_model_selection(mdl,
                         models: Dict,
                         x_train: pd.DataFrame,
                         y_train: Union[pd.DataFrame, pd.Series],
                         k_folds: int = 10,
                         x_test: Optional[pd.DataFrame] = None,
                         y_test: Optional[Union[pd.DataFrame, pd.Series]] = None,
                         title: Optional[str] = None) -> go.Figure:
    """

    Args:
            mdl: The scikit-learn model or pipeline.
            models: Dict of models to cross-validate, keyed by string name/code.
            x_train: X values provided to calculate cv error.
            y_train: y values provided to calculate cv error.
            k_folds: The number of cross validation folds.
            x_test: X values provided to calculate test error.
            y_test: y values provided to calculate test error.
            title: Optional plot title

    Returns:
        a plotly GraphObjects.Figure

    """

    return ModelSelection(mdl=mdl, models=models, x_train=x_train, y_train=y_train,
                          k_folds=k_folds, x_test=x_test, y_test=y_test).plot(title=title)


class ModelSelection:
    def __init__(self,
                 mdl,
                 models: Dict,
                 x_train: pd.DataFrame,
                 y_train: Union[pd.DataFrame, pd.Series],
                 k_folds: int = 10,
                 x_test: Optional[pd.DataFrame] = None,
                 y_test: Optional[Union[pd.DataFrame, pd.Series]] = None,
                 models_are_estimators_only: bool = True):
        """

        Args:
            mdl: The scikit-learn model or pipeline.
            models: Dict of models to cross-validate, keyed by string name/code.
            x_train: X values provided to calculate cv error.
            y_train: y values provided to calculate cv error.
            k_folds: The number of cross validation folds.
            x_test: X values provided to calculate test error.
            y_test: y values provided to calculate test error.
            models_are_estimators_only: if True, the models will be substituted in the last position of the mdl if
                it is a pipeline.  If False the models are treated as pipelines and compared directly, which is useful
                 for comparing models with different pre-processors.
        """
        self._logger = logging.getLogger(name=__class__.__name__)
        self.mdl = mdl
        self.models: Dict = models
        self.X_train: Optional[pd.DataFrame] = x_train
        self.y_train: Optional[Union[pd.DataFrame, pd.Series]] = y_train
        self.k_folds: int = k_folds
        self.X_test: Optional[pd.DataFrame] = x_test
        self.y_test: Optional[Union[pd.DataFrame, pd.Series]] = y_test
        self.models_are_estimators_only = models_are_estimators_only
        self._data: Optional[pd.DataFrame] = None
        self.is_pipeline: bool = isinstance(mdl, Pipeline)
        self.is_classifier: bool = is_classifier(mdl)
        self.is_regressor: bool = is_regressor(mdl)
        self.scorer = 'accuracy' if self.is_classifier else 'r2'

        # check_is_fitted(mdl[-1]) if self.is_pipeline else check_is_fitted(mdl)

    @property
    @log_timer
    def data(self) -> Optional[pd.DataFrame]:
        if self._data is not None:
            res = self._data
        else:

            self._logger.info("Commencing Cross Validation")
            cv_chunks: List = []
            test_chunks: List = []
            for name, model in self.models.items():
                if self.models_are_estimators_only and self.is_pipeline:
                    mdl = deepcopy(self.mdl)
                    mdl.steps[-1] = (model.__class__.__name__.lower(), model)
                else:
                    mdl = model
                kfold = model_selection.KFold(n_splits=self.k_folds)
                cv_results = model_selection.cross_val_score(mdl, self.X_train, self.y_train, cv=kfold,
                                                             scoring=self.scorer)
                cv_chunks.append(pd.Series(cv_results.T, name=name))
                self._logger.info(f"CV Results for {name}: Mean = {cv_results.mean()}, SD = {cv_results.std()}")

                # test error
                if self.X_test is not None and self.y_test is not None:
                    test_score: float = np.nan
                    if self.is_regressor:
                        test_score = r2_score(y_true=self.y_test,
                                              y_pred=mdl.fit(self.X_train, self.y_train).predict(
                                                  self.X_test))
                    elif self.is_classifier:
                        test_score = accuracy_score(y_true=self.y_test,
                                                    y_pred=mdl.fit(self.X_train, self.y_train).predict(self.X_test))
                    test_chunks.append(pd.Series([test_score], name=name))

                    self._logger.info(f"Test Result for {name} = {test_score}")

            res_cv: pd.DataFrame = pd.concat(cv_chunks, axis='columns')
            if self.X_test is not None and self.y_test is not None:
                res_test: pd.DataFrame = pd.concat(test_chunks, axis='columns')
                res: pd.DataFrame = pd.concat(
                    [res_test.assign(test_type='test'), res_cv.assign(test_type='cv')]).reset_index(drop=True)
            else:
                res: pd.DataFrame = res_cv.assign(test_type='cv')

            self._data = res

        return res

    def plot(self,
             sort: bool = False,
             title: Optional[str] = None) -> go.Figure:
        """Create the plot

        KUDOS: https://towardsdatascience.com/applying-a-custom-colormap-with-plotly-boxplots-5d3acf59e193

        Args:
            sort: If True, sort by decreasing importance
            title: title for the plot

        Returns:
            a plotly GraphObjects.Figure

        """
        data = self.data.query('test_type=="cv"').drop(columns=['test_type'])
        test_data_exists = self.X_test is not None and self.y_test is not None
        if test_data_exists:
            data_test = self.data.query('test_type=="test"').drop(columns=['test_type'])

        vmin, vmax = data.min().min(), data.max().max()
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.cm.get_cmap('RdYlGn')

        subtitle: str = f'Cross Validation folds={self.k_folds}'
        if title is None:
            title = subtitle
        else:
            title = title + '<br>' + subtitle

        if sort:
            pass

        fig = go.Figure()
        for col in data.columns:
            median = np.median(data[col])  # find the median
            color = 'rgb' + str(cmap(norm(median))[0:3])  # normalize
            fig.add_trace(go.Box(y=data[col], name=col, boxpoints='all', notched=True, fillcolor=color,
                                 line={"color": "grey"}, marker={"color": "grey"}))
            if test_data_exists:
                fig.add_trace(go.Scatter(x=[col], y=data_test[col], name=f'{col}_test',
                                         marker=dict(color='Orange', size=12, line=dict(width=2, color='Grey'))
                                         ))
        fig.update_layout(title=title, showlegend=False, yaxis_title=self.scorer, xaxis_title='Algorithm')
        return fig
