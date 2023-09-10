import logging
from typing import Union, Optional, Iterable, Any

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

from elphick.sklearn_viz.utils import log_timer


def plot_learning_curve(mdl,
                        x: pd.DataFrame,
                        y: Union[pd.DataFrame, pd.Series],
                        cv: Union[int, Any] = 5,
                        title: Optional[str] = None) -> go.Figure:
    """

    Args:
        mdl: The scikit-learn model or pipeline.
        x: X values provided to calculate the learning curve.
        y: y values provided to calculate the learning curve.
        cv: The number of cross validation folds or cv callable.
        title: Optional plot title

    Returns:
        a plotly GraphObjects.Figure

    """

    return LearningCurve(mdl=mdl, x=x, y=y, cv=cv).plot(title=title)


class LearningCurve:
    def __init__(self,
                 mdl,
                 x: pd.DataFrame,
                 y: Union[pd.DataFrame, pd.Series],
                 train_sizes: Iterable = np.linspace(0.1, 1.0, 5),
                 cv: Union[int, Any] = 5):
        """

        Args:
            mdl: The scikit-learn model or pipeline.
            x: X values provided to calculate the learning curve.
            y: y values provided to calculate the learning curve.
            train_sizes: list of training sample counts (or fractions if < 1)
            cv: The number of cross validation folds or a cv callable.
        """
        self._logger = logging.getLogger(name=__class__.__name__)
        self.mdl = mdl
        self.X: Optional[pd.DataFrame] = x
        self.y: Optional[Union[pd.DataFrame, pd.Series]] = y
        self.train_sizes: Iterable = train_sizes
        self.cv: int = cv
        self._data: Optional[pd.DataFrame] = None
        self.is_pipeline: bool = isinstance(mdl, Pipeline)
        self.is_classifier: bool = is_classifier(mdl)
        self.is_regressor: bool = is_regressor(mdl)
        self.scorer = None

        # check_is_fitted(mdl[-1]) if self.is_pipeline else check_is_fitted(mdl)

    @property
    @log_timer
    def data(self) -> Optional[pd.DataFrame]:
        if self._data is not None:
            res = self._data
        else:

            if self.is_regressor:
                self.scorer = 'neg_mean_squared_error'
            elif self.is_classifier:
                self.scorer = 'accuracy'
            else:
                raise NotImplementedError("Only Classifers and Regressors are supported.")

            self._logger.info("Commencing Cross Validation")
            train_size_abs, train_scores, val_scores = learning_curve(self.mdl, X=self.X, y=self.y,
                                                                      train_sizes=self.train_sizes)
            col_names = [f"train_count_{n}" for n in train_size_abs]
            train: pd.DataFrame = pd.DataFrame(train_scores.T, columns=col_names).assign(dataset='training')
            val: pd.DataFrame = pd.DataFrame(val_scores.T, columns=col_names).assign(dataset='validation')
            res: pd.DataFrame = pd.concat([train, val], axis='index').reset_index(drop=True)

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
        data_train = self.data.query('dataset=="training"').drop(columns=['dataset'])
        data_val = self.data.query('dataset=="validation"').drop(columns=['dataset'])

        subtitle: str = f'Cross Validation: {self.cv}'
        if title is None:
            title = subtitle
        else:
            title = title + '<br>' + subtitle

        if sort:
            pass

        fig = go.Figure()
        x = [int(col.split('_')[-1]) for col in data_train.columns]
        y_train = list(data_train.mean())
        train_sd = list(data_train.std())
        y_val = list(data_val.mean())
        val_sd = list(data_val.std())
        y_train_upper = list(data_train.mean() + train_sd)
        y_train_lower = list(data_train.mean() - train_sd)
        y_val_upper = list(data_val.mean() + val_sd)
        y_val_lower = list(data_val.mean() - val_sd)

        fig.add_trace(go.Scatter(
            x=x,
            y=y_train,
            line=dict(color='royalblue'),
            mode='lines',
            name='training',
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=y_val,
            line=dict(color='orange'),
            mode='lines',
            name='validation',
        ))
        fig.add_trace(go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_train_upper + y_train_lower[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor=f"rgba{str(matplotlib.colors.to_rgba('royalblue', 0.4))}",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='training error +/- 1SD'
        ))
        fig.add_trace(go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_val_upper + y_val_lower[::-1],  # upper, then lower reversed
            fill='toself',
            # fillcolor=f"rgba{str(matplotlib.colors.to_rgba('orange', 0.5))}",
            fillcolor="rgba(255, 165, 0, 0.5)",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='validation error +/- 1SD'
        ))
        fig.update_layout(title=title, showlegend=True, yaxis_title=self.scorer,
                          xaxis_title='Number of samples in the training set')
        return fig
