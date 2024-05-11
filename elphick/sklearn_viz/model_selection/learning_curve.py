import logging
import multiprocessing
from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import Union, Optional, Iterable, Any, Callable

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import learning_curve, train_test_split, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

from elphick.sklearn_viz.utils import log_timer


@dataclass
class LearningCurveResult:
    train_scores: np.ndarray
    val_scores: np.ndarray
    train_sizes: np.ndarray
    metrics: dict[str, dict[str, np.ndarray]] = None

    def get_results(self) -> pd.DataFrame:
        col_names = [f"train_count_{n}" for n in self.train_sizes]
        train: pd.DataFrame = pd.DataFrame(self.train_scores.T, columns=col_names)
        val: pd.DataFrame = pd.DataFrame(self.val_scores.T, columns=col_names)

        if self.metrics is not None:
            for metric_name in self.metrics.keys():
                train_metric_df = pd.DataFrame(self.metrics[metric_name]['train'].T, columns=col_names)
                val_metric_df = pd.DataFrame(self.metrics[metric_name]['val'].T, columns=col_names)
                train = pd.concat([train, train_metric_df], axis=1)
                val = pd.concat([val, val_metric_df], axis=1)

        return pd.concat([train.assign(dataset='training'), val.assign(dataset='validation')],
                         axis='index').reset_index(drop=True)

    def get_scorer_results(self, dataset_type) -> np.ndarray:
        return self.train_scores if dataset_type == 'training' else self.val_scores

    def get_metric_results(self, dataset_type, metric_name) -> np.ndarray:
        return self.metrics[metric_name][dataset_type]

    def get_plot_data(self, key, dataset_type) -> tuple:
        x = list(self.train_sizes)
        if key == 'scorer':
            data = self.get_scorer_results(dataset_type=dataset_type)
        else:
            data = self.get_metric_results(dataset_type=dataset_type, metric_name=key)
        y = np.mean(data, axis=1)
        y_sd = np.std(data, axis=1)
        y_lo = list(y - y_sd)
        y_hi = list(y + y_sd)
        y = list(y)
        return x, y, y_lo, y_hi


def plot_learning_curve(estimator,
                        x: pd.DataFrame,
                        y: Union[pd.DataFrame, pd.Series],
                        cv: Union[int, Any] = 5,
                        title: Optional[str] = None) -> go.Figure:
    """

    Args:
        estimator: The scikit-learn model or pipeline.
        x: X values provided to calculate the learning curve.
        y: y values provided to calculate the learning curve.
        cv: The number of cross validation folds or cv callable.
        title: Optional plot title

    Returns:
        a plotly GraphObjects.Figure

    """

    return LearningCurve(estimator=estimator, x=x, y=y, cv=cv).plot(title=title)


class LearningCurve:
    def __init__(self,
                 estimator,
                 x: pd.DataFrame,
                 y: Union[pd.DataFrame, pd.Series],
                 train_sizes: Iterable = np.linspace(0.1, 1.0, 5),
                 cv: Union[int, Any] = 5,
                 metrics: Optional[dict[str, Callable]] = None,
                 scorer: Optional[Any] = None,
                 random_state: int = 42,
                 n_jobs: int = 1):
        """

        Args:
            estimator: The scikit-learn model or pipeline.
            x: X values provided to calculate the learning curve.
            y: y values provided to calculate the learning curve.
            train_sizes: list of training sample counts (or fractions if < 1)
            cv: The number of cross validation folds or a cv callable.
            metrics: Optional Dict of callable metrics to calculate post-fitting
            scorer: The scoring method.  If None, 'accuracy' is used for classifiers and 'r2' for regressors.
            random_state: Optional random seed
            n_jobs: Number of parallel jobs to run.  If -1, then the number of jobs is set to the number of CPU cores.
             Recommend setting to -2 for large jobs to retain a core for system interaction.
            verbosity: Verbosity level.  0 = silent, 1 = overall (start/finish), 2 = each cross-validation.

        """
        self._logger = logging.getLogger(name=__class__.__name__)
        self.estimator = estimator
        self.X: Optional[pd.DataFrame] = x
        self.y: Optional[Union[pd.DataFrame, pd.Series]] = y
        self.train_sizes: Iterable = train_sizes
        self.cv: int = cv
        self.random_state: int = random_state
        self.n_jobs: int = n_jobs
        self.metrics = metrics

        self.is_pipeline: bool = isinstance(estimator, Pipeline)
        self.is_classifier: bool = is_classifier(estimator)
        self.is_regressor: bool = is_regressor(estimator)

        if scorer is None:
            scorer = 'accuracy' if self.is_classifier else 'r2'
        self.scorer: Optional[Any] = scorer

        self._results: Optional[pd.DataFrame] = None

        # check_is_fitted(mdl[-1]) if self.is_pipeline else check_is_fitted(mdl)

    @property
    def n_cores(self) -> int:
        n_cores = self.n_jobs
        if self.n_jobs < 0:
            n_cores = multiprocessing.cpu_count() + 1 + self.n_jobs
        return n_cores

    @property
    @log_timer
    def results(self) -> Optional[pd.DataFrame]:
        if self._results is None:
            start_time = datetime.now()  # Record the start time

            self._logger.info("Commencing Cross Validation")

            results = self.calculate_learning_curve()

            duration = str(timedelta(seconds=round((datetime.now() - start_time).total_seconds())))
            self._logger.info(f"Cross validation complete in {duration} using {self.n_cores} "
                              f"worker{'s' if self.n_cores > 1 else ''}")

            self._results = results

        return self._results

    def calculate_learning_curve(self) -> LearningCurveResult:
        if self.metrics is None:
            # Use the scikit-learn learning_curve method
            train_size_abs, train_scores, val_scores = learning_curve(self.estimator, X=self.X, y=self.y,
                                                                      train_sizes=self.train_sizes,
                                                                      scoring=self.scorer, cv=self.cv,
                                                                      n_jobs=self.n_jobs)
            results: LearningCurveResult = LearningCurveResult(train_scores=train_scores, val_scores=val_scores,
                                                               train_sizes=train_size_abs)

        else:
            # Use the ModelSelection class with the provided metrics
            results: LearningCurveResult = self.custom_learning_curve()

        return results

    def custom_learning_curve(self) -> LearningCurveResult:
        train_scores: list = []
        val_scores: list = []
        train_size_abs: list = []
        metrics: dict = {metric: {'train': [], 'val': []} for metric in self.metrics.keys()}

        # Determine the cross-validation strategy based on the estimator type
        if self.is_classifier:
            cv = StratifiedKFold(n_splits=self.cv)
        else:
            cv = KFold(n_splits=self.cv)

        for train_size in self.train_sizes:

            train_scores_fold: list = []
            val_scores_fold: list = []
            metrics_fold: dict = {metric: {'train': [], 'val': []} for metric in self.metrics.keys()}

            for i, (train_index, val_index) in enumerate(cv.split(self.X, self.y)):
                X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
                y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

                # Ensure that train_size doesn't exceed the size of the training set
                train_size = min(train_size, len(X_train))

                if train_size <= 1:
                    train_size = int(train_size * len(X_train))
                else:
                    train_size = int(train_size)

                X_train = X_train[:train_size]
                y_train = y_train[:train_size]

                if i == 0:
                    train_size_abs.append(len(X_train))

                self.estimator.fit(X_train, y_train)

                train_scores_fold.append(self.estimator.score(X_train, y_train))
                val_scores_fold.append(self.estimator.score(X_val, y_val))

                if self.metrics is not None:
                    for metric_name, metric_func in self.metrics.items():
                        train_metric = metric_func(y_train, self.estimator.predict(X_train))
                        val_metric = metric_func(y_val, self.estimator.predict(X_val))

                        metrics_fold[metric_name]['train'].append(train_metric)
                        metrics_fold[metric_name]['val'].append(val_metric)

            # Average the results over the folds
            train_scores.append(train_scores_fold)
            val_scores.append(val_scores_fold)
            for metric_name in metrics.keys():
                metrics[metric_name]['train'].append(metrics_fold[metric_name]['train'])
                metrics[metric_name]['val'].append(metrics_fold[metric_name]['val'])

        # Convert lists to numpy arrays
        for metric_name in metrics.keys():
            metrics[metric_name]['train'] = np.array(metrics[metric_name]['train'])
            metrics[metric_name]['val'] = np.array(metrics[metric_name]['val'])

        results = LearningCurveResult(train_scores=np.array(train_scores), val_scores=np.array(val_scores),
                                      train_sizes=np.array(train_size_abs), metrics=metrics)

        return results

    def plot(self,
             title: Optional[str] = None,
             metrics: Optional[list[str]] = None,
             col_wrap: int = 1) -> go.Figure:
        """Create the plot

        Args:
            title: title for the plot
            metrics: Optional list of metric keys to plot
            col_wrap: The number of columns to use for the facet grid if plotting metrics.

        Returns:
            a plotly GraphObjects.Figure

        """

        x, y_train, y_train_lo, y_train_hi = self.results.get_plot_data(key='scorer', dataset_type='training')
        x, y_val, y_val_lo, y_val_hi = self.results.get_plot_data(key='scorer', dataset_type='validation')

        subtitle: str = f'Cross Validation: {self.cv}'
        if title is None:
            title = subtitle
        else:
            title = title + '<br>' + subtitle

        fig = go.Figure()

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
            y=y_train_hi + y_train_lo[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor=f"rgba{str(matplotlib.colors.to_rgba('royalblue', 0.4))}",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='training error +/- 1SD'
        ))
        fig.add_trace(go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_val_hi + y_val_lo[::-1],  # upper, then lower reversed
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
