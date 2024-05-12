import logging
import math
import multiprocessing
from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import Union, Optional, Iterable, Any, Callable

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import learning_curve, train_test_split, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

from elphick.sklearn_viz.utils import log_timer


@dataclass
class LearningCurveResult:
    training_scores: np.ndarray
    validation_scores: np.ndarray
    training_sizes: np.ndarray
    metrics: dict[str, dict[str, np.ndarray]] = None

    def get_results(self) -> pd.DataFrame:
        col_names = [f"train_count_{n}" for n in self.training_sizes]
        train: pd.DataFrame = pd.DataFrame(self.training_scores.T, columns=col_names)
        val: pd.DataFrame = pd.DataFrame(self.validation_scores.T, columns=col_names)

        if self.metrics is not None:
            for metric_name in self.metrics.keys():
                train_metric_df = pd.DataFrame(self.metrics[metric_name]['training'].T, columns=col_names)
                val_metric_df = pd.DataFrame(self.metrics[metric_name]['validation'].T, columns=col_names)
                train = pd.concat([train, train_metric_df], axis=1)
                val = pd.concat([val, val_metric_df], axis=1)

        return pd.concat([train.assign(dataset='training'), val.assign(dataset='validation')],
                         axis='index').reset_index(drop=True)

    def get_scorer_results(self, dataset_type) -> np.ndarray:
        return self.training_scores if dataset_type == 'training' else self.validation_scores

    def get_metric_results(self, dataset_type, metric_name) -> np.ndarray:
        return self.metrics[metric_name][dataset_type]

    def get_plot_data(self, key, dataset_type) -> tuple:
        x = list(self.training_sizes)
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
            results: LearningCurveResult = LearningCurveResult(training_scores=train_scores,
                                                               validation_scores=val_scores,
                                                               training_sizes=train_size_abs)

        else:
            # Use the ModelSelection class with the provided metrics
            results: LearningCurveResult = self.custom_learning_curve()

        return results

    def custom_learning_curve(self) -> LearningCurveResult:
        train_scores: list = []
        val_scores: list = []
        train_size_abs: list = []
        metrics: dict = {metric: {'training': [], 'validation': []} for metric in self.metrics.keys()}

        # Determine the cross-validation strategy based on the estimator type
        if self.is_classifier:
            cv = StratifiedKFold(n_splits=self.cv)
        else:
            cv = KFold(n_splits=self.cv)

        def process_train_size(train_size):
            train_scores_fold: list = []
            val_scores_fold: list = []
            metrics_fold: dict = {metric: {'training': [], 'validation': []} for metric in self.metrics.keys()}

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
                    for metric_name_, metric_func in self.metrics.items():
                        train_metric = metric_func(y_train, self.estimator.predict(X_train))
                        val_metric = metric_func(y_val, self.estimator.predict(X_val))

                        metrics_fold[metric_name_]['training'].append(train_metric)
                        metrics_fold[metric_name_]['validation'].append(val_metric)

            return train_scores_fold, val_scores_fold, metrics_fold, train_size_abs

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_train_size)(train_size) for train_size in self.train_sizes)

        for train_scores_fold, val_scores_fold, metrics_fold, train_size_abs_fold in results:
            train_scores.append(train_scores_fold)
            val_scores.append(val_scores_fold)
            train_size_abs.append(train_size_abs_fold)
            for metric_name in metrics.keys():
                metrics[metric_name]['training'].append(metrics_fold[metric_name]['training'])
                metrics[metric_name]['validation'].append(metrics_fold[metric_name]['validation'])

        # Convert lists to numpy arrays
        for metric_name in metrics.keys():
            metrics[metric_name]['training'] = np.array(metrics[metric_name]['training'])
            metrics[metric_name]['validation'] = np.array(metrics[metric_name]['validation'])

        return LearningCurveResult(training_scores=np.array(train_scores), validation_scores=np.array(val_scores),
                                   training_sizes=np.array(train_size_abs).ravel(), metrics=metrics)

    def plot(self,
             title: Optional[str] = None,
             metrics: Optional[list[str]] = None,
             col_wrap: int = 1,
             plot_scorer: bool = True) -> go.Figure:
        """Create the plot

        Args:
            title: title for the plot
            metrics: Optional list of metric keys to plot
            col_wrap: The number of columns to use for the facet grid if plotting metrics.
            plot_scorer: If True, plot the scorer.  Use False to plot only the metrics.

        Returns:
            a plotly GraphObjects.Figure

        """

        # Determine the number of plots to create, their keys, and titles
        total_plots = 0
        plot_keys = []
        subplot_titles = []
        if plot_scorer:
            total_plots += 1
            plot_keys.append('scorer')
            subplot_titles.append(str(self.scorer))
        if metrics:
            total_plots += len(metrics)
            plot_keys += metrics
            subplot_titles += metrics

        num_rows, num_cols, subplot_order = self.calculate_grid_and_subplot_order(total_plots, col_wrap)

        fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=subplot_titles)

        subtitle: str = f'Cross Validation: {self.cv}'
        if title is None:
            title = subtitle
        else:
            title = title + '<br>' + subtitle

        for key, y_label, (row, col) in zip(plot_keys, subplot_titles, subplot_order):
            self._add_subplot(fig=fig, key=key, y_label=y_label, row=row, col=col)

        fig.update_layout(title=title, showlegend=True)

        return fig

    @staticmethod
    def calculate_grid_and_subplot_order(total_plots, col_wrap):
        num_cols = min(total_plots, col_wrap)
        num_rows = math.ceil(total_plots / num_cols)

        subplot_order = [(row, col) for row in range(1, num_rows + 1) for col in range(1, num_cols + 1)]
        subplot_order = subplot_order[:total_plots]  # Trim to the actual number of plots

        return num_rows, num_cols, subplot_order

    def _add_subplot(self, fig: go.Figure, key: str, y_label: str, row: int, col: int) -> go.Figure:
        x, y_train, y_train_lo, y_train_hi = self.results.get_plot_data(key=key, dataset_type='training')
        x, y_val, y_val_lo, y_val_hi = self.results.get_plot_data(key=key, dataset_type='validation')

        # Add legend only for the first subplot
        show_legend = (row == 1 and col == 1)

        fig.add_trace(go.Scatter(
            x=x,
            y=y_train,
            line=dict(color='royalblue'),
            mode='lines',
            name='training',
            showlegend=show_legend,
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x,
            y=y_val,
            line=dict(color='orange'),
            mode='lines',
            name='validation',
            showlegend=show_legend,
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_train_hi + y_train_lo[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor=f"rgba{str(matplotlib.colors.to_rgba('royalblue', 0.4))}",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=show_legend,
            name='training error +/- 1SD'
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_val_hi + y_val_lo[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor="rgba(255, 165, 0, 0.5)",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=show_legend,
            name='validation error +/- 1SD'
        ), row=row, col=col)

        fig.update_xaxes(title_text="Number of training samples", row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

        return fig
