import logging
from typing import Union, Optional, Dict, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import is_classifier

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from elphick.sklearn_viz.model_selection.scorers import r2_score_with_nan, classification_scorers, regression_scorers
from elphick.sklearn_viz.utils import log_timer


def plot_feature_importance(mdl,
                            sort: bool = False,
                            top_k: Optional[int] = None,
                            horizontal: bool = False,
                            permute: bool = False,
                            pipeline_input_features: bool = False,
                            x_test: Optional[pd.DataFrame] = None,
                            y_test: Optional[Union[pd.DataFrame, pd.Series]] = None,
                            title: Optional[str] = None
                            ) -> go.Figure:
    """

    Args:
        mdl: The scikit-learn model or pipeline.
        sort: If True, sort by decreasing importance
        top_k: Include only the top k features in the plot.  Will ignore the sort argument.
        horizontal: If True plot horizontal bars, if False vertical bars.
        permute: If True plot permutation importance.  Better, but slower.  Requires X_test and y_test to be provided.
        pipeline_input_features: If True, and a pipeline is provided, report the features provided as inputs to
         the pipeline.  If False, reports the estimator (last pipeline step) input features.  Requires permute = True.
        x_test: X values provided to execute permuted importance.
        y_test: y values provided to execute permuted importance.
        title: title for the plot

    Returns:
        a plotly GraphObjects.Figure

    """

    return FeatureImportance(mdl=mdl, permute=permute, pipeline_input_features=pipeline_input_features,
                             x_test=x_test, y_test=y_test).plot(sort=sort, top_k=top_k, horizontal=horizontal,
                                                                title=title)


class FeatureImportance:
    def __init__(self,
                 mdl,
                 permute: bool = False,
                 pipeline_input_features: bool = False,
                 x_test: Optional[pd.DataFrame] = None,
                 y_test: Optional[Union[pd.DataFrame, pd.Series]] = None,
                 scorer: Optional[Union[str, Callable]] = None,):
        """

        Args:
            mdl: The scikit-learn model or pipeline.
            permute: If True plot permutation importance.  Better, but slower.  Requires X_test and y_test to be provided.
            pipeline_input_features: If True, and a pipeline is provided, report the features provided as inputs to
             the pipeline.  If False, reports the estimator (last pipeline step) input features.  Requires permute = True.
            x_test: X values provided to execute permuted importance.
            y_test: y values provided to execute permuted importance.
            scorer: Optional callable scorer which the model will be fitted using

        """
        self._logger = logging.getLogger(name=__class__.__name__)
        self.mdl = mdl
        self.permute: bool = permute
        self.pipeline_input_features: bool = pipeline_input_features
        self.X_test: Optional[pd.DataFrame] = x_test
        self.y_test: Optional[Union[pd.DataFrame, pd.Series]] = y_test

        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = classification_scorers[list(classification_scorers.keys())[0]] if is_classifier(self.mdl) else \
                regression_scorers[list(regression_scorers.keys())[0]]

        self._data: Optional[pd.DataFrame] = None
        self.is_pipeline: bool = isinstance(mdl, Pipeline)

        if not self.permute:
            check_is_fitted(mdl[-1]) if self.is_pipeline else check_is_fitted(mdl)

    @property
    @log_timer
    def data(self) -> Optional[pd.DataFrame]:
        if self._data is not None:
            res = self._data
        else:
            mdl = self.mdl

            if self.permute:
                self._logger.info("Generating feature importance by permutation")
                x = self.X_test
                if self.is_pipeline and not self.pipeline_input_features:
                    mdl = mdl[-1]
                    x = self.mdl[0:-1].transform(self.X_test)

                result = permutation_importance(estimator=mdl, X=x, y=self.y_test, n_repeats=10, random_state=42,
                                                n_jobs=2, scoring=self.scorer)
                importances = result.importances_mean
                std = result.importances_std
            else:
                self._logger.info("Extracting feature importance from the fitted model")
                if self.is_pipeline:
                    mdl = mdl[-1]
                try:  # trees
                    importances = mdl.feature_importances_
                    std = np.std([tree.feature_importances_ for tree in mdl.estimators_], axis=0)
                except AttributeError:  # regression
                    importances = mdl.coef_
                    std = np.full(len(importances), np.nan)

            try:
                feature_names = mdl.feature_names_in_
            except AttributeError:
                self._logger.warning("Feature names are not available within the model."
                                     " Setting the transform output to pandas will correct this."
                                     " e.g. pipe.set_output(transform='pandas')."
                                     " Retrying with the pre-processed feature names.")
                try:
                    # Likely non-sklearn estimator like CatBoostRegressor
                    feature_names = list(self.mdl[0:-1].transform(self.X_test).columns)
                except AttributeError:
                    self._logger.warning("Feature names are not available within the model."
                                         " Setting default names")
                    feature_names = [f"F{i}" for i in range(1, mdl.n_features_in_ + 1)]

            res: pd.DataFrame = pd.DataFrame([importances, std], index=['importance', 'std'], columns=feature_names).T
            self._data = res

        return res

    def plot(self,
             sort: bool = False,
             top_k: Optional[int] = None,
             horizontal: bool = False,
             title: Optional[str] = None) -> go.Figure:
        """

        Args:
            sort: If True, sort by decreasing importance
            top_k: Include only the top k features in the plot.  Will ignore the sort argument.
            horizontal: If True plot horizontal bars, if False vertical bars.
             the pipeline.  If False, reports the estimator (last pipeline step) input features.  Requires permute = True.
            title: title for the plot

        Returns:
            a plotly GraphObjects.Figure

        """
        data = self.data
        subtitle: str = 'Feature Importance'
        if self.permute:
            subtitle = 'Permuted ' + subtitle

        if sort or top_k is not None:
            data = data.sort_values(by=['importance'], ascending=False)
            if top_k is not None:
                data = data.iloc[0:top_k, :]
            if horizontal:
                data = data.sort_values(by=['importance'], ascending=True)

        if horizontal:
            kwargs: Dict = {'y': data.index,
                            'x': data['importance'],
                            'error_x': dict(type='data', array=data['std']),
                            'orientation': 'h'}
        else:
            kwargs: Dict = {'x': data.index,
                            'y': data['importance'],
                            'error_y': dict(type='data', array=data['std'])}

        if title is None:
            title = subtitle
        else:
            title = title + '<br>' + subtitle

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Model 1', **kwargs))
        fig.update_layout(title=title, barmode='group')

        return fig
