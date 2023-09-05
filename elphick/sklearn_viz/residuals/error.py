import logging
from typing import Optional, Union

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import plotly.graph_objects as go
import plotly.express as px


class Errors:
    def __init__(self,
                 mdl,
                 x_test: Optional[pd.DataFrame] = None,
                 y_test: Optional[Union[pd.DataFrame, pd.Series]] = None):
        """

        Args:
            mdl: The scikit-learn model or pipeline.
            x_test: X values provided to calculate residuals.
            y_test: y values provided to calculate residuals.
        """
        self._logger = logging.getLogger(name=__class__.__name__)
        self.mdl = mdl
        self.X_test: Optional[pd.DataFrame] = x_test
        self.y_test: Optional[Union[pd.DataFrame, pd.Series]] = y_test
        self._data: Optional[pd.DataFrame] = None
        self.is_pipeline: bool = isinstance(mdl, Pipeline)

        check_is_fitted(mdl[-1]) if self.is_pipeline else check_is_fitted(mdl)

    def plot(self, color: Optional[str] = None, title: Optional[str] = 'Error Plot') -> go.Figure:
        """

        Args:
            color: The variable name to color (group) by.
            title: title for the plot

        Returns:
            a plotly GraphObjects.Figure

        """
        y_est = pd.Series(self.mdl.predict(self.X_test), name=f"{self.y_test.name}_est", index=self.X_test.index)
        y = self.y_test
        data = pd.concat([y, y_est], axis='columns')
        lims = [data.min().min(), data.max().max()]
        fig = px.scatter(
            data,
            x=y.name, y=y_est.name,
            marginal_x='histogram', marginal_y='histogram',
            color=color, trendline='ols'
        )
        fig.update_traces(histnorm='probability', selector={'type': 'histogram'})
        fig.add_shape(
            type="line", line=dict(dash='dash'),
            x0=lims[0], y0=lims[0],
            x1=lims[1], y1=lims[1]
        )

        fig.update_layout(title=title)

        return fig
