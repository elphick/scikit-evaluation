import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
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

    def plot(self, color: Optional[str] = None, title: Optional[str] = 'Error Plot',
             marginal: bool = False) -> go.Figure:
        """

        Args:
            color: The variable name to color (group) by.
            title: title for the plot
            marginal: If True, marginal histograms are added.

        Returns:
            a plotly GraphObjects.Figure

        """
        y_est = pd.Series(self.mdl.predict(self.X_test), name=f"{self.y_test.name}_est", index=self.X_test.index)
        y = self.y_test
        data = pd.concat([y, y_est], axis='columns')
        index_name: str = 'index' if data.index.name is None else data.index.name

        # Calculate the range of the data and manage padding.
        data_range = data.max().max() - data.min().min()
        padding = data_range * 0.05
        lims = [data.min().min() - padding, data.max().max() + padding]

        if marginal:
            fig = make_subplots(rows=2, cols=2, column_widths=[0.8, 0.2], row_heights=[0.2, 0.8],
                                subplot_titles=(False, False, False, False),
                                vertical_spacing=0.02,  # Adjust as needed
                                horizontal_spacing=0.02)  # Adjust as needed
        else:
            fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Scatter(
            x=data[y.name],
            y=data[y_est.name],
            mode='markers',
            marker=dict(color=color),
            name='Data',
            hovertemplate=f'{index_name}: ' + '%{text}<br>x: %{x}<br>y: %{y}',
            text=data.index
        ), row=2 if marginal else 1, col=1)

        fig.add_trace(go.Scatter(
            x=lims,
            y=lims,
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='y=x line',
            showlegend=False
        ), row=2 if marginal else 1, col=1)

        if marginal:

            fig.add_trace(
                go.Histogram(x=data[y.name], name='Top Histogram', marker=dict(color='grey'), nbinsx=20, ), row=1,
                col=1)
            fig.add_trace(
                go.Histogram(y=data[y_est.name], name='Right Histogram', marker=dict(color='grey'), nbinsy=20, ),
                row=2, col=2)

            # Set the range of the x-axis for the top histogram and the range of the y-axis for the right histogram
            fig.update_xaxes(range=lims, showgrid=True, gridcolor='white', title_text='', showticklabels=False, row=1,
                             col=1)
            fig.update_yaxes(range=[0, 'data max'], showgrid=False, title_text='', showticklabels=False, row=1, col=1)
            fig.update_xaxes(showgrid=False, title_text='', showticklabels=False, row=2, col=2)
            fig.update_yaxes(range=lims, showgrid=True, gridcolor='white', title_text='', showticklabels=False, row=2,
                             col=2)

        # The main plot
        fig.update_xaxes(range=lims, scaleanchor="y", scaleratio=1, title_text=y.name, row=2 if marginal else 1, col=1)
        fig.update_yaxes(range=lims, title_text=y_est.name, row=2 if marginal else 1, col=1)

        fig.update_layout(
            title=title,
            showlegend=False,
            width=800, height=800,
        )

        return fig
