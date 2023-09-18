from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_scatter_matrix(x: pd.DataFrame, color: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    """Plot a scatter matrix

    Args:
        x: X values to transform and plot.
        color: optional target vector
        title: Optional plot title

    Returns:

    """

    fig = px.scatter_matrix(data_frame=x, dimensions=list(x.columns), color=color)
    fig.update_traces(diagonal_visible=False)
    title = 'Scatter Matrix' if title is None else title
    fig.update_layout(title=title)

    return fig
