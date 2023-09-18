from typing import Optional, List

import pandas as pd

import plotly.graph_objects as go


def plot_parallel_coordinates(data: pd.DataFrame,
                              color: Optional[str] = None,
                              vars_include: Optional[List[str]] = None,
                              vars_exclude: Optional[List[str]] = None,
                              title: Optional[str] = None,
                              ) -> go.Figure:
    """Create an interactive parallel plot

    Useful to explore multidimensional data like mass-composition data

    Args:
        data: The DataFrame to plot
        color: The variable that sets the color, typically the target variable
        vars_include: Optional List of variables to include in the plot
        vars_exclude: Optional List of variables to exclude in the plot
        title: Optional plot title

    Returns:

    """
    df: pd.DataFrame = data.copy()
    if vars_include is not None:
        missing_vars = set(vars_include).difference(set(df.columns))
        if len(missing_vars) > 0:
            raise KeyError(f'vars_include provided contains variable not found in the data: {missing_vars}')
        df = df[vars_include]
    if vars_exclude:
        df = df[[col for col in df.columns if col not in vars_exclude]]

    # Kudos: https://stackoverflow.com/questions/72125802/parallel-coordinate-plot-in-plotly-with-continuous-
    # and-categorical-data

    categorical_columns = df.select_dtypes(include=['category', 'object'])
    col_list = []

    for col in df.columns:
        if col in categorical_columns:  # categorical columns
            values = data[col].unique()
            value2dummy = dict(zip(values, range(
                len(values))))  # works if values are strings, otherwise we probably need to convert them
            data[col] = [value2dummy[v] for v in data[col]]
            col_dict = dict(
                label=col,
                tickvals=list(value2dummy.values()),
                ticktext=list(value2dummy.keys()),
                values=data[col],
            )
        else:  # continuous columns
            col_dict = dict(
                range=(data[col].min(), data[col].max()),
                label=col,
                values=data[col],
            )
        col_list.append(col_dict)

    if color is None:
        fig = go.Figure(data=go.Parcoords(dimensions=col_list))
    else:
        fig = go.Figure(data=go.Parcoords(dimensions=col_list, line=dict(color=data[color])))

    fig.update_layout(title=title, height=700)

    return fig
