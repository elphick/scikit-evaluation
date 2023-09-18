"""
Developed from the example here: https://plotly.com/python/pca-visualization/
"""

import logging
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

from elphick.sklearn_viz.utils import log_timer


def plot_principal_components(x: pd.DataFrame,
                              y: Optional[pd.Series] = None,
                              plot_3d: bool = True,
                              loading_vectors: bool = True,
                              title: Optional[str] = None) -> go.Figure:
    """

    Args:
        x: X values to transform and plot.
        y: optional target vector
        plot_3d: If True plot the top 3 principal components in 3D, otherwise the top 2 in 2D.
        loading_vectors: If True and plot_type is '2D'|'3D' loading vectors will be displayed.
        title: Optional plot title

    Returns:
        a plotly GraphObjects.Figure

    """

    return PrincipalComponents(x=x, y=y).plot_principal_components(plot_3d=plot_3d, loading_vectors=loading_vectors,
                                                                   title=title)


def plot_explained_variance(x: pd.DataFrame, y: Optional[pd.Series] = None,
                            title: Optional[str] = None) -> go.Figure:
    """Plot the cumulative explained variance by principal component.

    Args:
        x: X values to transform and plot.
        y: optional target vector
        title: Optional plot title

    Returns:

    """
    return PrincipalComponents(x=x, y=y).plot_explained_variance(title=title)


def plot_scatter_matrix(x: pd.DataFrame, y: Optional[pd.Series] = None,
                        original_features: bool = False, title: Optional[str] = None) -> go.Figure:
    """Plot a scatter matrix

    Args:
        x: X values to transform and plot.
        y: optional target vector
        original_features: If True, plot the original features, otherwise plot the principal components.
        title: Optional plot title

    Returns:

    """
    return PrincipalComponents(x=x, y=y).plot_scatter_matrix(original_features=original_features, title=title)


class PrincipalComponents:
    def __init__(self, x: pd.DataFrame, y: Optional[pd.Series] = None):
        """

        Args:
            x: X values to transform and plot.
            y: the optional target vector.
        """
        self._logger = logging.getLogger(name=__class__.__name__)
        self.x: pd.DataFrame = x
        self.y: Optional[pd.Series] = y

        self._data: Optional[Dict] = None

    @property
    @log_timer
    def data(self) -> Optional[Dict]:
        if self._data is not None:
            res = self._data
        else:
            res: Dict = {}
            self._logger.info("Commencing PCA")
            pc_names = [f"PC{i}" for i in range(1, len(self.x.columns) + 1)]
            pca = PCA()
            data: pd.DataFrame = pd.DataFrame(data=pca.fit_transform(self.x),
                                              columns=pc_names)
            var: pd.Series = pd.Series(data=pca.explained_variance_ratio_ * 100., name='explained_variance')
            res['var'] = var
            dim_names = ['x', 'y', 'z'] + [f"dim{i + 1}" for i in range(3, len(self.x.columns))]
            res['loadings'] = pd.DataFrame(data=pca.components_.T * np.sqrt(pca.explained_variance_),
                                           index=self.x.columns, columns=dim_names)
            res['data'] = data
            self._data = res

        return res

    def plot_principal_components(self,
                                  plot_3d: bool = False,
                                  loading_vectors: bool = True,
                                  title: Optional[str] = None) -> go.Figure:
        """Create the pca plot

        Args:
            plot_3d: If True plot the top 3 principal components in 3D, otherwise the top 2 in 2D.
            loading_vectors: If True and plot_type is '2D'|'3D' loading vectors will be displayed.
            title: Optional plot title

        Loading vectors are implemented manually rather than with annotations (lines with arrows),
         the problem is described well here:
         https://community.plotly.com/t/set-pca-loadings-aka-arrows-in-a-3d-scatter-plot/72905

        Returns:
            a plotly GraphObjects.Figure

        """
        if plot_3d:
            fig = px.scatter_3d(
                self.data['data'], x='PC1', y='PC2', z='PC3', color=self.y,
            )
            if loading_vectors:

                annots: List = [dict(x=row.x, y=row.y, z=row.z,
                                     text=i, showarrow=False,
                                     xanchor="left", xshift=10, yshift=10, opacity=0.7) for i, row in
                                self.data['loadings'].iterrows()]
                fig.update_layout(scene=dict(annotations=annots))
                for feature_name, row in self.data['loadings'].iterrows():
                    # noinspection PyTypeChecker
                    fig.add_trace(
                        go.Scatter3d(x=(row.x,), y=(row.y,), z=(row.z,), mode='markers',
                                     marker={'size': 10, 'line': dict(width=2, color='black')},
                                     name=feature_name,
                                     showlegend=True,
                                     legendgroup="features",
                                     legendgrouptitle_text="feature vectors",
                                     ))
                    fig.add_trace(
                        go.Scatter3d(x=(0, row.x), y=(0, row.y), z=(0, row.z), mode='lines',
                                     line={'width': 5, 'color': 'black'},
                                     name=feature_name,
                                     showlegend=False))
                fig.update_layout(legend=dict(groupclick="toggleitem"))
                title = (f"Top 3 Principal Components<br>Explained Variance = "
                         f"{round(self.data['var'].iloc[0:3].sum(), 1)}%") if title is None else title
        else:  # 2D
            fig = px.scatter(self.data['data'], x='PC1', y='PC2', color=self.y)
            if loading_vectors:
                loadings = self.data['loadings'].iloc[:, 0:2]
                for i, feature in enumerate(loadings.index):
                    fig.add_annotation(
                        ax=0, ay=0,
                        axref="x", ayref="y",
                        x=loadings.iloc[i, 0],
                        y=loadings.iloc[i, 1],
                        showarrow=True,
                        arrowsize=2,
                        arrowhead=2,
                        xanchor="right",
                        yanchor="top"
                    )
                    fig.add_annotation(
                        x=loadings.iloc[i, 0],
                        y=loadings.iloc[i, 1],
                        ax=0, ay=0,
                        xanchor="center",
                        yanchor="bottom",
                        text=feature,
                        yshift=5,
                    )
            title = (f"Top 2 Principal Components<br>Explained Variance = "
                     f"{round(self.data['var'].iloc[0:2].sum(), 1)}%") if title is None else title

        fig.update_layout(legend_title_text=self.y.name)
        fig.update_layout(title=title)
        if self.y is not None:
            fig.update_layout(coloraxis_colorbar_title_text=self.y.name)

        return fig

    def plot_explained_variance(self, title: Optional[str] = None) -> go.Figure:
        """Plot the cumulative explained variance by principal component.

        Args:
            title: Optional plot title

        Returns:

        """
        exp_var_cumul = np.cumsum(self.data['var'])
        fig = px.area(
            x=range(1, exp_var_cumul.shape[0] + 1),
            y=exp_var_cumul,
            labels={"x": "# Components", "y": "Explained Variance"}
        )
        title = 'Cumulative Explained Variance by Principal Component' if title is None else title
        fig.update_layout(title=title)
        fig.update_xaxes(type='category')

        return fig

    def plot_scatter_matrix(self, original_features: bool = False, title: Optional[str] = None) -> go.Figure:
        """Plot a scatter matrix

        Args:
            original_features: If True, plot the original features, otherwise plot the principal components.
            title: Optional plot title

        Returns:

        """
        y = self.y
        if original_features:
            x = self.x
            title = 'Scatter Matrix - Original Feature Space' if title is None else title
        else:
            x = self.data['data']
            title = 'Scatter Matrix - All Principal Components' if title is None else title

        fig = px.scatter_matrix(
            data_frame=pd.concat([x, y], axis=1),
            dimensions=list(x.columns),
            color=y.name
        )
        fig.update_traces(diagonal_visible=False)
        title = 'Top 3 Principal Components' if title is None else title
        fig.update_layout(title=title)

        return fig
