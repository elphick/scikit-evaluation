"""
Developed from the example here: https://plotly.com/python/pca-visualization/
"""
import dataclasses
import logging
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from elphick.sklearn_viz.utils import log_timer


def plot_principal_components(x: pd.DataFrame,
                              color: Optional[pd.Series] = None,
                              plot_3d: bool = True,
                              loading_vectors: bool = True,
                              standardised: bool = False,
                              title: Optional[str] = None) -> go.Figure:
    """

    Args:
        x: X values to transform and plot.
        color: optional series by which to color the markers
        plot_3d: If True plot the top 3 principal components in 3D, otherwise the top 2 in 2D.
        loading_vectors: If True and plot_type is '2D'|'3D' loading vectors will be displayed.
        standardised: If True, plot the standardised PCA, where vectors are transformed to zero mean and
             unit variance.
        title: Optional plot title

    Returns:
        a plotly GraphObjects.Figure

    """

    return PrincipalComponents(x=x, color=color).plot_principal_components(plot_3d=plot_3d,
                                                                           loading_vectors=loading_vectors,
                                                                           standardised=standardised,
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
    return PrincipalComponents(x=x, color=y).plot_explained_variance(title=title)


def plot_scatter_matrix(x: pd.DataFrame, y: Optional[pd.Series] = None,
                        original_features: bool = False, title: Optional[str] = None) -> go.Figure:
    """Plot a scatter matrix

    Args:
        x: X values to transform and plot.
        y: optional series by which to color the markers
        original_features: If True, plot the original features, otherwise plot the principal components.
        title: Optional plot title

    Returns:

    """
    return PrincipalComponents(x=x, color=y).plot_scatter_matrix(original_features=original_features, title=title)


def plot_loading_vectors(x: pd.DataFrame,
                         color: Optional[pd.Series] = None,
                         standardised: bool = False,
                         title: Optional[str] = None) -> go.Figure:
    """

    Args:
        x: X values to transform and plot.
        color: optional series by which to color the markers
        standardised: If True, plot the standardised PCA, where vectors are transformed to zero mean and
             unit variance.
        title: Optional plot title

    Returns:
        a plotly GraphObjects.Figure

    """

    return PrincipalComponents(x=x, color=color).plot_loading_vectors(standardised=standardised, title=title,
                                                                      by_color=color is not None)


def plot_correlation_circle(x: pd.DataFrame,
                            color: Optional[pd.Series] = None,
                            title: Optional[str] = None) -> go.Figure:
    """

    Args:
        x: X values to transform and plot.
        color: optional series by which to color the markers
        title: Optional plot title

    Returns:
        a plotly GraphObjects.Figure

    """

    return PrincipalComponents(x=x, color=color).plot_loading_vectors(standardised=True, title=title,
                                                                      by_color=color is not None)


@dataclasses.dataclass
class PCResults:
    """Class to hold Principal Component results"""
    data: pd.DataFrame
    explained_variance: pd.Series
    loadings: pd.DataFrame


class PrincipalComponents:
    def __init__(self, x: pd.DataFrame, color: Optional[pd.Series] = None):
        """

        Args:
            x: X values to transform and plot.
            color: the optional series by which to color the markers
        """
        self._logger = logging.getLogger(name=__class__.__name__)
        self.x: pd.DataFrame = x
        self.color: Optional[pd.Series] = color

        self._data: Optional[Dict] = None

    @property
    @log_timer
    def data(self) -> Optional[Dict]:
        def get_pca_results(pipe, x):
            xt: pd.DataFrame = pipe.fit_transform(x)
            xt.columns = [f"PC{i + 1}" for i in range(len(x.columns))]
            var: pd.Series = pd.Series(data=pipe['pca'].explained_variance_ratio_ * 100., name='explained_variance')
            loadings = pd.DataFrame(data=pipe['pca'].components_.T * np.sqrt(pipe['pca'].explained_variance_),
                                    index=x.columns, columns=xt.columns)
            return PCResults(data=xt, explained_variance=var, loadings=loadings)

        if self._data is not None:
            res = self._data
        else:
            res: Dict = {}
            self._logger.info("Commencing PCA")
            pca = make_pipeline(PCA()).set_output(transform="pandas")
            pca_std = make_pipeline(StandardScaler(), PCA()).set_output(transform="pandas")
            for label, pipe in {'raw': pca, 'std': pca_std}.items():
                res[label] = get_pca_results(pipe=pipe, x=self.x)
                if (self.color is not None) & (not is_numeric_dtype(self.color)):
                    for grp in self.color.unique():
                        if 'group' not in res.keys():
                            res['group'] = dict()
                        if grp not in res['group'].keys():
                            res['group'][grp] = dict()
                        res['group'][grp][label] = get_pca_results(pipe=pipe, x=self.x.loc[self.color == grp, :])
            self._data = res

        return res

    def plot_principal_components(self,
                                  plot_3d: bool = False,
                                  loading_vectors: bool = True,
                                  standardised: bool = False,
                                  title: Optional[str] = None) -> go.Figure:
        """Create the pca plot

        Args:
            plot_3d: If True plot the top 3 principal components in 3D, otherwise the top 2 in 2D.
            loading_vectors: If True and plot_type is '2D'|'3D' loading vectors will be displayed.
            standardised: If True, plot the standardised PCA, where vectors are transformed to zero mean and
             unit variance.
            title: Optional plot title

        Loading vectors are implemented manually rather than with annotations (lines with arrows),
         the problem is described well here:
         https://community.plotly.com/t/set-pca-loadings-aka-arrows-in-a-3d-scatter-plot/72905

        Returns:
            a plotly GraphObjects.Figure

        """
        label: str = 'std' if standardised else 'raw'
        pca_data: pd.DataFrame = self.data[label].data
        pca_loadings: pd.DataFrame = self.data[label].loadings
        pca_variance: pd.DataFrame = self.data[label].explained_variance

        df_plot: pd.DataFrame = pd.concat([pca_data, self.x], axis=1).reset_index()
        if plot_3d:
            fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3',
                                color=self.color, hover_data=list(self.x.reset_index().columns))
            fig.update_traces(marker_size=4)
            if loading_vectors:

                annots: List = [dict(x=row.PC1, y=row.PC2, z=row.PC3,
                                     text=i, showarrow=False,
                                     xanchor="left", xshift=10, yshift=10, opacity=0.7) for i, row in
                                pca_loadings.iterrows()]
                fig.update_layout(scene=dict(annotations=annots))
                for feature_name, row in pca_loadings.iterrows():
                    # noinspection PyTypeChecker
                    fig.add_trace(
                        go.Scatter3d(x=(row.PC1,), y=(row.PC2,), z=(row.PC3,), mode='markers',
                                     marker={'size': 6, 'line': dict(width=2, color='black')},
                                     name=feature_name,
                                     showlegend=True,
                                     legendgroup="features",
                                     legendgrouptitle_text="feature vectors",
                                     ))
                    fig.add_trace(
                        go.Scatter3d(x=(0, row.PC1), y=(0, row.PC2), z=(0, row.PC3), mode='lines',
                                     line={'width': 5, 'color': 'black'},
                                     name=feature_name,
                                     showlegend=False))
                fig.update_layout(legend=dict(groupclick="toggleitem"))
                title = (f"Top 3 Principal Components<br>Explained Variance = "
                         f"{round(pca_variance.iloc[0:3].sum(), 1)}%") if title is None else title
        else:  # 2D
            fig = px.scatter(df_plot, x='PC1', y='PC2',
                             color=self.color, hover_data=list(self.x.reset_index().columns))
            fig.update_traces(marker_size=5)

            if loading_vectors:
                loadings = pca_loadings.iloc[:, 0:2]
                self.add_loading_vectors(fig, loadings)
            title = (f"Top 2 Principal Components<br>Explained Variance = "
                     f"{round(pca_variance.iloc[0:2].sum(), 1)}%") if title is None else title

        fig.update_layout(legend_title_text=self.color.name)
        fig.update_layout(title=title,
                          xaxis_title=f"PC1 ({round(self.data['std'].explained_variance.iloc[0], 1)}%)",
                          yaxis_title=f"PC2 ({round(self.data['std'].explained_variance.iloc[1], 1)}%)")
        if self.color is not None:
            fig.update_layout(coloraxis_colorbar_title_text=self.color.name)

        return fig

    def plot_explained_variance(self,
                                standardised: bool = False,
                                title: Optional[str] = None) -> go.Figure:
        """Plot the cumulative explained variance by principal component.

        Args:
            standardised: If True, plot the standardised PCA, where vectors are transformed to zero mean and
             unit variance.
            title: Optional plot title

        Returns:

        """
        pca_variance: pd.DataFrame = self.data['std'].explained_variance if standardised else self.data[
            'raw'].explained_variance
        exp_var_cumul = np.cumsum(pca_variance)
        fig = px.area(
            x=range(1, exp_var_cumul.shape[0] + 1),
            y=exp_var_cumul,
            labels={"x": "# Components", "y": "Explained Variance"}
        )
        title = 'Cumulative Explained Variance by Principal Component' if title is None else title
        fig.update_layout(title=title)
        fig.update_xaxes(type='category')

        return fig

    def plot_scatter_matrix(self, original_features: bool = False, standardised: bool = False,
                            title: Optional[str] = None) -> go.Figure:
        """Plot a scatter matrix

        Args:
            original_features: If True, plot the original features, otherwise plot the principal components.
            standardised: If True, plot the standardised PCA, where vectors are transformed to zero mean and
             unit variance.
            title: Optional plot title

        Returns:

        """
        label: str = 'std' if standardised else 'raw'
        y = self.color
        if original_features:
            x = self.x
            title = 'Scatter Matrix - Original Feature Space' if title is None else title
        else:
            x = self.data[label].data
            title = 'Scatter Matrix - All Principal Components' if title is None else title

        if original_features:
            df_plot: pd.DataFrame = pd.concat([x, y], axis=1).reset_index()
            hover_data = ['index' if x.index.name is None else x.index.name]
        else:
            df_plot: pd.DataFrame = pd.concat([x, y, self.x], axis=1).reset_index()
            hover_data = list(self.x.reset_index().columns)

        fig = px.scatter_matrix(data_frame=df_plot, dimensions=list(x.columns),
                                color=y.name, hover_data=hover_data)
        fig.update_traces(diagonal_visible=False)
        title = 'Top 3 Principal Components' if title is None else title
        fig.update_layout(title=title)

        return fig

    def plot_loading_vectors(self, standardised: bool = False, by_color: bool = False,
                             title: Optional[str] = None) -> go.Figure:
        """plot the loading vectors.

        Args:
            standardised: If True, plot the standardised PCA, where vectors are transformed to zero mean and
             unit variance.
             by_color: If True, plot the loading vectors by color group.
            title: Optional plot title

        Returns:
            a plotly GraphObjects.Figure

        """
        label: str = 'std' if standardised else 'raw'
        if by_color:
            chunks = []

            for grp, d_label in self.data['group'].items():
                chunks.append(self.data['group'][grp][label].loadings.assign(group=grp))
            loadings = pd.concat(chunks, axis='index')
            fig = px.scatter(loadings, x='PC1', y='PC2', color='group',
                             hover_data=loadings.columns.tolist())

        else:
            loadings = self.data[label].loadings.iloc[:, 0:2]
            fig = px.scatter(loadings, x='PC1', y='PC2', hover_data=loadings.columns.tolist())
        fig.update_traces(marker=dict(size=1))

        if standardised:
            fig.add_shape(type="circle",
                          xref="x", yref="y",
                          x0=-1, y0=-1, x1=1, y1=1,
                          line_color="gray")
            title_main = "Correlation Circle"
        else:
            title_main = "Top 2 Principal Components"

        fig = self.add_loading_vectors(fig, loadings)
        title = (f"{title_main}<br>Explained Variance = "
                 f"{round(self.data['std'].explained_variance.iloc[0:2].sum(), 1)}%") if title is None else title

        fig.update_layout(title=title,
                          xaxis_title=f"PC1 ({round(self.data['std'].explained_variance.iloc[0], 1)}%)",
                          yaxis_title=f"PC2 ({round(self.data['std'].explained_variance.iloc[1], 1)}%)")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(scene=dict(aspectmode="data"))

        return fig

    def add_loading_vectors(self, fig, loadings) -> go.Figure:
        if 'group' in loadings.columns:
            cm = px.colors.qualitative.Plotly
            grp_colors = dict(zip(loadings['group'].unique(), cm[0:len(loadings['group'].unique())]))
        for i, feature in enumerate(loadings.index):
            if 'group' in loadings.columns:
                grp = loadings.iloc[i, :]['group']
                arrowcolor = grp_colors[grp]
                font = dict(color=grp_colors[grp])
            else:
                arrowcolor = None
                font = None
            fig.add_annotation(
                ax=0, ay=0,
                axref="x", ayref="y",
                x=loadings.iloc[i, 0],
                y=loadings.iloc[i, 1],
                showarrow=True,
                arrowsize=2,
                arrowhead=2,
                xanchor="right",
                yanchor="top",
                arrowcolor=arrowcolor,
            )
            fig.add_annotation(
                x=loadings.iloc[i, 0],
                y=loadings.iloc[i, 1],
                ax=0, ay=0,
                xanchor="center",
                yanchor="bottom",
                text=feature,
                yshift=5,
                font=font
            )
        return fig
