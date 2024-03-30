import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2
import plotly.graph_objects as go

from elphick.sklearn_viz.features import PrincipalComponents
from elphick.sklearn_viz.features.principal_components import PCResults
from elphick.sklearn_viz.features.scatter_matrix import plot_scatter_matrix
from elphick.sklearn_viz.utils import log_timer


def mahalanobis(x: pd.DataFrame, data: Optional[pd.DataFrame] = None, cov=None) -> pd.DataFrame:
    if data is None:
        data = x
    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T).diagonal()
    pvals = 1 - chi2.cdf(mahal, len(x.columns) - 1)
    res: pd.DataFrame = pd.DataFrame(np.vstack((mahal, pvals)).T, columns=['mahal_dist', 'p_val'], index=x.index)
    return res


def plot_outlier_matrix(x: pd.DataFrame, pca_spec: Union[float, int] = 0, p_val: float = 0.001,
                        principal_components: bool = False) -> go.Figure:
    """Detect and plot outliers

    Args:
        x: X values for outlier detection.
        pca_spec: If zero, pca is not used.  For integers (n) > 0 outlier detection is performed on the
         top n principal components. For values (f) < 1, outlier detection is performed on the number of
         principal components that explain f% of the variance.
        p_val: the p-value threshold for outlier detection.
        principal_components: If True (and pca_spec is not 0) the principal components will be plotted.  Otherwise,
         will plot in the original feature space.
    """
    return OutlierDetection(x=x, pca_spec=pca_spec, p_val=p_val).plot_outlier_matrix(
        principal_components=principal_components)


class OutlierDetection:
    def __init__(self, x: pd.DataFrame, pca_spec: Union[float, int] = 0,
                 standardise: bool = False, p_val: float = 0.001):
        """

        Args:
            x: X values for outlier detection.
            pca_spec: If zero, pca is not used.  For integers (n) > 0 outlier detection is performed on the
             top n principal components. For values (f) < 1, outlier detection is performed on the number of
             principal components that explain f% of the variance.
            standardise: If True, standardise the data prior to PCA, where vectors are transformed to zero mean and
             unit variance.
            p_val: the p-value threshold for outlier detection.
        """
        self._logger = logging.getLogger(name=__class__.__name__)
        self.x: pd.DataFrame = x
        self.pca_spec: Union[float, int] = pca_spec
        self.standardise: bool = standardise
        self.p_val: float = p_val

        self._data: Optional[Dict] = None

    @property
    @log_timer
    def data(self) -> Optional[Dict]:
        if self._data is not None:
            res = self._data
        else:
            label: str = 'std' if self.standardise else 'raw'
            res: Dict = {}
            if self.pca_spec != 0:
                res['pca'] = PrincipalComponents(self.x)
                pca_data: PCResults = res['pca'].data[label]
                if self.pca_spec >= 1:
                    mahal = mahalanobis(x=pca_data.data.iloc[:, 0:self.pca_spec])
                elif self.pca_spec < 1:
                    num_required: int = next(i for i, v in
                                             enumerate(pca_data.explained_variance.cumsum() / 100 >= self.pca_spec) if
                                             v is True) + 1
                    mahal = mahalanobis(x=pca_data.data.iloc[:, 0:num_required])
                else:
                    raise ValueError("pca_spec cannot be negative")
            else:
                mahal = mahalanobis(x=self.x)

            res['mahal'] = mahal
            res['outlier'] = pd.Series(res['mahal']['p_val'] < self.p_val, name='outlier')
            self._data = res
        return res

    def plot_outlier_matrix(self, principal_components: bool = False) -> go.Figure:
        if principal_components:
            if 'pca' in self.data.keys():
                fig = self.data['pca'].plot_scatter_matrix(original_features=True, y=self.data['outlier'])
            else:
                raise ValueError("Outliers not defined using PCA.  Try changing pca_spec.")
        else:
            fig = plot_scatter_matrix(x=pd.concat([self.x, self.data['outlier']], axis=1), color='outlier',
                                      title="Outlier Scatter Matrix")
        return fig
