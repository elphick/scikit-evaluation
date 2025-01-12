import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin

from elphick.sklearn_viz.features import PrincipalComponents
from elphick.sklearn_viz.features.principal_components import PCResults
from elphick.sklearn_viz.features.scatter_matrix import plot_scatter_matrix
from elphick.sklearn_viz.utils import log_timer

from dataclasses import dataclass
import pandas as pd


@dataclass
class OutlierMetrics:
    mahal_dist: pd.Series
    p_val: pd.Series

    def is_outlier(self, outlier_threshold: float, possible_outlier_threshold: Optional[float]) -> pd.Series:
        outlier = np.where(self.p_val < possible_outlier_threshold, 'Possible',
                           'False') if possible_outlier_threshold is not None else np.full_like(self.p_val, 'False')
        outlier = np.where(self.p_val < outlier_threshold, 'True', outlier)
        outlier = pd.Series(outlier, index=self.p_val.index, name='outlier_class')
        outlier = pd.Series(pd.Categorical(outlier, categories=['False', 'Possible', 'True'], ordered=True))
        return outlier


# @dataclass
# class OutlierResults:
#     x_fit: OutlierMetrics
#     y_fit: OutlierMetrics
#     x_transform: OutlierMetrics


@dataclass
class OutlierResults:
    space: str  # name of the "space" the analysis was performed in.
    pca_spec: Union[float, int]
    x_fit: OutlierMetrics
    y_fit: OutlierMetrics
    x_transform: OutlierMetrics

    def __post_init__(self):
        valid_names = ['pca', 'feature', 'target']
        if self.space not in valid_names:
            raise ValueError(f"name must be one of {valid_names}, but got {self.space}")


def mahalanobis(x: pd.DataFrame, data: Optional[pd.DataFrame] = None, cov=None) -> pd.DataFrame:
    if data is None:
        data = x
    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
        # Regularization:
        cov += np.eye(cov.shape[0]) * 1e-6
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


class OutlierDetection(BaseEstimator, TransformerMixin):
    def __init__(self, pca_spec: Union[float, int] = 0,
                 p_val_probable: float = 0.001,
                 p_val_possible: Optional[float] = 0.01):
        """

        Args:
            pca_spec: If zero, pca is not used.  For integers (n) > 0 outlier detection is performed on the
             top n principal components. For values (f) < 1, outlier detection is performed on the number of
             principal components that explain f% of the variance.
            p_val_probable: the p-value threshold for detection of probable outliers
            p_val_possible: the p-value threshold for detection of possible outliers

        """
        self._logger = logging.getLogger(name=__class__.__name__)
        self.pca_spec: Union[float, int] = pca_spec
        self.p_val_probable: float = p_val_probable
        self.p_val_possible: Optional[float] = p_val_possible

        self.results_ = None

        self._data: Optional[Dict] = None

    @property
    @log_timer
    def data(self) -> Optional[Dict]:
        if self._data is not None:
            res = self._data
        else:
            res: Dict = {}
            if self.pca_spec != 0:
                res['pca'] = PrincipalComponents(self.x)
                pca_data: PCResults = res['pca'].data['std']
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

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None):
        res: Dict = {}
        if self.pca_spec != 0:
            res['pca'] = PrincipalComponents(X)
            pca_data: PCResults = res['pca'].data['std']
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

        return self

    def transform(self, X: pd.DataFrame):
        return self

    def _fit_outlier_detection(self, data: Union[pd.DataFrame, pd.Series], pval: float, data_name: str):
        # detect potential outliers in features or targets
        res = self._outlier_detection(data, outlier_threshold=pval, possible_outlier_threshold=None)
        potential_outliers: pd.Series = res['outlier'].map({'True': True, 'Possible': False, 'False': False})
        self._logger.info(f"Detected {potential_outliers.sum()}, {potential_outliers.sum() / len(data) * 100}%"
                          f" potential outliers from the {data_name} data")
        return potential_outliers

    def _outlier_detection(self, X, outlier_threshold, possible_outlier_threshold):
        res: pd.DataFrame = mahalanobis(x=X, data=self.X_).rename(
            columns={'mahal_dist': 'outlier_mahal', 'p_val': 'outlier_pval'})
        res['outlier'] = 'False'
        if possible_outlier_threshold is not None:
            res['outlier'] = np.where(res['outlier_pval'] < possible_outlier_threshold, 'Possible', res['outlier'])
        res['outlier'] = np.where(res['outlier_pval'] < outlier_threshold, 'True', res['outlier'])
        res['outlier'] = pd.Categorical(res['outlier'], categories=['False', 'Possible', 'True'], ordered=True)
        outlier_counts = res['outlier'].value_counts()
        outlier_percentages = (outlier_counts / outlier_counts.sum() * 100).round(2)
        self._logger.info(
            f"Detected outliers: {outlier_counts.to_dict()}, percentages: {outlier_percentages.to_dict()}")
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
