import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from elphick.sklearn_viz.features.outlier_detection import mahalanobis, OutlierDetection


class OutlierPipeline(Pipeline):

    def __init__(self, steps, *, name: str = 'Outlier Pipeline', memory=None, verbose=False,
                 pca_spec: Union[float, int] = 0, standardise: bool = False, p_val: float = 0.001):
        super().__init__(steps, memory=memory, verbose=verbose)
        self.name = name
        self._logger = logging.getLogger(self.__class__.__name__)
        self.outlier_detection = OutlierDetection(pca_spec=pca_spec, standardise=standardise, p_val=p_val)

        # available after fitting
        self.X_ = None  # Feature dataframe used for fitting
        self.y_ = None  # Target dataframe used for fitting
        self.Xf_ = None  # The filtered X that the final fit is performed on.
        self.yf_ = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame] = None,
            filter_outliers: bool = True, pval: float = 0.001, **kwargs):
        """Fit the pipleine with outlier detection and removal

        Potential outliers are detected using the Mahalanobis distance.  The model is fitted with these points
        removed.  The model is then refitted with the full dataset, and outliers are detected in the residual space.
        The pipeline removes residual outliers, which are also potential outliers, before performing the final model fit.

        Args:
            X: Features
            y: Targets
            filter_outliers: If True, the pipeline removes outliers from the training data.
            pval: p-value threshold for outlier detection. The default is 0.001.
            **kwargs:

        Returns:

        """

        self.X_ = X.copy()
        self.y_ = y

        if filter_outliers:
            # detect and remove potential outliers in the features and targets
            potential_outliers_features = self._fit_outlier_detection(X, pval, data_name='features')
            potential_outliers_targets = self._fit_outlier_detection(y, pval, data_name='targets')
            potential_outliers = potential_outliers_features | potential_outliers_targets
            X = X.loc[~potential_outliers]
            y = y.loc[~potential_outliers]

            # fit the model with the filtered data, and collect the residuals
            super().fit(X, y, **kwargs)
            residuals = y - super().predict(X)

            # detect outliers in the residuals, and then remove final outliers
            residual_outliers = self._fit_outlier_detection(pd.concat([y, residuals], axis=1), pval,
                                                            data_name='residuals')
            final_outliers = potential_outliers | residual_outliers
            X, y = X.loc[~final_outliers], y.loc[~final_outliers]

            self.Xf_ = X
            self.yf_ = y

        return super().fit(X, y, **kwargs)

    def _fit_outlier_detection(self, data: Union[pd.DataFrame, pd.Series], pval: float, data_name: str):
        # detect potential outliers in features or targets
        res = self._predict_outlier_detection(data, outlier_threshold=pval, possible_outlier_threshold=None)
        potential_outliers: pd.Series = res['outlier'].map({'True': True, 'Possible': False, 'False': False})
        self._logger.info(f"Detected {potential_outliers.sum()}, {potential_outliers.sum() / len(data) * 100}%"
                          f" potential outliers from the {data_name} data")
        return potential_outliers

    def predict(self, X: pd.DataFrame, with_outlier_metrics: bool = False,
                outlier_threshold: float = 0.001, possible_outlier_threshold: Optional[float] = 0.01,
                return_all_outlier_metrics: bool = False):
        """Predict using the pipeline with outlier detection

        The usual predict method with optional arguments to return the outlier metrics related to the features matrix
        passed in for prediction.  Useful, not to indicate the quality of the prediction, but flags if the X records
        are outliers with respect to the training data.  If they are the model performance is likely to be poor.

        Args:
            X: X values for prediction.
            with_outlier_metrics: If True, outlier metrics are returned.
            outlier_threshold: p-value threshold for being considered an outlier. The default is 0.001.  Only
             applicable when with_outlier_metrics is True.
            possible_outlier_threshold: p-value threshold for being considered a possible outlier.  If None,
             he 'outlier' column will only contain 'True' or 'False'. Only applicable when with_outlier_metrics is True.
            return_all_outlier_metrics: If True, the outlier metrics 'outlier_mahal', 'outlier_pval' and 'outlier'
             columns will be returned. If False, only the 'outlier' column is returned. Only applicable when
             the with_outlier_metrics is True.

            """
        y_est = pd.DataFrame(super().predict(X), columns=self.y_.columns, index=X.index)
        if with_outlier_metrics:
            res = self._predict_outlier_detection(X, outlier_threshold, possible_outlier_threshold)
            if not return_all_outlier_metrics:
                res = res['outlier']
            y_est = pd.concat([y_est, res], axis=1)
        return y_est

    def _predict_outlier_detection(self, X, outlier_threshold, possible_outlier_threshold):
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

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline, name: str = 'Pandas Pipeline'):
        return cls(pipeline.steps, memory=pipeline.memory, verbose=pipeline.verbose, name=name)
