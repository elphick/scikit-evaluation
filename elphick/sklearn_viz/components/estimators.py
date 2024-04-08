import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils._estimator_html_repr import _VisualBlock
from sklearn.utils.validation import check_is_fitted


class PartitionRegressor(RegressorMixin, BaseEstimator):
    """Prediction for partitioned subsets of records (estimation domains) defined by filter criteria.

    A PartitionRegressor is designed to allow multiple models to be fitted on a subset of records defined by
     a criteria applied to the input features.  So in essence the criteria definitions define estimation domains
    (a.k.a. partitions) for individual model fitting.
    """

    def __init__(self, estimator, *, partition_defs: dict[str, str], n_jobs=None, verbose=False):
        super(RegressorMixin).__init__()
        super(BaseEstimator).__init__()

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.estimator = estimator

        self.estimators: list[tuple] = [(k, estimator) for k in partition_defs.keys()]
        self.partition_defs = partition_defs

        # set post fitting
        self.x_train_: pd.DataFrame
        self.y_train_: pd.DataFrame
        self.domain_indexes_: dict
        self.feature_names_in_: list

    def score(self, X, y, sample_weight=None):
        pass

    def fit(self, X: pd.DataFrame, y: [pd.DataFrame, None] = None) -> 'PartitionRegressor':
        # fit the estimators only on the appropriate records
        self.x_train_ = X
        self.y_train_ = y if isinstance(y, pd.DataFrame) else y.to_frame()
        domain_indexes: dict = dict()
        domains: pd.Series = pd.Series(np.nan, index=X.index, name='domains')
        for domain, criteria in self.partition_defs.items():
            mdl = [est[1] for est in self.estimators if est[0] == domain][0]
            xf = X.query(criteria)
            domain_indexes[domain] = xf.index
            if len(domain_indexes[domain]) > 0:
                mdl.fit(xf, y.loc[xf.index])
                domains[xf.index] = domain
        domains = domains.astype('category')
        self.domain_indexes_ = domain_indexes
        self.feature_names_in_ = X.columns.to_list()
        self.domains_ = domains
        self.undomained_ = X.loc[domains.isna() == True]
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # predict on records that match the criteria for each model only and merge.
        # nans will be present in the result of the domains do not provide full coverage
        chunks: list = []
        for domain, criteria in self.partition_defs.items():
            mdl = [est[1] for est in self.estimators if est[0] == domain][0]
            xf = X.query(criteria)
            if len(self.domain_indexes_[domain]) > 0:
                check_is_fitted(mdl)
                chunks.append(
                    pd.DataFrame(mdl.predict(xf), index=xf.index, columns=self.y_train_.columns).assign(domain=domain))
            else:
                # this model had no data to fit on, so return nans
                chunks.append(pd.DataFrame(np.nan, index=xf.index, columns=self.y_train_.columns).assign(domain=domain))
        res: pd.DataFrame = pd.merge(left=pd.Series(X.index, index=X.index, name='dummy'),
                                     right=pd.concat(chunks, axis=0),
                                     left_index=True, right_index=True,
                                     how='left', ).drop(columns=['dummy', 'domain'])
        # res['domain'] = res['domain'].astype('category')
        return res

    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        return self.estimators_[0].n_features_in_

    def _sk_visual_block_(self):
        names, estimators = zip(*self.estimators)
        criteria = list(self.partition_defs.values())
        return _VisualBlock("parallel", estimators, names=names, name_details=criteria)
