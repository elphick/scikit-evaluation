import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection import KFold, cross_validate

from elphick.sklearn_viz.model_selection.metrics import classification_metrics, regression_metrics
from elphick.sklearn_viz.model_selection.scorers import classification_scorers, regression_scorers


@dataclass
class CrossValidationResult:
    test_scores: List[float]
    train_scores: List[float]
    fit_times: List[float]
    score_times: List[float]
    estimator: List[Any]
    metrics: Dict[str, List[float]]
    metrics_group: Dict[str, Dict[str, List[float]]]


class CrossValidatorBase(ABC):
    def __init__(self,
                 estimators: Union[BaseEstimator, Dict[str, BaseEstimator]],
                 datasets: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 target: str,
                 pre_processor: Optional[Any],
                 cv: Union[int, Any],
                 scorer: Any,
                 metrics: Optional[Dict[str, Any]],
                 group: Any,
                 random_state: int):

        # If algorithms is not a dictionary, convert it into a dictionary with a default key
        if not isinstance(estimators, dict):
            estimators = {'estimator': estimators}

        # If datasets is not a dictionary, convert it into a dictionary with a default key
        if not isinstance(datasets, dict):
            datasets = {'dataset': datasets}

        # Check if all estimators are classifiers or regressors
        self.is_classifier = all(is_classifier(estimator) for estimator in estimators.values())
        self.is_regressor = all(is_regressor(estimator) for estimator in estimators.values())

        if not self.is_classifier and not self.is_regressor:
            raise ValueError("All estimators must be either classifiers or regressors.")

        if scorer is None:
            default_scorer = 'accuracy' if self.is_classifier else 'r2'
            scorer_dict = classification_scorers if self.is_classifier else regression_scorers
            scorer = scorer_dict.get(default_scorer)

        if metrics is None:
            default_metrics = {
                'classification': classification_metrics,
                'regression': regression_metrics
            }
            metrics_type = 'classification' if self.is_classifier else 'regression'
            metrics = default_metrics.get(metrics_type)

        self.estimators: Dict[str, BaseEstimator] = estimators
        self.datasets: Dict[str, pd.DataFrame] = datasets
        self.target: str = target
        self.pre_processor: Optional[Any] = pre_processor
        self.cv: Union[int, Any] = cv
        self.scorer: Any = scorer
        self.metrics: Dict[str, Any] = metrics
        self.group: Any = group
        self.random_state: int = random_state

        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self._results: Optional[CrossValidationResult] = None

        self.features_in: List[str] = [col for col in self.datasets[list(self.datasets.keys())[0]] if
                                       col != self.target]

        self._data: Optional[Dict] = None
        self._num_algorithms: int = len(list(self.estimators.keys()))
        self._num_datasets: int = len(list(self.datasets.keys()))

    @property
    def results(self) -> Optional[Dict[str, Dict[str, CrossValidationResult]]]:
        if self._results is None:
            if self.metrics is None:
                cv_kwargs: Dict = dict()
            else:
                cv_kwargs: Dict = dict(return_estimator=True, return_indices=True, error_score=np.nan)

            d_results: Dict = {data_key: {algo_key: {} for algo_key in self.estimators.keys()} for data_key in
                               self.datasets.keys()}
            for data_key, data in self.datasets.items():
                self._logger.info(f"Commencing Cross Validation for dataset {data_key}")
                d_results[data_key] = {}
                # Ensure that only the features present in the dataset are used
                features_in_dataset = list(set(self.features_in).intersection(set(data.columns)))
                x: pd.DataFrame = data[features_in_dataset]
                y: pd.DataFrame = data[self.target]
                if self.pre_processor:
                    x = self.pre_processor.set_output(transform="pandas").fit_transform(X=x)

                for estimator_key, estimator in self.estimators.items():
                    if isinstance(self.cv, int):
                        cv = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
                    else:
                        cv = self.cv
                    res = cross_validate(estimator, x, y, cv=cv, scoring=self.scorer, return_train_score=True,
                                         **cv_kwargs)
                    if self.metrics is not None:
                        res['metrics'], res['metrics_group'] = self.calculate_metrics(x=x, y=y,
                                                                                      estimators=res['estimator'],
                                                                                      indices=res['indices'],
                                                                                      group=self.group)
                    # Convert the results to a CrossValidationResult instance and assign it
                    d_results[data_key][estimator_key] = CrossValidationResult(
                        test_scores=res['test_score'],
                        train_scores=res['train_score'],
                        fit_times=res['fit_time'],
                        score_times=res['score_time'],
                        estimator=res['estimator'],
                        metrics=res['metrics'],
                        metrics_group=res['metrics_group']
                    )

                    res_mean = res[f"test_score"].mean()
                    res_std = res[f"test_score"].std()

                    self._logger.info(f"CV Results for {estimator_key}: Mean = {res_mean}, SD = {res_std}")

            self._results = d_results

        return self._results

    @abstractmethod
    def get_cv_scores(self):
        chunks: List = []
        for data_key, data in self.datasets.items():
            for estimator_key, estimator in self.estimators.items():
                chunks.append(
                    pd.Series(self.results[data_key][estimator_key].test_scores, name=(data_key, estimator_key)))
        return pd.concat(chunks, axis=1)

    @abstractmethod
    def get_cv_metrics(self, metrics, by_group: bool = False) -> pd.DataFrame:
        chunks: List = []
        metric_key = "metrics_group" if by_group else "metrics"
        for data_key, data in self.datasets.items():
            for estimator_key, estimator in self.estimators.items():
                for metric in metrics:
                    # Access the metrics or metrics_group property of the CrossValidationResult instance
                    metric_data = getattr(self.results[data_key][estimator_key], metric_key)[metric]
                    chunks.append(pd.DataFrame(metric_data).assign(
                        **dict(data_key=data_key, algo_key=estimator_key, metric=metric)))
        res: pd.DataFrame = pd.concat(chunks, axis=0).set_index(['data_key', 'algo_key'], append=True).rename(
            columns={0: 'value'})
        res.index.names = ['fold', 'data_key', 'algo_key']
        return res

    @abstractmethod
    def calculate_metrics(self, x, y, estimators, indices, group):
        metric_results: Dict = {}
        metric_results_group: Dict = {}

        for k, fn_metric in self.metrics.items():
            metric_values: List = []
            metric_groups: Dict = {}
            for estimator, test_indexes in zip(estimators, indices['test']):
                y_true = y[y.index[test_indexes]]
                y_est = estimator.predict(x.loc[x.index[test_indexes], :])
                if isinstance(y_est, pd.DataFrame) and y_est.shape[1] == 1:
                    y_est = y_est.iloc[:, 0]
                metric_values.append(fn_metric(y_true, y_est))
                if group is not None:
                    # calculate the metric by each group in the group series.
                    y_est = pd.merge(left=pd.Series(y_est, name='y_est', index=x.index[test_indexes]),
                                     right=group, left_index=True, right_index=True)
                    y_est_grouped = y_est.groupby([group.name])
                    grouped_results = [y_est_grouped.get_group(x) for x in y_est_grouped.groups]
                    for grp_res in grouped_results:
                        group_value = str(grp_res[group.name].iloc[0])
                        group_metric_results = fn_metric(y_true[grp_res.index], grp_res['y_est'].values)
                        if group_value not in metric_groups.keys():
                            metric_groups[group_value] = [group_metric_results]
                        else:
                            metric_groups[group_value].append(group_metric_results)
            metric_results[k] = metric_values
            if group is not None:
                metric_results_group[k] = metric_groups

        return metric_results, metric_results_group

    def get_model_by_group_data(self, estimator, dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
        results: dict = {}
        by_group_score_chunks: list = []
        by_group_metric_chunks: list = []
        for grp in self.group.unique():
            grp_index: pd.Index = self.group.loc[self.group == grp].index
            x: pd.DataFrame = self.datasets[dataset][self.features_in].loc[grp_index]
            y: pd.DataFrame = self.datasets[dataset][self.target].loc[grp_index]
            if self.pre_processor:
                x = self.pre_processor.set_output(transform="pandas").fit_transform(X=x)
            if isinstance(self.cv, int):
                cv = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
            else:
                cv = self.cv
            res = cross_validate(self.estimators[estimator], x, y, cv=cv, scoring=self.scorer, return_estimator=True,
                                 return_indices=True)
            by_group_score_chunks.append(pd.Series(res['test_score'], name=grp))
            if self.metrics is not None:
                res['metrics'], _ = self.calculate_metrics(x=x, y=y,
                                                           estimators=res['estimator'],
                                                           indices=res['indices'],
                                                           group=None)
            results[grp] = res
            by_group_metric_chunks.append(
                pd.DataFrame(res['metrics']).assign(group=grp).rename_axis('fold', axis='index'))
        by_group_metrics: pd.DataFrame = pd.concat(by_group_metric_chunks)
        by_group_scores = pd.concat(by_group_score_chunks, axis=1)

        return by_group_metrics, by_group_scores

    @abstractmethod
    def plot(self, metrics=None, show_group=False, title=None, col_wrap=None):
        pass
