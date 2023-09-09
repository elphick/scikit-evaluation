from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Dict

import sklearn.base
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class ModelType(Enum):
    REGRESSOR = 'regressor'
    CLASSIFIER = 'classifier'


@dataclass
class Model:
    code: str
    model: Union[sklearn.base.RegressorMixin, sklearn.base.ClassifierMixin]
    type: ModelType
    fast: bool


class Models:
    def __init__(self):
        self.LR_C = Model('LR', LogisticRegression(), ModelType.CLASSIFIER, True)
        self.LDA = Model('LDA', LinearDiscriminantAnalysis(), ModelType.CLASSIFIER, True)
        self.KNN = Model('KNN', KNeighborsClassifier(), ModelType.CLASSIFIER, True)
        self.CART_C = Model('CART', DecisionTreeClassifier(), ModelType.CLASSIFIER, True)
        self.NB = Model('NB', GaussianNB(), ModelType.CLASSIFIER, True)
        self.SVM = Model('SVM', SVC(), ModelType.CLASSIFIER, True)
        self.RF_C = Model('RF', RandomForestClassifier(), ModelType.REGRESSOR, False)
        self.LR_R = Model('LR', LinearRegression(), ModelType.REGRESSOR, True)
        self.LASSO = Model('LASSO', LassoCV(), ModelType.REGRESSOR, True)
        self.RIDGE = Model('RIDGE', RidgeCV(), ModelType.REGRESSOR, True)
        self.ELASTIC = Model('ELASTIC', ElasticNetCV(), ModelType.REGRESSOR, True)
        self.CART_R = Model('CART', DecisionTreeRegressor(), ModelType.REGRESSOR, True)
        self.RF_R = Model('RF', RandomForestRegressor(), ModelType.REGRESSOR, False)

    def fast_classifiers(self) -> Dict:
        return {v.code: v.model for k, v in self.__dict__.items() if
                v.type == ModelType.CLASSIFIER and v.fast is True}

    def all_classifiers(self) -> Dict:
        return {v.code: v.model for k, v in self.__dict__.items() if v.type == ModelType.CLASSIFIER}

    def fast_regressors(self) -> Dict:
        return {v.code: v.model for k, v in self.__dict__.items() if v.type == ModelType.REGRESSOR and v.fast is True}

    def all_regressors(self) -> Dict:
        return {v.code: v.model for k, v in self.__dict__.items() if v.type == ModelType.REGRESSOR}
