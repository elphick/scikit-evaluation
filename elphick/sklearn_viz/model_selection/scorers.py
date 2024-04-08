import numpy as np

from elphick.sklearn_viz.model_selection.metrics import r2_with_nan

classification_scorers = {'acc': 'accuracy',
                          'prec_macro': 'precision_macro',
                          'rec_macro': 'recall_macro'}


def r2_score_with_nan(estimator, X, y):
    return r2_with_nan(y_true=y, y_est=estimator.predict(X))


regression_scorers = {
    # 'r2_nan': r2_score_with_nan,
    'r2_score': 'r2',
    'mae': 'neg_mean_absolute_error',
    'mse': 'neg_mean_squared_error'}
