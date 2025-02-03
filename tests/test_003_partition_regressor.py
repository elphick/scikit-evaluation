import numpy as np
import pytest
import pandas as pd
from sklearn.base import is_regressor
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from elphick.sklearn_viz.components.estimators import PartitionRegressor


@pytest.fixture
def data():
    x, y = fetch_california_housing(return_X_y=True, as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


@pytest.fixture
def preprocessor(data):
    x_train, _, _, _ = data
    numerical_cols = x_train.select_dtypes(include=[float]).columns.to_list()
    categorical_cols = x_train.select_dtypes(include=[object, 'category']).columns.to_list()
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer(
        [
            ("one-hot-encoder", categorical_preprocessor, categorical_cols),
            ("standard_scaler", numerical_preprocessor, numerical_cols),
        ], verbose_feature_names_out=False
    )
    return preprocessor


@pytest.fixture
def partition_criteria():
    return {'small': 'AveRooms < -0.43',
            'medium': '(AveRooms >= -0.43) and (AveRooms < 0.24)',
            'large': 'AveRooms >= 0.24'}


def test_partition_regressor_fit(data, preprocessor, partition_criteria):
    x_train, _, y_train, _ = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    partition_mdl = make_pipeline(pp, PartitionRegressor(LinearRegression(), partition_defs=partition_criteria))
    partition_mdl.fit(X=x_train, y=y_train)
    check_is_fitted(partition_mdl[-1])


def test_partition_regressor_predict(data, preprocessor, partition_criteria):
    x_train, x_test, y_train, _ = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    partition_mdl = make_pipeline(pp, PartitionRegressor(LinearRegression(), partition_defs=partition_criteria))
    partition_mdl.fit(X=x_train, y=y_train)
    predictions = partition_mdl.predict(X=x_test)
    assert len(predictions) == len(x_test)
    assert predictions.index.equals(x_test.index)


def test_partition_regressor_partitioning(data, preprocessor, partition_criteria):
    x_train, _, y_train, _ = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    partition_mdl = make_pipeline(pp, PartitionRegressor(LinearRegression(), partition_defs=partition_criteria))
    partition_mdl.fit(X=x_train, y=y_train)
    domains = partition_mdl[-1].domains_
    assert set(domains.unique()) == {'small', 'medium', 'large'}


def test_partition_regressor_is_regressor(partition_criteria):
    partition_regressor = PartitionRegressor(LinearRegression(), partition_defs=partition_criteria)
    assert is_regressor(partition_regressor)


def test_partition_regressor_identical_results(data, preprocessor, partition_criteria):
    x_train, x_test, y_train, y_test = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    partition_mdl = make_pipeline(pp, PartitionRegressor(LinearRegression(), partition_defs=partition_criteria))
    partition_mdl.fit(X=x_train, y=y_train)
    partition_predictions: pd.DataFrame = partition_mdl.predict(X=x_test)

    # Separate models for each partition
    x_train_transformed = pp.fit_transform(x_train)
    x_test_transformed = pp.transform(x_test)

    small_model = LinearRegression()
    medium_model = LinearRegression()
    large_model = LinearRegression()

    small_train_idx = x_train_transformed.query(partition_criteria['small']).index
    medium_train_idx = x_train_transformed.query(partition_criteria['medium']).index
    large_train_idx = x_train_transformed.query(partition_criteria['large']).index

    small_model.fit(x_train_transformed.loc[small_train_idx], y_train.loc[small_train_idx])
    medium_model.fit(x_train_transformed.loc[medium_train_idx], y_train.loc[medium_train_idx])
    large_model.fit(x_train_transformed.loc[large_train_idx], y_train.loc[large_train_idx])

    small_test = x_test_transformed.query(partition_criteria['small'])
    medium_test = x_test_transformed.query(partition_criteria['medium'])
    large_test = x_test_transformed.query(partition_criteria['large'])

    small_predictions = pd.DataFrame(small_model.predict(x_test_transformed.loc[small_test.index]),
                                     columns=[y_test.name], index=small_test.index)
    medium_predictions = pd.DataFrame(medium_model.predict(x_test_transformed.loc[medium_test.index]),
                                      columns=[y_test.name], index=medium_test.index)
    large_predictions = pd.DataFrame(large_model.predict(x_test_transformed.loc[large_test.index]),
                                     columns=[y_test.name], index=large_test.index)

    combined_predictions = pd.concat([small_predictions, medium_predictions, large_predictions], axis=0)

    pd.testing.assert_frame_equal(partition_predictions.sort_index(), combined_predictions.sort_index())
