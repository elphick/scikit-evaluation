import numpy as np
import pytest
import pandas as pd
from sklearn.base import is_classifier
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from elphick.sklearn_viz.components.estimators import PartitionClassifier


@pytest.fixture
def data():
    x, y = fetch_openml(data_id=1590, as_frame=True, return_X_y=True)  # Adult dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


@pytest.fixture
def preprocessor(data):
    x_train, _, _, _ = data
    numerical_cols = x_train.select_dtypes(include=[np.number]).columns.to_list()
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
def partition_criteria(data, preprocessor):
    x_train, _, _, _ = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    x_train_transformed = pp.fit_transform(x_train)
    age_scaled = x_train_transformed['age']

    young_threshold = age_scaled.quantile(0.33)
    middle_aged_threshold = age_scaled.quantile(0.66)

    return {
        'young': f'age < {young_threshold}',
        'middle_aged': f'(age >= {young_threshold}) and (age < {middle_aged_threshold})',
        'senior': f'age >= {middle_aged_threshold}'
    }


def test_partition_classifier_fit(data, preprocessor, partition_criteria):
    x_train, _, y_train, _ = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    partition_mdl = make_pipeline(pp, PartitionClassifier(LogisticRegression(), partition_defs=partition_criteria))
    partition_mdl.fit(X=x_train, y=y_train)
    check_is_fitted(partition_mdl[-1])


def test_partition_classifier_predict(data, preprocessor, partition_criteria):
    x_train, x_test, y_train, _ = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    partition_mdl = make_pipeline(pp, PartitionClassifier(LogisticRegression(), partition_defs=partition_criteria))
    partition_mdl.fit(X=x_train, y=y_train)
    predictions = partition_mdl.predict(X=x_test)
    assert len(predictions) == len(x_test)
    assert predictions.index.equals(x_test.index)


def test_partition_classifier_partitioning(data, preprocessor, partition_criteria):
    x_train, _, y_train, _ = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    partition_mdl = make_pipeline(pp, PartitionClassifier(LogisticRegression(), partition_defs=partition_criteria))
    partition_mdl.fit(X=x_train, y=y_train)
    domains = partition_mdl[-1].domains_
    assert set(domains.unique()) == {'young', 'middle_aged', 'senior'}


def test_partition_classifier_is_classifier(partition_criteria):
    partition_classifier = PartitionClassifier(LogisticRegression(), partition_defs=partition_criteria)
    assert is_classifier(partition_classifier)


def test_partition_classifier_identical_results(data, preprocessor, partition_criteria):
    x_train, x_test, y_train, y_test = data
    pp = make_pipeline(preprocessor).set_output(transform='pandas')
    partition_mdl = make_pipeline(pp, PartitionClassifier(LogisticRegression(), partition_defs=partition_criteria))
    partition_mdl.fit(X=x_train, y=y_train)
    partition_predictions: pd.DataFrame = partition_mdl.predict(X=x_test)

    # Separate models for each partition
    x_train_transformed = pp.fit_transform(x_train)
    x_test_transformed = pp.transform(x_test)

    young_model = LogisticRegression()
    middle_aged_model = LogisticRegression()
    senior_model = LogisticRegression()

    young_train_idx = x_train_transformed.query(partition_criteria['young']).index
    middle_aged_train_idx = x_train_transformed.query(partition_criteria['middle_aged']).index
    senior_train_idx = x_train_transformed.query(partition_criteria['senior']).index

    young_model.fit(x_train_transformed.loc[young_train_idx], y_train.loc[young_train_idx])
    middle_aged_model.fit(x_train_transformed.loc[middle_aged_train_idx], y_train.loc[middle_aged_train_idx])
    senior_model.fit(x_train_transformed.loc[senior_train_idx], y_train.loc[senior_train_idx])

    young_test = x_test_transformed.query(partition_criteria['young'])
    middle_aged_test = x_test_transformed.query(partition_criteria['middle_aged'])
    senior_test = x_test_transformed.query(partition_criteria['senior'])

    young_predictions = pd.DataFrame(young_model.predict(x_test_transformed.loc[young_test.index]),
                                     columns=[y_test.name], index=young_test.index)
    middle_aged_predictions = pd.DataFrame(middle_aged_model.predict(x_test_transformed.loc[middle_aged_test.index]),
                                           columns=[y_test.name], index=middle_aged_test.index)
    senior_predictions = pd.DataFrame(senior_model.predict(x_test_transformed.loc[senior_test.index]),
                                      columns=[y_test.name], index=senior_test.index)

    combined_predictions = pd.concat([young_predictions, middle_aged_predictions, senior_predictions], axis=0)

    pd.testing.assert_frame_equal(partition_predictions.sort_index(), combined_predictions.sort_index())
