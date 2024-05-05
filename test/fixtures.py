import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from elphick.sklearn_viz.model_selection import ModelSelection


# Define your fixtures here
@pytest.fixture
def algorithm():
    # Replace with your actual algorithms
    return DummyRegressor()


@pytest.fixture
def algorithms():
    # Replace with your actual algorithms
    return {'NULL', DummyRegressor()}


@pytest.fixture
def dataset():
    # Replace with your actual datasets
    return pd.DataFrame(
        {'feature1': [1, 2, 3, 4, 5, 6], 'group': ['A', 'B', 'C', 'A', 'B', 'C'], 'target': [1, 2, 3, 4, 5, 6]})


@pytest.fixture
def model_selection(algorithm, dataset):
    return ModelSelection(algorithm, dataset, target='target', group=dataset['group'], k_folds=3)
