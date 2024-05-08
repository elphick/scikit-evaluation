import pandas as pd
import pytest
import numpy as np

from elphick.sklearn_viz.components.kfold import FeatureStratifiedKFold


def test_feature_stratified_kfold():
    # Initialize the FeatureStratifiedKFold class
    fs_kfold = FeatureStratifiedKFold(n_splits=5)

    # Create dummy data
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4])

    # Test if ValueError is raised when groups is None
    with pytest.raises(ValueError):
        list(fs_kfold.split(X, y, groups=None))


def test_feature_stratified_kfold_with_dataframe():
    # Initialize the FeatureStratifiedKFold class
    fs_kfold = FeatureStratifiedKFold(n_splits=5)

    # Create dummy data as DataFrame
    X = pd.DataFrame(np.array([[1, 2], [3, 4], [1, 2], [3, 4]]), columns=['feature1', 'feature2'])
    y = pd.Series(np.array([1, 2, 3, 4]))

    # Test if ValueError is raised when groups is None
    with pytest.raises(ValueError):
        list(fs_kfold.split(X, y, groups=None))


def test_feature_stratified_kfold_with_categorical_group():
    # Initialize the FeatureStratifiedKFold class with shuffle=True
    fs_kfold = FeatureStratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Create dummy data as DataFrame
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 3, 4, 5, 6, 7],
        'group': ['A', 'A', 'B', 'B', 'C', 'C']
    })
    y = pd.Series([1, 2, 1, 2, 1, 2])

    # Split the data using the 'group' column as the groups
    splits = list(fs_kfold.split(X, y, groups=X['group']))

    # Check the number of splits
    assert len(splits) == 2

    # Check the indices of the splits
    assert len(splits[0][0]) == 3  # The training set of the first split should have 3 samples
    assert len(splits[0][1]) == 3  # The test set of the first split should have 3 samples
    assert len(splits[1][0]) == 3  # The training set of the second split should have 3 samples
    assert len(splits[1][1]) == 3  # The test set of the second split should have 3 samples

    # Check that each split has one of each group
    for train_index, test_index in splits:
        train_groups = X.loc[train_index, 'group'].unique()
        test_groups = X.loc[test_index, 'group'].unique()
        assert set(train_groups) == {'A', 'B', 'C'}
        assert set(test_groups) == {'A', 'B', 'C'}
