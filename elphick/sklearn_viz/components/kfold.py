from sklearn.model_selection import StratifiedKFold


class FeatureStratifiedKFold:
    """Strategy to split data into folds based on a feature column.

    The standard StratifiedKFold class from scikit-learn supports stratification based on a feature column.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.skfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")
        return self.skfold.split(X, groups)

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
