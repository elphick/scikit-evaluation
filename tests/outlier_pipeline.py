import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from elphick.sklearn_viz.components.pipeline import OutlierPipeline
from sklearn.linear_model import LinearRegression


def test_pandas_pipeline():
    # Initialize the PandasPipeline class with a LinearRegression model
    pipeline = OutlierPipeline(steps=[('lr', LinearRegression())])

    # Create dummy data
    X = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=['feature1', 'feature2'])
    y = pd.DataFrame(np.array([1, 2, 3]), columns=['target'])

    # Fit the pipeline
    pipeline.fit(X, y)

    # Predict using the pipeline
    y_pred = pipeline.predict(X)

    # Check if the predict method returns a DataFrame
    assert isinstance(y_pred, pd.DataFrame)

    # Check if the predict method returns the expected number of predictions
    assert len(y_pred) == len(y)

    # Check if the predict method returns the expected index
    assert y_pred.index.equals(X.index)


def test_pandas_pipeline_from_pipeline():
    # Initialize a base Pipeline with a LinearRegression model
    base_pipeline = Pipeline(steps=[('lr', LinearRegression())])

    # Instantiate the PandasPipeline from the base pipeline
    pipeline = OutlierPipeline.from_pipeline(base_pipeline)

    # Create dummy data
    X = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6]]), columns=['feature1', 'feature2'])
    y = pd.DataFrame(np.array([1, 2, 3]), columns=['target'])

    # Fit the pipeline
    pipeline.fit(X, y)

    # Predict using the pipeline
    y_pred = pipeline.predict(X)

    # Check if the predict method returns a DataFrame
    assert isinstance(y_pred, pd.DataFrame)


def test_pandas_pipeline_predict_with_outliers():
    np.random.seed(42)

    # Initialize the PandasPipeline class with a LinearRegression model
    pipeline = OutlierPipeline(steps=[('lr', LinearRegression())])

    # Define the number of samples and features
    num_samples = 100
    num_features = 2

    # Define the mean vector and covariance matrix for the features
    mean_features = np.zeros(num_features)
    cov_features = np.array([[1, 0.5], [0.5, 1]])  # covariance matrix with a moderate correlation

    # Generate the synthetic features
    X = np.random.multivariate_normal(mean_features, cov_features, num_samples)

    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=[f'feature{i + 1}' for i in range(num_features)])

    # Define the mean and standard deviation for the target
    mean_target = 0
    std_target = 1

    # Generate the synthetic target, with a relationship to X
    y = np.dot(X, [0.3, -0.2]) + np.random.normal(mean_target, std_target, num_samples)

    # Convert to DataFrame
    y_df = pd.DataFrame(y, columns=['target'])

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

    # Modify some X_test records to create outliers
    X_test.iloc[-10:-5, :] = 2.5
    X_test.iloc[-5:, :] = 5

    pipeline.fit(X_train, y_train)

    # Test 1 - 3-state outlier detection
    # Predict using the pipeline with the outlier metric
    y_pred = pipeline.predict(X_test, with_outlier_metrics=True, return_all_outlier_metrics=True)

    # Check if the predict method returns the expected columns
    for col in ['outlier_mahal', 'outlier_pval', 'outlier']:
        assert col in y_pred.columns

    # Check if the predict method returns the expected index
    assert y_pred.index.equals(X_test.index)

    # Check if the last 5 records are classified as outliers...
    assert (y_pred['outlier'].iloc[:-10] == 'False').all()
    assert (y_pred['outlier'].iloc[-10:-5] == 'Possible').all()
    assert (y_pred['outlier'].iloc[-5:] == 'True').all()

    # Test 2 - 2-state outlier detection
    # Predict using the pipeline with the outlier metric
    y_pred = pipeline.predict(X_test, with_outlier_metrics=True, return_all_outlier_metrics=True,
                              possible_outlier_threshold=None)

    # Check if the last 5 records are classified as outliers...
    assert (y_pred['outlier'].iloc[:-5] == 'False').all()
    assert (y_pred['outlier'].iloc[-5:] == 'True').all()