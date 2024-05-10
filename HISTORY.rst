Elphick.Sklearn_Viz 0.6.0 (2024-05-10)
======================================

Features
--------

- Implemented PartitionClassifier, complementing the existing PartitionRegressor.

  The previous references to `algo` (algorithm) have been replaced with `estimators` in the API.
  This is to reflect the fact that the estimators can be either classifiers, regressors or pipelines. (#43)


Bugfixes
--------

- Bugfix for ModelSelection that now supports datasets having different columns. (#36)


Elphick.Sklearn_Viz 0.5.0 (2024-05-08)
======================================

Features
--------

- Added MultiPlot to enable plotly generation in loops, while still displaying all plots in the docs. (#41)


Improved Documentation
----------------------

- Added a changelog. (#40)
