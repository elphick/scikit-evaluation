Sklearn_Viz 0.6.5 (2024-05-12)
==============================

Features
--------

- Improvements to the LearningCurve.  Added dataclass for improved data management. Refactored plot method to support plotting of goodness of fit (gof) metrics other than the scorer. (#37)


Sklearn_Viz 0.6.6 (2024-05-12)
==============================

Features
--------

- Parallel processing for the LearningCurve when using metrics. (#37)


Sklearn_Viz 0.6.5 (2024-05-12)
==============================

Features
--------

- Improvements to the LearningCurve.  Added dataclass for improved data management. Refactored plot method to support plotting of goodness of fit (gof) metrics other than the scorer. (#37)


Sklearn_Viz 0.6.4 (2024-05-11)
==============================

Features
--------

- The Errors object plot method has been improved.  It is now square as it should be for better interpretation of residuals, with the y=x line at 45 degrees.  Still a glitch with the example top marginal y-axis limits.  Example in the gallery has been updated. (#27)


Sklearn_Viz 0.6.3 (2024-05-11)
==============================

Features
--------

- ModelSelection - added verbosity for more granular logging. Updated the gallery example. (#47)


Sklearn_Viz 0.6.2 (2024-05-11)
==============================

Features
--------

- Added multiproccessor support to ModelSelection.  Updated the example in the gallery. (#47)


Sklearn_Viz 0.6.1 (2024-05-11)
==============================

Bugfixes
--------

- Bugfix for ModelSelection default scorer bug. (#46)


Sklearn_Viz 0.6.0 (2024-05-10)
==============================

Features
--------

- Implemented PartitionClassifier, complementing the existing PartitionRegressor.

  The previous references to `algo` (algorithm) have been replaced with `estimators` in the API.
  This is to reflect the fact that the estimators can be either classifiers, regressors or pipelines. (#43)


Bugfixes
--------

- Bugfix for ModelSelection that now supports datasets having different columns. (#36)


Sklearn_Viz 0.5.0 (2024-05-08)
==============================

Features
--------

- Added MultiPlot to enable plotly generation in loops, while still displaying all plots in the docs. (#41)


Improved Documentation
----------------------

- Added a changelog. (#40)
