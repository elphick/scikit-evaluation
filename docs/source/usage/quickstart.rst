Quick Start Guide
=================

Plots can be generated with either a convenience function or via an object.


..  code-block:: python

    from elphick.scikit_evaluation.features import plot_feature_importance, FeatureImportance

Using a function is simple:

..  code-block:: python

    fig = plot_feature_importance(pipe)
    fig

Using an object may be your preference, with additional flexibility.

..  code-block:: python

    fi: FeatureImportance = FeatureImportance(pipe)
    fig = fi.plot()
    fig

    # get the feature DataFrame
    df: pd.DataFrame = fi.data

For examples that demonstrate a range of use cases, see the :doc:`/auto_examples/index`.
