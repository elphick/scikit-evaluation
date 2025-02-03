import pytest
from elphick.sklearn_viz.model_selection.model_selection import ModelSelection
from .fixtures import model_selection, dataset, algorithm


def test_instantiation(model_selection):
    # Instantiate the class
    mdl_sel: ModelSelection = model_selection
    assert isinstance(mdl_sel, ModelSelection)
    print(mdl_sel.results)