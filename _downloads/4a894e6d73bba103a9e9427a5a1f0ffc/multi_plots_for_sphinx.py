"""
Multi-Plots for Sphinx
======================

Using plotly in Sphinx is great to deliver interactivity in you documentation.
However, if you're using Sphinx to document your datascience project you'll likely come across the challenge of
wanting to execute plots within a loop. This snippet shows how to use the MultiPlot class to save multiple plots and
present them like subplots in your documentation.

"""

# this next line provides the thumbnail, but you must have set save_as_png=True in the MultiPlot object.
# sphinx_gallery_thumbnail_path = '_static/multi_plots_for_sphinx/figures.tagged.0.png'

# %%

from pathlib import Path

import plotly.graph_objects as go

from elphick.sklearn_viz.utils.file import script_path
from elphick.sklearn_viz.utils.plotly import MultiPlot

# %%
# Create some data and figures
# ----------------------------
# We use a loop to generate the figures but instead of displaying them inside the loop,
# we add them to a list.

ys = [
    [2, 3, 1],
    [1, 2, 2],
    [4, 2, 3],
    [3, 2, 5]
]

# Define the directory where the .rst and .html files will be created
static_dir: Path = script_path().parents[1] / 'docs' / 'source' / '_static'

figs: list[go.Figure] = []
for i, y in enumerate(ys):
    title = '-'.join([str(i) for i in y])
    fig = go.Figure(data=[go.Bar(y=y)], layout_title_text=title)
    figs.append(fig)

# %%
# Save the figures to the _static directory of the docs using MultiPlot

MultiPlot(docs_static_dir=static_dir).save_plots(figs)

# %%
# This rst directive will include the figures in the documentation. Note how a subdirectory is created in the
# static folder using the stem of the script filename::
#
#   .. include:: ../_static/multi_plots_for_sphinx/figures.rst
#
# .. include:: ../_static/multi_plots_for_sphinx/figures.rst

# %%
# Multiple Instances
# ------------------
# If you need to use the MultiPlot class in multiple instances, you can use the tag parameter to ensure
# the saved files are unique. This is useful when you want to save multiple sets of plots in the same script.

MultiPlot(docs_static_dir=static_dir, super_title="Second Instance using a tag",
          col_wrap=2, save_as_png=True, tag="tagged").save_plots(figs)

# %%
# Here is how you include the images when using the tag parameter::
#
#   .. include:: ../_static/multi_plots_for_sphinx/figures.tagged.rst
#
# .. include:: ../_static/multi_plots_for_sphinx/figures.tagged.rst
