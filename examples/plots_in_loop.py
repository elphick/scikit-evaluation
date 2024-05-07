"""
Loop Plotting for Sphinx
========================

This script is used to test the plotting of multiple figures in a loop.  The purpose is to test the rendering of the
plots in Sphinx.

"""

# sphinx_gallery_thumbnail_path = '_static/plots_in_loop/figures.tagged.0.png'

from pathlib import Path

import plotly.graph_objects as go

from elphick.sklearn_viz.utils.file import script_path
from elphick.sklearn_viz.utils.plotly import MultiPlot

# get the script filepath whether it is run in a terminal or in a notebook
__file__: str = script_path()

ys = [
    [2, 3, 1],
    [1, 2, 2],
    [4, 2, 3],
    [3, 2, 5]
]

# Define the directory where the .rst and .html files will be created
static_dir = Path(__file__).parents[1] / 'docs' / 'source' / '_static'

plot_helper = MultiPlot(docs_static_dir=static_dir, super_title="Plots in Loop")

figs = []
for i, y in enumerate(ys):
    title = '-'.join([str(i) for i in y])
    fig = go.Figure(data=[go.Bar(y=y)], layout_title_text=title)
    figs.append(fig)

plot_helper.save_plots(figs)

# %%
#
# .. include:: ../_static/plots_in_loop/figures.rst

# %%
# Save the figures with a tag

plot_helper_2 = MultiPlot(docs_static_dir=static_dir, super_title="Second Instance using a tag",
                          col_wrap=2, save_as_png=True, tag="tagged")
plot_helper_2.save_plots(figs)

# %%
#
# .. include:: ../_static/plots_in_loop/figures.tagged.rst
