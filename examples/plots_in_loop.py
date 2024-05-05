"""
Loop Plotting for Sphinx
========================

This script is used to test the plotting of multiple figures in a loop.  The purpose is to test the rendering of the
plots in Sphinx.

"""
from IPython.core.display_functions import display
from IPython.display import HTML
import plotly.io as pio
import plotly.graph_objects as go

ys = [
    [2, 3, 1],
    [1, 2, 2],
    [4, 2, 3],
    [3, 2, 5]
]

for y in ys:
    title = '-'.join([str(i) for i in y])
    fig = go.Figure(
        data=[go.Bar(y=y)],
        layout_title_text=title
    )
    # Convert the plot to an HTML string
    html_string = pio.to_html(fig, full_html=False)
    # Display the HTML string
    HTML(html_string)

    # %%

    print('plot')

# Add a dummy line that produces output
print("End of loop")
