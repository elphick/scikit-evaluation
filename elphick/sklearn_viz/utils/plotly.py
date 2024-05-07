import inspect
from pathlib import Path
from typing import Optional

import plotly.offline as pyo


class MultiPlot:
    def __init__(self, docs_static_dir: Path, tag: Optional[str] = None, super_title: Optional[str] = None,
                 col_wrap: int = 1, save_as_png: bool = False):
        self.tag: Optional[str] = tag
        self.super_title: Optional[str] = super_title
        self.col_wrap: int = col_wrap
        self.save_as_png: bool = save_as_png

        calling_file = inspect.stack()[1].filename

        # Create a subdirectory in docs_static_dir with the stem of the script filename
        self.output_dir: Path = docs_static_dir / Path(calling_file).stem
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create the .rst file path
        self.rst_file_path = self.output_dir / f"figures.rst"
        if self.tag:
            self.rst_file_path = self.output_dir / f"figures.{self.tag}.rst"

    def save_plots(self, figs, height: int = 600, save_as_png=False):
        for i, fig in enumerate(figs):
            suffix: str = f".{i}.html"

            # Save the figure as an HTML file
            file_path = self.output_dir / self.rst_file_path.with_suffix(suffix).name
            pyo.plot(fig, filename=str(file_path), image_height=height, auto_open=False)

            # Open the .rst file, write the iframe, and close the file
            file_mode = 'w' if i == 0 else 'a'
            with open(self.rst_file_path, file_mode) as rst_file:
                if i == 0:
                    if self.super_title is not None:
                        rst_file.write(f".. raw:: html\n\n    <h1>{self.super_title}</h1>\n\n")
                    rst_file.write(".. raw:: html\n\n    <table style='width: 100%;'>\n")
                if i % self.col_wrap == 0:
                    rst_file.write("        <tr>\n")
                rst_file.write(
                    f"            <td><iframe src='../_static/{file_path.parent.name}/{file_path.name}' width='100%' height='{height}px' style='border:1px solid lightgray;'></iframe></td>\n")
                if (i + 1) % self.col_wrap == 0 or i == len(figs) - 1:
                    rst_file.write("        </tr>\n")
                if i == len(figs) - 1:
                    rst_file.write("    </table>\n")

            # Save the figure as a PNG file if save_as_png is True
            if save_as_png:
                fig.write_image(file_path.with_suffix('.png'))
