import inspect
import os
from pathlib import Path

from docutils.core import publish_file


def script_path() -> Path:
    """Return the path of the script being run, whether in a terminal or in a notebook."""
    return Path(os.path.abspath(inspect.stack()[1].filename))


def convert_rst_to_html(input_file: Path, output_file: Path):
    with open(input_file, 'r') as rst_file, open(output_file, 'w') as html_file:
        publish_file(
            source=rst_file,
            destination=html_file,
            writer_name='html'
        )
