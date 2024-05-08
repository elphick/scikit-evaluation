from importlib import metadata

try:
    __version__ = metadata.version('sklearn-viz')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass
