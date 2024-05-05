from importlib import metadata

try:
    __version__ = metadata.version('mass-composition')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass
