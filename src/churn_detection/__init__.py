"""Init file for churn_detection"""

import sys

if sys.version_info[:2] >= (3, 11):
    from importlib.metadata import PackageNotFoundError, version
else:
    from importlib_metadata import PackageNotFoundError, version

try:
    DIST_NAME = __name__
    __version__ = version(DIST_NAME)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
