
# flake8: noqa

# Import all APIs into this package.
# If you have many APIs here with many many models used in each API this may
# raise a `RecursionError`.
# In order to avoid this, import only the API that you directly need like:
#
#   from .api.account_v1_api import AccountV1Api
#
# or import this package, but before doing it, use:
#
#   import sys
#   sys.setrecursionlimit(n)

# Import APIs into API package:
from furiosa.common.openapi.api.account_v1_api import AccountV1Api
from furiosa.common.openapi.api.compiler_v1_api import CompilerV1Api
from furiosa.common.openapi.api.dss_api import DssApi
from furiosa.common.openapi.api.perfeye_api import PerfeyeApi
from furiosa.common.openapi.api.version_api import VersionApi
