"""Tensor object and its utilities"""

import warnings

try:
    import package_extras
except ModuleNotFoundError:
    from furiosa.native_runtime.tensor import *

    warnings.warn(
        "'furiosa.runtime.tensor' module is deprecated and will be removed in a future release.",
        category=FutureWarning,
    )
else:
    from furiosa.runtime.legacy.tensor import *
