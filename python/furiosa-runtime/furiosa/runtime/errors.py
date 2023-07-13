import warnings

try:
    import package_extras
except ModuleNotFoundError:
    from furiosa.native_runtime.errors import *

    warnings.warn(
        "'furiosa.runtime.errors' module is deprecated and will be removed in a future release.",
        category=FutureWarning,
    )
else:
    from furiosa.runtime.legacy.errors import *
