import warnings

try:
    import package_extras
except ModuleNotFoundError:
    from furiosa.native_runtime.model import *

    warnings.warn(
        "'furiosa.runtime.model' module is deprecated and will be removed in a future release.",
        category=FutureWarning,
    )
else:
    from furiosa.runtime.legacy.model import *
