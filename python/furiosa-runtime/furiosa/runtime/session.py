import warnings

try:
    import package_extras
except ModuleNotFoundError:
    from furiosa.native_runtime.session import *

    warnings.warn(
        "'furiosa.runtime.session' module is deprecated and will be removed in a future release.",
        category=FutureWarning,
    )
else:
    from furiosa.runtime.legacy.session import *
