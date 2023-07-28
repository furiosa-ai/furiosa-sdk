import warnings

try:
    import package_extras
except ModuleNotFoundError:
    warnings.warn(
        "'furiosa.runtime.envs' module is deprecated and will be removed in a future release.",
        category=FutureWarning,
    )

# Module 'envs' is same for two implementations
from furiosa.runtime.legacy.envs import *
