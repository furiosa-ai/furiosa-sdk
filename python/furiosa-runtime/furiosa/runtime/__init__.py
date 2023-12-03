"""Provide high-level Python APIs to access Furiosa AI's NPUs and its eco-system"""

import warnings

from furiosa.common.utils import get_sdk_version

from . import _utils

try:
    import package_extras
except ModuleNotFoundError:
    try:
        from furiosa.native_runtime import *
    except ImportError as e:
        raise e from None

    def full_version() -> str:
        """Returns a full version from furiosa-rt version"""
        import furiosa.native_runtime as rt

        return "Furiosa SDK Runtime {} (furiosa-rt {} {} {})".format(
            __version__,
            rt.__version__,
            rt.__git_short_hash__,
            rt.__build_timestamp__,
        )

    is_legacy = False
else:
    warnings.warn(
        "'furiosa-runtime[legacy]' is deprecated and will be removed in a future release.",
        category=FutureWarning,
    )

    def full_version() -> str:
        """Returns a full version string including the Nux version"""

        from .legacy._api import LIBNUX as rt

        return "Furiosa SDK Runtime {} (libnux {} {} {})".format(
            __version__,
            rt.version().decode("utf-8"),
            rt.git_short_hash().decode("utf-8"),
            rt.build_timestamp().decode("utf-8"),
        )

    is_legacy = True

    del package_extras


__version__ = get_sdk_version(__name__)
__full_version__ = full_version()

del warnings, get_sdk_version, full_version


__all__ = ["__version__", "__full_version__", "is_legacy"]
