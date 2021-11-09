"""Provide high-level Python APIs to access Furiosa AI's NPUs and its eco-system"""

from furiosa.common.utils import get_sdk_version

from ._api import LIBNUX, runtime_version


def full_version() -> str:
    """Returns a full version string including the native library version"""
    return "Furiosa SDK Runtime {} (libnux {} {} {})".format(
        __version__,
        LIBNUX.version().decode('utf-8'),
        LIBNUX.git_short_hash().decode('utf-8'),
        LIBNUX.build_timestamp().decode('utf-8'),
    )


__version__ = get_sdk_version(__name__)
__full_version__ = full_version()
