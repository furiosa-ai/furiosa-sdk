"""Provide high-level Python APIs to access Furiosa AI's NPUs and its eco-system"""
__all__ = ["model", "session", "tensor", "errors"]

import logging
import pkgutil

from . import session, model, tensor, errors
from ._api import LIBNUX

logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_sdk_git_version():
    """Returns the git commit hash representing the current version of the application."""
    git_version = None
    try:
        git_version = str(pkgutil.get_data('furiosa_sdk_runtime', 'git_version'), encoding="UTF-8")
    except Exception as err:  # pylint: disable=broad-except
        LOG.debug(err)

    return git_version


__version__ = get_sdk_git_version()


def full_version() -> str:
    """Returns a full version string including the native library version"""
    return "Furiosa SDK Runtime {} (libnux {} {} {})" \
        .format(__version__,
                LIBNUX.version().decode('utf-8'),
                LIBNUX.git_short_hash().decode('utf-8'),
                LIBNUX.build_timestamp().decode('utf-8'))
