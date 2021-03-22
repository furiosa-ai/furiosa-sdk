"""Provide high-level Python APIs to access Furiosa AI's NPUs and its eco-system"""
__version__ = "0.2.1"
__all__ = ["model", "session", "tensor", "errors"]

import logging

from .errors import NativeError, is_ok, is_err
from . import session, model, tensor, errors
from ._api import LIBNUX

logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def full_version() -> str:
    return "Nuxpy {} (libnux {} {} {})".format(__version__, LIBNUX.version().decode('utf-8'),
                                               LIBNUX.git_short_hash().decode('utf-8'),
                                               LIBNUX.build_timestamp().decode('utf-8'))
