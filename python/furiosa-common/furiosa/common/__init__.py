"""
Furiosa sdk tools
"""
from .utils import get_sdk_version

__all__ = ["get_sdk_version"]

__version__ = get_sdk_version(__name__)
