"""Furiosa serving framework, easy to use inference server"""

from .application import ServeAPI
from .model import ServeModel

__all__ = ["ServeAPI", "ServeModel"]
