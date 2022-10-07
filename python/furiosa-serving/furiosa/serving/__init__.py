"""Furiosa serving framework, easy to use inference server."""

from .application import ServeAPI
from .model import CPUServeModel, NPUServeModel, OpenVINOServeModel, ServeModel
from .processors import Processor

__all__ = [
    "ServeAPI",
    "Processor",
    "ServeModel",
    "NPUServeModel",
    "CPUServeModel",
    "OpenVINOServeModel",
]
