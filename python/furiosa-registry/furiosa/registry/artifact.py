from enum import Enum
from typing import Optional

from pydantic import BaseConfig, BaseModel


class Config(BaseConfig):
    # Extra fields not permitted
    extra = "forbid"


class Format(str, Enum):
    """
    Model binary format to represent the binary specified.
    """

    Code = "code"
    ONNX = "onnx"
    TFLite = "tflite"


class Publication(BaseModel):
    """
    Model publication information.
    """

    arxiv: Optional[str] = None
    year: Optional[int] = None
    month: Optional[int] = None


class ModelMetadata(BaseModel):
    """
    Model metadata to understand a model.
    """

    description: Optional[str] = None
    publication: Optional[Publication] = None


class Artifact(BaseModel):
    """
    Data including model binary, metadata, and configurations to run a single model.
    """

    __config__ = Config

    name: str
    family: str
    location: str
    format: Format
    doc: Optional[str] = None

    metadata: Optional[ModelMetadata] = None
