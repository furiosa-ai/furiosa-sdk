import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseConfig, BaseModel, Extra


class Config(BaseConfig):
    # Extra fields not permitted
    extra: Extra = Extra.forbid


class Format(str, Enum):
    """Model binary format to represent the binary specified."""

    Code = "code"
    ONNX = "onnx"
    TFLite = "tflite"


class Publication(BaseModel):
    """Model publication information."""

    __config__ = Config

    authors: Optional[List[str]] = None
    title: Optional[str] = None
    publisher: Optional[str] = None
    date: Optional[datetime.date] = None
    url: Optional[str] = None


class ModelMetadata(BaseModel):
    """Model metadata to understand a model."""

    __config__ = Config

    description: Optional[str] = None
    publication: Optional[Publication] = None


class Artifact(BaseModel):
    """Data including model binary, metadata, and configurations to run a single model."""

    __config__ = Config

    name: str
    version: Optional[str] = None
    family: str
    location: str
    format: Format
    doc: Optional[str] = None

    metadata: Optional[ModelMetadata] = None
