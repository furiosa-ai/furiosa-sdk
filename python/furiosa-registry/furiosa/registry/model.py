import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseConfig, BaseModel, Extra, Field


class Config(BaseConfig):
    # Extra fields not permitted
    extra: Extra = Extra.forbid


class Format(str, Enum):
    """Model binary format to represent the binary specified."""

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


class Metadata(BaseModel):
    """Model metadata to understand a model."""

    __config__ = Config

    description: Optional[str] = None
    publication: Optional[Publication] = None


class Tags(BaseModel):
    class Config:
        extra = Extra.allow

    content_type: Optional[str] = None


class ModelTensor(BaseModel):
    name: str
    datatype: str
    shape: List[int]
    tags: Optional[Tags] = None


class Model(BaseModel):
    """Model for a Furiosa SDK."""

    # class Config(BaseConfig):
    #     # Non pydantic attribute allowed
    #     # https://pydantic-docs.helpmanual.io/usage/types/#arbitrary-types-allowed
    #     arbitrary_types_allowed = True

    name: str
    model: bytes = Field(repr=False)
    format: Format

    family: Optional[str] = None
    version: Optional[str] = None

    metadata: Optional[Metadata] = None

    inputs: Optional[List[ModelTensor]] = []
    outputs: Optional[List[ModelTensor]] = []
