import datetime
from enum import Enum
from typing import Dict, List, Optional

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
    """Represent the artifacts and metadata of a neural network model

    Attributes:
        name: a name of this model
        format: the binary format type of model source; e.g., ONNX, tflite
        source: a source binary in ONNX or tflite. It can be used for compiling this model
            with a custom compiler configuration.
        dfg: an intermediate representation of furiosa-compiler. Native post processor implementation uses dfg binary.
            Users don't need to use `dfg` directly.
        enf: the executable binary for furiosa runtime and NPU
        version: model version
        inputs: data type and shape of input tensors
        outputs: data type and shape of output tensors
        compiler_config: a pre-defined compiler option
    """

    # class Config(BaseConfig):
    #     # Non pydantic attribute allowed
    #     # https://pydantic-docs.helpmanual.io/usage/types/#arbitrary-types-allowed
    #     arbitrary_types_allowed = True

    name: str
    source: bytes = Field(repr=False)
    format: Format
    dfg: Optional[bytes] = Field(repr=False)
    enf: Optional[bytes] = Field(repr=False)

    family: Optional[str] = None
    version: Optional[str] = None

    metadata: Optional[Metadata] = None

    inputs: Optional[List[ModelTensor]] = []
    outputs: Optional[List[ModelTensor]] = []

    compiler_config: Optional[Dict] = None
