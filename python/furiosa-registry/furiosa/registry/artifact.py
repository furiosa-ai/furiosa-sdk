from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel


class Format(str, Enum):
    """
    Model binary format to represent the binary specified.
    """

    Code = "code"
    ONNX = "onnx"
    TFLite = "tflite"


class RuntimeConfig(BaseModel):
    """
    Runtime configuration for FuriosaAI system.
    """

    npu_device: Optional[str] = None
    compiler_config: Optional[Dict] = None


class ModelMetadata(BaseModel):
    """
    Model metadata to understand a model.
    """

    arxiv: Optional[str] = None
    year: Optional[int] = None
    month: Optional[int] = None


class Artifact(BaseModel):
    """
    Data including model binary, metadata, and configurations to run a single model.
    """

    name: str
    family: str
    location: str
    format: Format
    description: Optional[str] = None

    config: Optional[RuntimeConfig] = None
    metadata: Optional[ModelMetadata] = None
