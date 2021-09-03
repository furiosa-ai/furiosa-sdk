from typing import List, Optional

from pydantic import BaseModel, Extra

from .artifact import RuntimeConfig


class Tags(BaseModel):
    class Config:
        extra = Extra.allow

    content_type: Optional[str] = None


class MetadataTensor(BaseModel):
    name: str
    datatype: str
    shape: List[int]
    tags: Optional[Tags] = None


class Model(BaseModel):
    """
    Model for a FuriosaAI system
    """

    name: str
    model: bytes
    version: Optional[str] = None
    description: Optional[str] = None
    config: Optional[RuntimeConfig] = None

    inputs: Optional[List[MetadataTensor]] = []
    outputs: Optional[List[MetadataTensor]] = []
