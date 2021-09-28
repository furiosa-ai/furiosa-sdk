from typing import List, Optional

from pydantic import BaseModel, Extra


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
    # FIXME(yan): This 'model' field should be trucated as it has very long contents.
    # For next pydantic release, we will bypass via "model: bytes = Field(repr=False)"
    #
    # See https://github.com/samuelcolvin/pydantic/discussions/2756
    model: bytes
    version: Optional[str] = None
    description: Optional[str] = None

    inputs: Optional[List[MetadataTensor]] = []
    outputs: Optional[List[MetadataTensor]] = []
