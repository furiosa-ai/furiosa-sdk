# generated by datamodel-codegen:
#   filename:  predict.yaml
#   timestamp: 2023-07-24T08:46:45+00:00

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel


class MetadataServerResponse(BaseModel):
    name: str
    version: str
    extensions: List[str]


class MetadataServerErrorResponse(BaseModel):
    error: str


class Tags(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    content_type: Optional[str] = None


class MetadataTensor(BaseModel):
    name: str
    datatype: str
    shape: List[int]
    tags: Optional[Tags] = None


class MetadataModelErrorResponse(BaseModel):
    error: str


class Parameters(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    content_type: Optional[str] = None


class TensorData(RootModel):
    root: Any = Field(..., title='tensor_data')

    # FIXME(yan): This was manually added. Replace codegen template later
    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, idx):
        return self.root[idx]

    def __len__(self):
        return len(self.root)


class RequestOutput(BaseModel):
    name: str
    parameters: Optional[Parameters] = None


class ResponseOutput(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    parameters: Optional[Parameters] = None
    data: TensorData


class InferenceResponse(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    id: Optional[str] = None
    parameters: Optional[Parameters] = None
    outputs: List[ResponseOutput]

    # FIXME(mg): This was manually added. Replace codegen template later
    # To use `model_` prefix
    model_config = ConfigDict(protected_namespaces=())


class InferenceErrorResponse(BaseModel):
    error: Optional[str] = None


class MetadataModelResponse(BaseModel):
    name: str
    versions: Optional[List[str]] = None
    platform: str
    inputs: Optional[List[MetadataTensor]] = None
    outputs: Optional[List[MetadataTensor]] = None


class RequestInput(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    parameters: Optional[Parameters] = None
    data: TensorData


class InferenceRequest(BaseModel):
    id: Optional[str] = None
    parameters: Optional[Parameters] = None
    inputs: List[RequestInput]
    outputs: Optional[List[RequestOutput]] = None
