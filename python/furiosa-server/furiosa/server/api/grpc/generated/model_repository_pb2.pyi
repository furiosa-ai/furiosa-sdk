"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor = ...

class RepositoryIndexRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    REPOSITORY_NAME_FIELD_NUMBER: builtins.int
    READY_FIELD_NUMBER: builtins.int
    # The name of the repository. If empty the index is returned
    # for all repositories.
    repository_name: typing.Text = ...
    # If true return only models currently ready for inferencing.
    ready: builtins.bool = ...
    def __init__(self,
        *,
        repository_name : typing.Text = ...,
        ready : builtins.bool = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"ready",b"ready",u"repository_name",b"repository_name"]) -> None: ...
global___RepositoryIndexRequest = RepositoryIndexRequest

class RepositoryIndexResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    # Index entry for a model.
    class ModelIndex(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
        NAME_FIELD_NUMBER: builtins.int
        VERSION_FIELD_NUMBER: builtins.int
        STATE_FIELD_NUMBER: builtins.int
        REASON_FIELD_NUMBER: builtins.int
        # The name of the model.
        name: typing.Text = ...
        # The version of the model.
        version: typing.Text = ...
        # The state of the model.
        state: typing.Text = ...
        # The reason, if any, that the model is in the given state.
        reason: typing.Text = ...
        def __init__(self,
            *,
            name : typing.Text = ...,
            version : typing.Text = ...,
            state : typing.Text = ...,
            reason : typing.Text = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal[u"name",b"name",u"reason",b"reason",u"state",b"state",u"version",b"version"]) -> None: ...

    MODELS_FIELD_NUMBER: builtins.int
    # An index entry for each model.
    @property
    def models(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RepositoryIndexResponse.ModelIndex]: ...
    def __init__(self,
        *,
        models : typing.Optional[typing.Iterable[global___RepositoryIndexResponse.ModelIndex]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"models",b"models"]) -> None: ...
global___RepositoryIndexResponse = RepositoryIndexResponse

class RepositoryModelLoadRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    REPOSITORY_NAME_FIELD_NUMBER: builtins.int
    MODEL_NAME_FIELD_NUMBER: builtins.int
    # The name of the repository to load from. If empty the model
    # is loaded from any repository.
    repository_name: typing.Text = ...
    # The name of the model to load, or reload.
    model_name: typing.Text = ...
    def __init__(self,
        *,
        repository_name : typing.Text = ...,
        model_name : typing.Text = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"model_name",b"model_name",u"repository_name",b"repository_name"]) -> None: ...
global___RepositoryModelLoadRequest = RepositoryModelLoadRequest

class RepositoryModelLoadResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    def __init__(self,
        ) -> None: ...
global___RepositoryModelLoadResponse = RepositoryModelLoadResponse

class RepositoryModelUnloadRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    REPOSITORY_NAME_FIELD_NUMBER: builtins.int
    MODEL_NAME_FIELD_NUMBER: builtins.int
    # The name of the repository from which the model was originally
    # loaded. If empty the repository is not considered.
    repository_name: typing.Text = ...
    # The name of the model to unload.
    model_name: typing.Text = ...
    def __init__(self,
        *,
        repository_name : typing.Text = ...,
        model_name : typing.Text = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"model_name",b"model_name",u"repository_name",b"repository_name"]) -> None: ...
global___RepositoryModelUnloadRequest = RepositoryModelUnloadRequest

class RepositoryModelUnloadResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    def __init__(self,
        ) -> None: ...
global___RepositoryModelUnloadResponse = RepositoryModelUnloadResponse