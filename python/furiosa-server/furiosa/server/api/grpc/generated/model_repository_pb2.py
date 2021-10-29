# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: model_repository.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='model_repository.proto',
    package='inference.model_repository',
    syntax='proto3',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x16model_repository.proto\x12\x1ainference.model_repository\"@\n\x16RepositoryIndexRequest\x12\x17\n\x0frepository_name\x18\x01 \x01(\t\x12\r\n\x05ready\x18\x02 \x01(\x08\"\xb5\x01\n\x17RepositoryIndexResponse\x12N\n\x06models\x18\x01 \x03(\x0b\x32>.inference.model_repository.RepositoryIndexResponse.ModelIndex\x1aJ\n\nModelIndex\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\x12\r\n\x05state\x18\x03 \x01(\t\x12\x0e\n\x06reason\x18\x04 \x01(\t\"I\n\x1aRepositoryModelLoadRequest\x12\x17\n\x0frepository_name\x18\x01 \x01(\t\x12\x12\n\nmodel_name\x18\x02 \x01(\t\"\x1d\n\x1bRepositoryModelLoadResponse\"K\n\x1cRepositoryModelUnloadRequest\x12\x17\n\x0frepository_name\x18\x01 \x01(\t\x12\x12\n\nmodel_name\x18\x02 \x01(\t\"\x1f\n\x1dRepositoryModelUnloadResponse2\xb2\x03\n\x16ModelRepositoryService\x12|\n\x0fRepositoryIndex\x12\x32.inference.model_repository.RepositoryIndexRequest\x1a\x33.inference.model_repository.RepositoryIndexResponse\"\x00\x12\x88\x01\n\x13RepositoryModelLoad\x12\x36.inference.model_repository.RepositoryModelLoadRequest\x1a\x37.inference.model_repository.RepositoryModelLoadResponse\"\x00\x12\x8e\x01\n\x15RepositoryModelUnload\x12\x38.inference.model_repository.RepositoryModelUnloadRequest\x1a\x39.inference.model_repository.RepositoryModelUnloadResponse\"\x00\x62\x06proto3',
)


_REPOSITORYINDEXREQUEST = _descriptor.Descriptor(
    name='RepositoryIndexRequest',
    full_name='inference.model_repository.RepositoryIndexRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='repository_name',
            full_name='inference.model_repository.RepositoryIndexRequest.repository_name',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='ready',
            full_name='inference.model_repository.RepositoryIndexRequest.ready',
            index=1,
            number=2,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=False,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=54,
    serialized_end=118,
)


_REPOSITORYINDEXRESPONSE_MODELINDEX = _descriptor.Descriptor(
    name='ModelIndex',
    full_name='inference.model_repository.RepositoryIndexResponse.ModelIndex',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='name',
            full_name='inference.model_repository.RepositoryIndexResponse.ModelIndex.name',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='version',
            full_name='inference.model_repository.RepositoryIndexResponse.ModelIndex.version',
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='state',
            full_name='inference.model_repository.RepositoryIndexResponse.ModelIndex.state',
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='reason',
            full_name='inference.model_repository.RepositoryIndexResponse.ModelIndex.reason',
            index=3,
            number=4,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=228,
    serialized_end=302,
)

_REPOSITORYINDEXRESPONSE = _descriptor.Descriptor(
    name='RepositoryIndexResponse',
    full_name='inference.model_repository.RepositoryIndexResponse',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='models',
            full_name='inference.model_repository.RepositoryIndexResponse.models',
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[
        _REPOSITORYINDEXRESPONSE_MODELINDEX,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=121,
    serialized_end=302,
)


_REPOSITORYMODELLOADREQUEST = _descriptor.Descriptor(
    name='RepositoryModelLoadRequest',
    full_name='inference.model_repository.RepositoryModelLoadRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='repository_name',
            full_name='inference.model_repository.RepositoryModelLoadRequest.repository_name',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='model_name',
            full_name='inference.model_repository.RepositoryModelLoadRequest.model_name',
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=304,
    serialized_end=377,
)


_REPOSITORYMODELLOADRESPONSE = _descriptor.Descriptor(
    name='RepositoryModelLoadResponse',
    full_name='inference.model_repository.RepositoryModelLoadResponse',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=379,
    serialized_end=408,
)


_REPOSITORYMODELUNLOADREQUEST = _descriptor.Descriptor(
    name='RepositoryModelUnloadRequest',
    full_name='inference.model_repository.RepositoryModelUnloadRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='repository_name',
            full_name='inference.model_repository.RepositoryModelUnloadRequest.repository_name',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='model_name',
            full_name='inference.model_repository.RepositoryModelUnloadRequest.model_name',
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=410,
    serialized_end=485,
)


_REPOSITORYMODELUNLOADRESPONSE = _descriptor.Descriptor(
    name='RepositoryModelUnloadResponse',
    full_name='inference.model_repository.RepositoryModelUnloadResponse',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=487,
    serialized_end=518,
)

_REPOSITORYINDEXRESPONSE_MODELINDEX.containing_type = _REPOSITORYINDEXRESPONSE
_REPOSITORYINDEXRESPONSE.fields_by_name['models'].message_type = _REPOSITORYINDEXRESPONSE_MODELINDEX
DESCRIPTOR.message_types_by_name['RepositoryIndexRequest'] = _REPOSITORYINDEXREQUEST
DESCRIPTOR.message_types_by_name['RepositoryIndexResponse'] = _REPOSITORYINDEXRESPONSE
DESCRIPTOR.message_types_by_name['RepositoryModelLoadRequest'] = _REPOSITORYMODELLOADREQUEST
DESCRIPTOR.message_types_by_name['RepositoryModelLoadResponse'] = _REPOSITORYMODELLOADRESPONSE
DESCRIPTOR.message_types_by_name['RepositoryModelUnloadRequest'] = _REPOSITORYMODELUNLOADREQUEST
DESCRIPTOR.message_types_by_name['RepositoryModelUnloadResponse'] = _REPOSITORYMODELUNLOADRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RepositoryIndexRequest = _reflection.GeneratedProtocolMessageType(
    'RepositoryIndexRequest',
    (_message.Message,),
    {
        'DESCRIPTOR': _REPOSITORYINDEXREQUEST,
        '__module__': 'model_repository_pb2'
        # @@protoc_insertion_point(class_scope:inference.model_repository.RepositoryIndexRequest)
    },
)
_sym_db.RegisterMessage(RepositoryIndexRequest)

RepositoryIndexResponse = _reflection.GeneratedProtocolMessageType(
    'RepositoryIndexResponse',
    (_message.Message,),
    {
        'ModelIndex': _reflection.GeneratedProtocolMessageType(
            'ModelIndex',
            (_message.Message,),
            {
                'DESCRIPTOR': _REPOSITORYINDEXRESPONSE_MODELINDEX,
                '__module__': 'model_repository_pb2'
                # @@protoc_insertion_point(class_scope:inference.model_repository.RepositoryIndexResponse.ModelIndex)
            },
        ),
        'DESCRIPTOR': _REPOSITORYINDEXRESPONSE,
        '__module__': 'model_repository_pb2'
        # @@protoc_insertion_point(class_scope:inference.model_repository.RepositoryIndexResponse)
    },
)
_sym_db.RegisterMessage(RepositoryIndexResponse)
_sym_db.RegisterMessage(RepositoryIndexResponse.ModelIndex)

RepositoryModelLoadRequest = _reflection.GeneratedProtocolMessageType(
    'RepositoryModelLoadRequest',
    (_message.Message,),
    {
        'DESCRIPTOR': _REPOSITORYMODELLOADREQUEST,
        '__module__': 'model_repository_pb2'
        # @@protoc_insertion_point(class_scope:inference.model_repository.RepositoryModelLoadRequest)
    },
)
_sym_db.RegisterMessage(RepositoryModelLoadRequest)

RepositoryModelLoadResponse = _reflection.GeneratedProtocolMessageType(
    'RepositoryModelLoadResponse',
    (_message.Message,),
    {
        'DESCRIPTOR': _REPOSITORYMODELLOADRESPONSE,
        '__module__': 'model_repository_pb2'
        # @@protoc_insertion_point(class_scope:inference.model_repository.RepositoryModelLoadResponse)
    },
)
_sym_db.RegisterMessage(RepositoryModelLoadResponse)

RepositoryModelUnloadRequest = _reflection.GeneratedProtocolMessageType(
    'RepositoryModelUnloadRequest',
    (_message.Message,),
    {
        'DESCRIPTOR': _REPOSITORYMODELUNLOADREQUEST,
        '__module__': 'model_repository_pb2'
        # @@protoc_insertion_point(class_scope:inference.model_repository.RepositoryModelUnloadRequest)
    },
)
_sym_db.RegisterMessage(RepositoryModelUnloadRequest)

RepositoryModelUnloadResponse = _reflection.GeneratedProtocolMessageType(
    'RepositoryModelUnloadResponse',
    (_message.Message,),
    {
        'DESCRIPTOR': _REPOSITORYMODELUNLOADRESPONSE,
        '__module__': 'model_repository_pb2'
        # @@protoc_insertion_point(class_scope:inference.model_repository.RepositoryModelUnloadResponse)
    },
)
_sym_db.RegisterMessage(RepositoryModelUnloadResponse)


_MODELREPOSITORYSERVICE = _descriptor.ServiceDescriptor(
    name='ModelRepositoryService',
    full_name='inference.model_repository.ModelRepositoryService',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=521,
    serialized_end=955,
    methods=[
        _descriptor.MethodDescriptor(
            name='RepositoryIndex',
            full_name='inference.model_repository.ModelRepositoryService.RepositoryIndex',
            index=0,
            containing_service=None,
            input_type=_REPOSITORYINDEXREQUEST,
            output_type=_REPOSITORYINDEXRESPONSE,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name='RepositoryModelLoad',
            full_name='inference.model_repository.ModelRepositoryService.RepositoryModelLoad',
            index=1,
            containing_service=None,
            input_type=_REPOSITORYMODELLOADREQUEST,
            output_type=_REPOSITORYMODELLOADRESPONSE,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name='RepositoryModelUnload',
            full_name='inference.model_repository.ModelRepositoryService.RepositoryModelUnload',
            index=2,
            containing_service=None,
            input_type=_REPOSITORYMODELUNLOADREQUEST,
            output_type=_REPOSITORYMODELUNLOADRESPONSE,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
)
_sym_db.RegisterServiceDescriptor(_MODELREPOSITORYSERVICE)

DESCRIPTOR.services_by_name['ModelRepositoryService'] = _MODELREPOSITORYSERVICE

# @@protoc_insertion_point(module_scope)
