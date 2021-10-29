# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

# FIXME(yan): This is manually modified
from . import model_repository_pb2 as model__repository__pb2


class ModelRepositoryServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RepositoryIndex = channel.unary_unary(
            '/inference.model_repository.ModelRepositoryService/RepositoryIndex',
            request_serializer=model__repository__pb2.RepositoryIndexRequest.SerializeToString,
            response_deserializer=model__repository__pb2.RepositoryIndexResponse.FromString,
        )
        self.RepositoryModelLoad = channel.unary_unary(
            '/inference.model_repository.ModelRepositoryService/RepositoryModelLoad',
            request_serializer=model__repository__pb2.RepositoryModelLoadRequest.SerializeToString,
            response_deserializer=model__repository__pb2.RepositoryModelLoadResponse.FromString,
        )
        self.RepositoryModelUnload = channel.unary_unary(
            '/inference.model_repository.ModelRepositoryService/RepositoryModelUnload',
            request_serializer=model__repository__pb2.RepositoryModelUnloadRequest.SerializeToString,
            response_deserializer=model__repository__pb2.RepositoryModelUnloadResponse.FromString,
        )


class ModelRepositoryServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RepositoryIndex(self, request, context):
        """Get the index of model repository contents."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RepositoryModelLoad(self, request, context):
        """Load or reload a model from a repository."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RepositoryModelUnload(self, request, context):
        """Unload a model."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelRepositoryServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'RepositoryIndex': grpc.unary_unary_rpc_method_handler(
            servicer.RepositoryIndex,
            request_deserializer=model__repository__pb2.RepositoryIndexRequest.FromString,
            response_serializer=model__repository__pb2.RepositoryIndexResponse.SerializeToString,
        ),
        'RepositoryModelLoad': grpc.unary_unary_rpc_method_handler(
            servicer.RepositoryModelLoad,
            request_deserializer=model__repository__pb2.RepositoryModelLoadRequest.FromString,
            response_serializer=model__repository__pb2.RepositoryModelLoadResponse.SerializeToString,
        ),
        'RepositoryModelUnload': grpc.unary_unary_rpc_method_handler(
            servicer.RepositoryModelUnload,
            request_deserializer=model__repository__pb2.RepositoryModelUnloadRequest.FromString,
            response_serializer=model__repository__pb2.RepositoryModelUnloadResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'inference.model_repository.ModelRepositoryService', rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class ModelRepositoryService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RepositoryIndex(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/inference.model_repository.ModelRepositoryService/RepositoryIndex',
            model__repository__pb2.RepositoryIndexRequest.SerializeToString,
            model__repository__pb2.RepositoryIndexResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def RepositoryModelLoad(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/inference.model_repository.ModelRepositoryService/RepositoryModelLoad',
            model__repository__pb2.RepositoryModelLoadRequest.SerializeToString,
            model__repository__pb2.RepositoryModelLoadResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def RepositoryModelUnload(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/inference.model_repository.ModelRepositoryService/RepositoryModelUnload',
            model__repository__pb2.RepositoryModelUnloadRequest.SerializeToString,
            model__repository__pb2.RepositoryModelUnloadResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
