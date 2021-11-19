"""
Configuration settings via Pydantic
"""

from typing import Dict, List, Optional, Union

from pydantic import BaseSettings, StrictBytes, StrictStr

from furiosa.common.utils import get_sdk_version

from .types import MetadataTensor

__version__ = get_sdk_version("furiosa.server")


class GRPCServerConfig(BaseSettings):
    """
    GRPC server configuration
    """

    host: str = "0.0.0.0"
    port: int = 8081
    workers: int = 1


class RESTServerConfig(BaseSettings):
    """
    Rest server configuration
    """

    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    debug: bool = False


class FileRegistryConfig(BaseSettings):
    """
    File based registry configs.
    """

    repository_root: str = "."


class ServerConfig(BaseSettings):
    """
    Server (GRPC server + REST server) configuration
    """

    # Server metadata for predict API
    server_name: str = "furiosa-server"
    server_version: str = __version__
    extensions: List[str] = []

    # Repository/Registry configs
    intial_model_autoload: bool = True
    file_registry_config: FileRegistryConfig = FileRegistryConfig()

    # Internal server configs
    grpc_server_config: GRPCServerConfig = GRPCServerConfig()
    rest_server_config: RESTServerConfig = RESTServerConfig()


class ModelConfig(BaseSettings):
    """
    Model configuration for a Nux model
    """

    # Model property for Nux
    name: str
    model: Union[StrictStr, StrictBytes]  # File name string or file binary bytes
    version: Optional[str] = None
    description: Optional[str] = None

    npu_device: Optional[str] = None
    compiler_config: Optional[Dict] = None

    # Model metadata for repository API
    platform: str = "nux"
    versions: Optional[List[str]] = []
    inputs: Optional[List[MetadataTensor]] = []
    outputs: Optional[List[MetadataTensor]] = []
