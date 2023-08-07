from collections.abc import MutableMapping
from io import TextIOWrapper
from typing import Dict, List, Union

import yaml

from ..settings import FuriosaRTModelConfig, ModelConfig, ServerConfig


def load_model_config(stream: Union[TextIOWrapper, Dict]) -> List[ModelConfig]:
    """Load model configs from opened file (file-like object) or Python dictionary."""
    source = stream if isinstance(stream, MutableMapping) else yaml.safe_load(stream)

    def config(model):
        return FuriosaRTModelConfig if model["platform"] == "npu" else ModelConfig

    return [config(model).model_validate(model) for model in source["model_config_list"]]


def load_server_config(stream: Union[TextIOWrapper, Dict]) -> ServerConfig:
    """Load a server config from opened file (file-like object) or Python dictionary."""
    source = stream if isinstance(stream, MutableMapping) else yaml.safe_load(stream)
    return ServerConfig.model_validate(source)
