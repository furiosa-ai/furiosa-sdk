import asyncio
import importlib
import logging
import os
import types
from typing import Any, Callable

from ...artifact import Artifact
from ...model import Model
from ...utils import python_path, working_directory
from ..transport import download, is_relative
from .base import Resolver

Logger = logging.getLogger(__name__)


class CodeResolver(Resolver):
    """Model resolver from a artifact with a code (Pytorch Module, TensorFlow SavedModel)."""

    async def resolve(self, uri: str, artifact: Artifact, *args: Any, **kwargs: Any) -> Model:
        assert is_relative(
            artifact.location
        ), "'location' should be relative path for 'code' format."

        model = await self.load(uri, artifact.name, artifact.location, *args, **kwargs)

        # Override values provided by the artifact when it exists
        model.name = artifact.name
        model.description = (
            artifact.metadata and artifact.metadata.description
        ) or model.description
        model.version = artifact.version or model.version

        if artifact.doc:
            model.__doc__ = (await self.read(uri, artifact.doc)).decode()

        return model

    async def load(self, uri: str, name: str, path: str, *args: Any, **kwargs: Any) -> Model:
        directory = await download(uri)

        entry = self.import_entry(directory, name, path)

        # Replace working directory to use file system dependent function like 'open' in a registry.
        with python_path(directory), working_directory(directory):
            if asyncio.iscoroutinefunction(entry):
                return await entry(*args, **kwargs)

            return entry(*args, **kwargs)

    @staticmethod
    def import_entry(directory: str, name: str, path: str) -> Callable:
        def get_attr(module: types.ModuleType, name: str) -> Callable:
            if name not in dir(module):
                raise RuntimeError(f"Cannot find '{name}' in module '{path}'")

            entry = getattr(module, name)

            if not callable(entry):
                raise RuntimeError(f"'{name}' in module '{path}' is not callable")

            return entry

        # Replace working directory to use file system dependent function like 'open' in a registry.
        with python_path(directory), working_directory(directory):
            # Remove file extension .py: models/model.py -> models/model
            path = os.path.splitext(path)[0]
            # Replace slash(/) with dot(.): models/model -> models.model
            path = path.replace("/", ".")

            try:
                module = importlib.import_module(path)
            except ModuleNotFoundError as e:
                Logger.error(
                    f"Module dependencies for the loaded code not found. "
                    f"You should install required dependencies for loaded module '{path}'. "
                    f"See error trace to identify the missing module"
                )
                raise e

            entry = get_attr(module, name)

        return entry
