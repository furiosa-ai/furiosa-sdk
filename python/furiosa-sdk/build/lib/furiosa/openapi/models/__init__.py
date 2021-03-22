# flake8: noqa

# import all models into this package
# if you have many models here with many references from one model to another this may
# raise a RecursionError
# to avoid this, import only the models that you directly need like:
# from from furiosa.openapi.model.pet import Pet
# or import this package, but before doing it, use:
# import sys
# sys.setrecursionlimit(n)

from furiosa.openapi.model.api_key import ApiKey
from furiosa.openapi.model.api_key_patch import ApiKeyPatch
from furiosa.openapi.model.api_key_request import ApiKeyRequest
from furiosa.openapi.model.api_response import ApiResponse
from furiosa.openapi.model.artifact_meta import ArtifactMeta
from furiosa.openapi.model.compile_task import CompileTask
from furiosa.openapi.model.compile_task_list import CompileTaskList
from furiosa.openapi.model.inline_object import InlineObject
from furiosa.openapi.model.inline_object1 import InlineObject1
from furiosa.openapi.model.inline_response200 import InlineResponse200
from furiosa.openapi.model.login_output import LoginOutput
from furiosa.openapi.model.login_request import LoginRequest
from furiosa.openapi.model.toolchains import Toolchains
from furiosa.openapi.model.version_info import VersionInfo
