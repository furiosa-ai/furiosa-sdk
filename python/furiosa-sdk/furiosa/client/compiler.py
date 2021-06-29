"""Compiler Client and ways to access compile tasks"""
import time
import uuid

from furiosa.config import load_furiosa_config
from furiosa.openapi import Configuration, ApiClient
from furiosa.openapi.api.compiler_v1_api import CompilerV1Api

PENDING = 'Pending'
RUNNING = 'Running'
SUCCEEDED = 'Succeeded'
FAILED = 'Failed'


class CompileTask(object):
    """A compile task and its API"""
    def __init__(self, api: CompilerV1Api, compile_task):
        self.api = api
        self.compile_task = compile_task

    def task_id(self) -> str:
        return self.compile_task.task_id

    def wait_for_complete(self, check_interval=0.5):
        phase = self.compile_task.phase
        while phase == 'Running' or phase == 'Pending':
            time.sleep(check_interval)
            self.compile_task = self.api.get_task(task_id=self.compile_task.task_id)
            phase = self.compile_task.phase

    def is_succeeded(self):
        return self.compile_task.phase == SUCCEEDED

    def is_failed(self):
        return self.compile_task.phase == FAILED

    def is_completed(self):
        return self.is_succeeded() or self.is_failed()

    def phase(self) -> str:
        return self.compile_task.phase

    def list_artifacts(self):
        if self.compile_task.phase == SUCCEEDED:
            return self.api.list_artifacts(task_id=self.compile_task.task_id)

        raise BaseException(self.api.get_log(self.compile_task.task_id))

    def get_ir(self):
        if self.compile_task.phase == SUCCEEDED:
            response = self.api.get_artifact(task_id=self.task_id(),
                                             name='output.enf',
                                             _preload_content=False)
            return response.data

        raise BaseException(self.api.get_log(self.compile_task.task_id))

    def get_compiler_report(self):
        if self.compile_task.phase == SUCCEEDED:
            response = self.api.get_artifact(task_id=self.task_id(),
                                             name='report.txt')
            return response

        raise BaseException(self.api.get_log(self.compile_task.task_id))

    def get_memory_alloc_report(self):
        if self.compile_task.phase == SUCCEEDED:
            response = self.api.get_artifact(task_id=self.task_id(),
                                             name='memory_alloc.html')
            return response

        raise BaseException(self.api.get_log(self.compile_task.task_id))

    def get_dot_graph(self):
        if self.compile_task.phase == SUCCEEDED:
            response = self.api.get_artifact(task_id=self.task_id(),
                                             name='graph.gv')
            return response

        raise BaseException(self.api.get_log(self.compile_task.task_id))

    def get_logs(self):
        return self.api.get_log(self.compile_task.task_id)

    def get_error_message(self):
        if self.is_succeeded():
            return None

        return self.compile_task.error_message


class CompilerClient:  # pylint: disable=too-few-public-methods
    """CompilerClient"""
    def __init__(self):
        self.config = Configuration()
        load_furiosa_config(self.config)
        self.client = ApiClient(configuration=self.config)
        self.api = CompilerV1Api(api_client=self.client)

    def submit_compile(self, source, x_request_id=None,
                       compiler_config='{}', target_npu_spec='{}') -> CompileTask:
        if x_request_id is None:
            x_request_id = uuid.uuid4().__str__()

        response = self.api.create_task(x_request_id=x_request_id,
                                        source=source,
                                        compiler_config=compiler_config,
                                        target_npu_spec=target_npu_spec)
        return CompileTask(self.api, response)
