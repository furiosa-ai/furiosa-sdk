import json
import logging
import uuid
from typing import Dict, Tuple

import requests
import yaml
from furiosa import consts, __version__
from furiosa.client import CompilerClient
from requests_toolbelt.multipart.encoder import MultipartEncoder

from . import http
from .exceptions import CliError, ApiError


class ApiKeyAuth(requests.auth.AuthBase):
    def __init__(self, session):
        self.session = session

    def __call__(self, r):
        r.headers[consts.ACCESS_KEY_ID_HTTP_HEADER] = self.session.access_key_id
        r.headers[consts.SECRET_ACCESS_KEY_HTTP_HEADER] = self.session.secret_key_access
        return r


class Command(object):
    def __init__(self, session, args, args_map):
        self.session = session
        self.args = args
        self.args_map = args_map

    def print_message(self, msg):
        if not self.args.quiet:
            print(msg)

    def run(self) -> int:
        pass


def read_config_file(path: str):
    with open(path, 'r') as yaml_file:
        yaml_obj = yaml.safe_load(yaml_file)
        return json.dumps(yaml_obj)


def pretty_yaml(json) -> str:
    return yaml.dump(json, default_flow_style=False)


def handle_target_npu_spec(args) -> str:
    if args.target_npu_spec is not None:
        return read_config_file(args.target_npu_spec)
    else:
        # raise CliError('--target-npu-spec is required')
        return '{}'


def handle_compiler_config(args) -> str:
    if args.config is not None:
        return read_config_file(args.config)
    else:
        return '{}'


def handle_target_ir(args_map) -> str:
    if args_map['target_ir'] not in consts.SUPPORT_TARGET_IRS:
        raise CliError('target-ir must be one of {}'.format(consts.SUPPORT_TARGET_IRS))
    else:
        return args_map['target_ir']


class Version(Command):
    def __init__(self, session, args, args_map):
        super().__init__(session, args, args_map)

    def run(self) -> int:
        request_url = '{}/version'.format(self.session.api_endpoint)

        r = requests.get(request_url,
                         headers=http.DEFAULT_HEADERS,
                         auth=ApiKeyAuth(self.session))

        if r.status_code == 200:
            content = r.json()
            server_version = content['version']
            server_revision = content['revision']
            server_build_time = content['build_time']
            print("Server version: {} (rev: {} built_at: {})"
                  .format(server_version, server_revision, server_build_time))
            print("Client version: {}".format(__version__))
        else:
            print("Client version: {}".format(__version__))
            raise ApiError('fail to get version', r)


def read_yaml_config(path) -> str:
    if path is not None:
        with open(path, 'r') as yaml_file:
            obj = yaml.safe_load(yaml_file)
            return yaml.dump(obj)
    else:
        return '~'


class Compile(Command):
    def __init__(self, session, args, args_map):
        super().__init__(session, args, args_map)

    def run(self) -> int:
        source_path = self.args_map['source']
        compiler_config = read_yaml_config(self.args.config)
        target_npu_spec = read_yaml_config(self.args.target_npu_spec)
        target_ir = handle_target_ir(self.args_map)

        client = CompilerClient()
        with open(source_path, 'rb') as file:
            task = client.submit_compile(source=file,
                                         compiler_config=compiler_config,
                                         target_npu_spec=target_npu_spec)
            task.wait_for_complete()

        if task.is_succeeded():
            if 'o' in self.args and self.args_map['o'] is not None:
                output_path = self.args_map['o']
            else:
                output_path = 'output.{}'.format(target_ir)

            with open(output_path, 'wb') as output_file:
                output_file.write(task.get_ir())
                ms_elapsed = 100
                self.print_message('{} has been generated (elapsed: {} ms)'.format(output_path, ms_elapsed))

            if self.args.compiler_report is not None:
                compiler_report_path = self.args.compiler_report
                with open(compiler_report_path, 'w') as compiler_report_file:
                    compiler_report = task.get_compiler_report()
                    compiler_report_file.write(compiler_report)
                    self.print_message('the compiler report has been written to {}'
                                       .format(compiler_report_path))

            if self.args.mem_alloc_report is not None:
                mem_alloc_report_path = self.args.mem_alloc_report
                with open(self.args.mem_alloc_report, 'w') as mem_alloc_report_file:
                    mem_alloc_report = task.get_memory_alloc_report()
                    mem_alloc_report_file.write(mem_alloc_report)
                    self.print_message('the memory allocation report has been written to {}'
                                       .format(mem_alloc_report_path))
        else:
            raise CliError('fail to compile {}: \n{}'.format(source_path, task.get_error_message()))


class Perf(Command):
    def __init__(self, session, args, args_map, api_path='perf', content_type='csv'):
        super().__init__(session, args, args_map)
        self.api_path = "api/v1/" + api_path
        self.content_type = content_type

    def run(self) -> int:
        source_path = self.args_map['source']
        target_npu_spec = handle_target_npu_spec(self.args)
        compiler_config = handle_compiler_config(self.args)

        if 'o' in self.args and self.args_map['o'] is not None:
            output_path = self.args_map['o']
        else:
            output_path = 'output.{}'.format(self.content_type)

        multi_parts = MultipartEncoder(
            fields={
                'target_npu_spec': target_npu_spec,
                'compiler_config': compiler_config,
                'source': (source_path, open(source_path, mode='rb'), 'application/octet-stream')
            }
        )

        request_url = '{}/{}'.format(self.session.api_endpoint, self.api_path)
        headers = {
            consts.REQUEST_ID_HTTP_HEADER: str(uuid.uuid4()),
            'Content-Type': multi_parts.content_type,
            **http.DEFAULT_HEADERS
        }

        logging.debug("submitting the perf request to {}".format(request_url))
        logging.debug("source path: {}".format(source_path))
        logging.debug("output path: {}".format(output_path))
        logging.debug("target npu spec: \n{}\n".format(pretty_yaml(target_npu_spec)))
        logging.debug("compiler config: \n{}\n".format(pretty_yaml(compiler_config)))

        r = requests.post(request_url,
                          data=multi_parts,
                          headers=headers,
                          auth=ApiKeyAuth(self.session))

        if r.status_code == 200:
            with open(output_path, 'wb') as output_file:
                content = r.content
                output_file.write(content)

                ms_elapsed = r.elapsed.microseconds / 1000
                self.print_message('{} has been generated (elapsed: {} ms)'.format(output_path, ms_elapsed))
        else:
            raise ApiError('fail to estimate the performance {}'.format(source_path), r)


class Perfeye(Perf):
    def __init__(self, session, args, args_map):
        super().__init__(session, args, args_map, api_path='perfeye', content_type='html')


class Optimize(Command):
    def __init__(self, session, args, args_map):
        super().__init__(session, args, args_map)

    @staticmethod
    def optimize(session, model: bytes, model_path: str = None) -> bytes:
        model_path = model_path or 'model.onnx'
        multi_parts = MultipartEncoder(
            fields={
                'source': (model_path, model, 'application/octet-stream')
            }
        )

        request_url = '{}/api/v1/dss/optimize'.format(session.api_endpoint)
        headers = {
            consts.REQUEST_ID_HTTP_HEADER: str(uuid.uuid4()),
            'Content-Type': multi_parts.content_type,
            **http.DEFAULT_HEADERS
        }

        logging.debug("submitting the build calibration model request to {}".format(request_url))
        logging.debug("source path: {}".format(model_path))

        r = requests.post(request_url,
                          data=multi_parts,
                          headers=headers,
                          auth=ApiKeyAuth(session))

        if r.status_code == 200:
            return r.content
        else:
            raise ApiError('fail to build calibration model {}'.format(model_path), r)

    def run(self) -> int:
        source_path = self.args_map['source']

        if 'o' in self.args and self.args_map['o'] is not None:
            output_path = self.args_map['o']
        else:
            output_path = 'output.onnx'

        with open(source_path, 'rb') as model, \
                open(output_path, 'wb') as output_file:
            model = Optimize.optimize(self.session, model.read(), model_path=source_path)
            output_file.write(model)


class BuildCalibrationModel(Command):
    def __init__(self, session, args, args_map):
        super().__init__(session, args, args_map)

    @staticmethod
    def build_calibration_model(session,
                                model: bytes,
                                model_path: str = None) -> bytes:
        model_path = model_path or 'model.onnx'
        multi_parts = MultipartEncoder(
            fields={
                'source': (model_path, model, 'application/octet-stream')
            }
        )

        request_url = '{}/api/v1/dss/build-calibration-model'.format(session.api_endpoint)
        headers = {
            consts.REQUEST_ID_HTTP_HEADER: str(uuid.uuid4()),
            'Content-Type': multi_parts.content_type,
            **http.DEFAULT_HEADERS
        }

        logging.debug("submitting the build calibration model request to {}".format(request_url))
        logging.debug("source path: {}".format(model_path))

        r = requests.post(request_url,
                          data=multi_parts,
                          headers=headers,
                          auth=ApiKeyAuth(session))

        if r.status_code == 200:
            return r.content
        else:
            raise ApiError('fail to build calibration model {}'.format(model_path), r)

    def run(self) -> int:
        source_path = self.args_map['source']

        if 'o' in self.args and self.args_map['o'] is not None:
            output_path = self.args_map['o']
        else:
            output_path = 'output.onnx'

        with open(source_path, 'rb') as model, \
                open(output_path, 'wb') as output_file:
            model = BuildCalibrationModel.build_calibration_model(self.session,
                                                                  model.read(),
                                                                  model_path=source_path)
            output_file.write(model)


class Quantize(Command):
    def __init__(self, session, args, args_map):
        super().__init__(session, args, args_map)

    @staticmethod
    def quantize(session,
                 model: bytes,
                 dynamic_ranges: Dict[str, Tuple[float, float]],
                 model_path: str = None) -> bytes:
        model_path = model_path or 'model.onnx'
        multi_parts = MultipartEncoder(
            fields={
                'dynamic_ranges': json.dumps(dynamic_ranges),
                'source': (model_path, model, 'application/octet-stream')
            }
        )

        request_url = '{}/api/v1/dss/quantize'.format(session.api_endpoint)
        headers = {
            consts.REQUEST_ID_HTTP_HEADER: str(uuid.uuid4()),
            'Content-Type': multi_parts.content_type,
            **http.DEFAULT_HEADERS
        }

        logging.debug("submitting the quantize request to {}".format(request_url))
        logging.debug("source path: {}".format(model_path or 'model.onnx'))
        logging.debug("dynamic ranges: \n{}\n".format(dynamic_ranges))

        r = requests.post(request_url,
                          data=multi_parts,
                          headers=headers,
                          auth=ApiKeyAuth(session))

        if r.status_code == 200:
            return r.content
        else:
            raise ApiError('fail to quantize th model {}'.format(model_path), r)

    def run(self) -> int:
        source_path = self.args_map['source']
        dynamic_ranges = self.args_map['dynamic_ranges']

        if 'o' in self.args and self.args_map['o'] is not None:
            output_path = self.args_map['o']
        else:
            output_path = 'output.onnx'

        with open(source_path, 'rb') as model, \
                open(dynamic_ranges, 'r') as dynamic_ranges, \
                open(output_path, 'wb') as output_file:
            dynamic_ranges = json.load(dynamic_ranges)
            model = Quantize.quantize(self.session,
                                      model.read(),
                                      dynamic_ranges,
                                      model_path=source_path)
            output_file.write(model)


class ToolchainList(Command):
    def __init__(self, session, args, args_map):
        super().__init__(session, args, args_map)

    def run(self) -> int:
        request_url = '{}/api/v1/compiler'.format(self.session.api_endpoint)
        r = requests.get(request_url,
                         headers=http.DEFAULT_HEADERS,
                         auth=ApiKeyAuth(self.session))

        if r.status_code == 200:
            content = r.json()

            print("\nAvailable Toolchains:")
            for idx, toolchain in enumerate(content):
                version = toolchain['version']
                revision = toolchain['revision']
                build_time = toolchain['build_time']
                print("[{}] {} (rev: {} built_at: {})".format(idx, version, revision, build_time))

            print()
        else:
            print("Client version: {}".format(__version__))
            raise ApiError('fail to get version', r)
