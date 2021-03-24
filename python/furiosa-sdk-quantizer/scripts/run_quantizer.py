from typing import List
import argparse
import time
import os
import json
import traceback
import pathlib
import urllib.parse
import subprocess

import onnx
from ml_metadata import MetadataStore

from quantizer.frontend.onnx import spec
from quantizer.frontend.onnx import optimize_model
from quantizer.frontend.onnx.quantizer.calibrator import ONNXCalibrator
from quantizer.frontend.onnx.quantizer import quantizer
from weaver.utils import query
from weaver import utils
from weaver.datasource.k8s_office import K8SOffice
from weaver.dss.export_spec import ExportSpec
from weaver.dss.calibrate import Calibrate
from weaver.dss.quantize import Quantize
from weaver.onnx_model_exporter.export_model import OnnxModelExporter


def allowed_models() -> List[str]:
    with open(pathlib.Path(__file__).parent.joinpath('allowed_models'), 'r') as f:
        return list(map(lambda s: s.strip(), f.readlines()))


def get_commit_hash() -> str:
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").strip()
    print(f'GIT COMMIT HASH: {commit_hash}')
    return commit_hash


def get_store(use_fake_mlmd=False) -> MetadataStore:
    if use_fake_mlmd:
        return utils.store.mlmd_fake_store()
    else:
        return utils.store.mlmd_store(
            os.environ["MLMD_HOST"],
            int(os.environ["MLMD_PORT"]),
            os.environ["MLMD_DATABASE"],
            os.environ["MLMD_USER"],
            os.environ["MLMD_PASSWORD"])


def init_mlmd(claim_name: str = None, use_fake_mlmd=False) -> (ExportSpec, Calibrate, Quantize):
    store = get_store(use_fake_mlmd)
    if use_fake_mlmd:
        return ExportSpec(store), Calibrate(store), Quantize(store)
    return ExportSpec(store,
                      K8SOffice(claim_name=claim_name, basepath=pathlib.Path(""))), \
           Calibrate(store,
                     K8SOffice(claim_name=claim_name, basepath=pathlib.Path(""))), \
           Quantize(store,
                    K8SOffice(claim_name=claim_name, basepath=pathlib.Path("")))


def apply_model(model_names: List[str],
                claim_root_path: str,
                mlmd_onnx_model_exporter_root_path: str,
                mlmd_onnx_model_exporter_commit_hash: str,
                num_calib_data: int = 1,
                claim_name: str = None,
                use_fake_mlmd: bool = False,
                commit_hash: str = None) -> List[bool]:
    print(f'original models ({len(model_names)}): {" ".join(model_names)}')
    model_names = list(filter(lambda model_name: model_name in allowed_models(), model_names))
    print(f'filtered models ({len(model_names)}), {" ".join(model_names)}')

    apply_model_begins = time.time()
    mlmd_onnx_model_exporter_root_path = pathlib.Path(mlmd_onnx_model_exporter_root_path).absolute()

    if commit_hash is None:
        commit_hash = get_commit_hash()

    spec_path = pathlib.Path(commit_hash).joinpath("spec")
    dynamic_range_path = pathlib.Path(commit_hash).joinpath("dynamic_range")
    quantized_model_path = pathlib.Path(commit_hash).joinpath("quantized")
    print(f"spec path: {spec_path}")
    print(f"dynamic range path: {dynamic_range_path}")
    print(f"quantized model path: {quantized_model_path}")
    output_path = pathlib.Path(claim_root_path).absolute()
    if not output_path.exists():
        print(f'mkdir output dir {output_path}')
        os.makedirs(output_path, exist_ok=True)
    if not output_path.joinpath(spec_path).exists():
        print(f'mkdir output dir {output_path.joinpath(spec_path)}')
        os.makedirs(output_path.joinpath(spec_path), exist_ok=True)
    if not output_path.joinpath(dynamic_range_path).exists():
        print(f'mkdir output dir {output_path.joinpath(dynamic_range_path)}')
        os.makedirs(output_path.joinpath(dynamic_range_path), exist_ok=True)
    if not output_path.joinpath(quantized_model_path).exists():
        print(f'mkdir output dir {output_path.joinpath(quantized_model_path)}')
        os.makedirs(output_path.joinpath(quantized_model_path), exist_ok=True)

    mlmd_export_spec, mlmd_calibrate, mlmd_quantize = init_mlmd(claim_name, use_fake_mlmd)
    store = mlmd_export_spec.store
    artifact_models = query.list_artifacts(store,
                                           OnnxModelExporter(store).context_type_export_model.name,
                                           mlmd_onnx_model_exporter_commit_hash)
    artifact_models = list(filter(
        lambda artifact_model: artifact_model.properties["model_name"].string_value in model_names, artifact_models))
    print(f'Process target models: [{len(artifact_models)}]')

    results = []
    for artifact_model in artifact_models:
        artifact_begins = time.time()

        uri = urllib.parse.urlparse(artifact_model.uri)
        model_name = artifact_model.properties["model_name"].string_value
        artifact_model_path = mlmd_onnx_model_exporter_root_path.joinpath(pathlib.Path(uri.path).relative_to('/'))
        print(f'[{model_name}] Read model from {artifact_model_path}')

        spec_filename = pathlib.Path(uri.path).with_suffix('').with_suffix('.spec').name
        artifact_spec, execution_spec = mlmd_export_spec.export_spec(
            commit_hash, spec_path.joinpath(spec_filename), artifact_model)
        artifact_spec.properties["model_name"].string_value = model_name
        store.put_artifacts([artifact_spec])
        execution_spec.properties["state"].string_value = "RUNNING"
        store.put_executions([execution_spec])
        print(f'[{model_name}] spec file uri: {artifact_spec.uri}')

        # Get model
        with open(artifact_model_path, 'rb') as readable:
            begins = time.time()
            model = onnx.load_model(readable, onnx.helper.ModelProto)
            print(f'[{model_name}] Read model, elapsed {time.time() - begins:.6f}s', flush=True)

        try:
            begins = time.time()
            model = optimize_model(model)
            print(f'[{model_name}] Optimize model, elapsed {time.time() - begins:.6f}s', flush=True)
        except Exception as e:
            print(f'[{model_name}]: Failed to optimize model, elapsed {time.time() - begins:.6f}s \n{e}\n',
                  flush=True)
            traceback.print_tb(e.__traceback__)
            print('', flush=True)
            execution_spec.properties["state"].string_value = "FAILED"
            execution_spec.properties["reason"].string_value = traceback.format_exc()
            store.put_executions([execution_spec])
            results.append(False)
            continue

        print(f'[{model_name}] Export', flush=True)
        export_specs_begins = time.time()
        try:
            with open(output_path.joinpath(spec_path, spec_filename), 'w') as f:
                begins = time.time()
                spec.export_spec.OnnxExportSpec(model).dump(f)
                print(f'[{model_name}]: Exported, {spec_filename}, elapsed {time.time() - begins:.6f}s',
                      flush=True)

            execution_spec.properties["state"].string_value = "COMPLETED"
            store.put_executions([execution_spec])

        except Exception as e:
            print(
                f'[{model_name}]: Failed to export, elapsed {time.time() - export_specs_begins:.6f}s \n{e}\n',
                flush=True)
            traceback.print_tb(e.__traceback__)
            print('', flush=True)
            execution_spec.properties["state"].string_value = "FAILED"
            execution_spec.properties["reason"].string_value = traceback.format_exc()
            store.put_executions([execution_spec])
            # Ignore export spec failures

        os.environ['TQDM_DISABLE'] = 'True'
        print(f'[{model_name}] Calibrate', flush=True)
        calibrate_begins = time.time()

        dynamic_range_filename = pathlib.Path(uri.path).with_suffix('').with_suffix('.dynamic_range').name
        artifact_dynamic_range, execution_calibrate = mlmd_calibrate.calibrate_from_random_buffer(
            commit_hash, dynamic_range_path.joinpath(dynamic_range_filename), artifact_model)
        artifact_dynamic_range.properties["model_name"].string_value = model_name
        execution_calibrate.properties["state"].string_value = "RUNNING"
        store.put_artifacts([artifact_dynamic_range])
        store.put_executions([execution_calibrate])
        print(f'[{model_name}] dynamic range file uri: {artifact_dynamic_range.uri}')

        try:
            if not num_calib_data:
                print(f'[{model_name}] num clib data = 2')
                num_calib_data = 2

            begins = time.time()
            clib_model = ONNXCalibrator(model).build_calibration_model()
            print(f'[{model_name}]: Build calibration model, elapsed {time.time() - begins:.6f}s',
                  flush=True)

            begins = time.time()
            dynamic_ranges = ONNXCalibrator(clib_model).calibrate_with_random(num_calib_data)
            with open(output_path.joinpath(dynamic_range_path, dynamic_range_filename), 'w') as f:
                json.dump(dynamic_ranges, f, ensure_ascii=True, indent=2)
            print(
                f'[{model_name}]: Calibrate with random data ({num_calib_data}), elapsed {time.time() - begins:.6f}s',
                flush=True)

            execution_calibrate.properties["state"].string_value = "COMPLETED"
            store.put_executions([execution_calibrate])
        except Exception as e:
            print(
                f'[{model_name}]: Failed to calibrate, elapsed {time.time() - calibrate_begins:.6f}s \n{e}\n',
                flush=True)
            traceback.print_tb(e.__traceback__)
            print('', flush=True)
            execution_calibrate.properties["state"].string_value = "FAILED"
            execution_calibrate.properties["reason"].string_value = traceback.format_exc()
            store.put_executions([execution_spec])
            results.append(False)
            continue

        print(f'[{model_name}] Quantize', flush=True)
        quantize_begins = time.time()

        quantized_model_filename = pathlib.Path(uri.path).name
        artifact_quantized_model, execution_quantize = \
            mlmd_quantize.quantize(commit_hash,
                                   quantized_model_path.joinpath(quantized_model_filename),
                                   artifact_model,
                                   artifact_dynamic_range)
        artifact_quantized_model.properties["model_name"].string_value = model_name
        execution_quantize.properties["state"].string_value = "RUNNING"
        store.put_artifacts([artifact_quantized_model])
        store.put_executions([execution_quantize])
        print(f'[{model_name}] quantized model file uri: {artifact_quantized_model.uri}')

        try:
            begins = time.time()
            model = quantizer.FuriosaONNXQuantizer(model,
                                                   per_channel=True,
                                                   static=True,
                                                   mode=quantizer.QuantizationMode.dfg,
                                                   dynamic_ranges=dynamic_ranges).quantize()
            print(f'[{model_name}]: Quantized, elapsed {time.time() - begins:.6f}s',
                  flush=True)
            with open(output_path.joinpath(quantized_model_path, quantized_model_filename), 'wb') as f:
                begins = time.time()
                onnx.save_model(model, f)
                print(
                    f'[{model_name}]: Done to quantize, {quantized_model_filename}, elapsed {time.time() - begins:.6f}s',
                    flush=True)
            execution_quantize.properties["state"].string_value = "COMPLETED"
            store.put_executions([execution_quantize])
        except Exception as e:
            print(f'[{model_name}]: Failed to quantized, elapsed {time.time() - quantize_begins:.6f}s \n{e}\n',
                  flush=True)
            traceback.print_tb(e.__traceback__)
            print('', flush=True)
            execution_quantize.properties["state"].string_value = "FAILED"
            execution_quantize.properties["reason"].string_value = traceback.format_exc()
            store.put_executions([execution_quantize])
            results.append(False)
            continue

        print(f'[{model_name}]: Done to export & quantize, elapsed {time.time() - artifact_begins:.6f}s', flush=True)
        results.append(True)

    print(f'[{len(artifact_models)} models] Done to export & quantize, elapsed {time.time() - apply_model_begins:.6f}s',
          flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSS')
    parser.add_argument('model_names', type=str, nargs='+', help="target model names")
    parser.add_argument('--claim-root-path', type=str, required=True)
    parser.add_argument('--num-calib-data', type=int, default=2)
    parser.add_argument('--mlmd-onnx-model-exporter-root-path', type=str)
    parser.add_argument('--mlmd-onnx-model-exporter-commit-hash', type=str)
    parser.add_argument('--claim-name', type=str)
    parser.add_argument('--use-fake-mlmd', default=False, action="store_true")
    args = parser.parse_args()
    apply_model(model_names=args.model_names,
                num_calib_data=args.num_calib_data,
                mlmd_onnx_model_exporter_root_path=args.mlmd_onnx_model_exporter_root_path,
                mlmd_onnx_model_exporter_commit_hash=args.mlmd_onnx_model_exporter_commit_hash,
                claim_root_path=args.claim_root_path,
                claim_name=args.claim_name,
                use_fake_mlmd=args.use_fake_mlmd)
