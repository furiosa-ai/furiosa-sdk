from typing import List
import argparse
import os
import pathlib
import subprocess
import urllib.parse
import asyncio
import multiprocessing

from ml_metadata.proto import metadata_store_pb2
from ml_metadata import MetadataStore

from weaver import utils
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


def all_models(store, commit_hash: str, mlmd_onnx_model_exporter_root_path: str) -> List[metadata_store_pb2.Artifact]:
    print(
        f'all models from ctx, name: {OnnxModelExporter(store).context_type_export_model.name}, commit hash: {commit_hash}')
    artifacts = utils.query.list_artifacts(store, OnnxModelExporter(store).context_type_export_model.name, commit_hash)
    # https://github.com/furiosa-ai/dss/issues/113
    # https://github.com/furiosa-ai/dss/issues/114
    commented_out_models = [
        'resnext101_32x32d_ig',
        'efficientnet_l2_ns',
        'efficientnet_l2_ns_475',
        'efficientdet_d7x',
    ]
    mlmd_onnx_model_exporter_root_path = pathlib.Path(mlmd_onnx_model_exporter_root_path).absolute()
    allowed = allowed_models()
    models = []
    for artifact in artifacts:
        if artifact.properties["model_name"].string_value.lower() in commented_out_models:
            print(f"Ignore the model, {artifact.properties['model_name'].string_value}")
            continue
        if artifact.properties["model_name"].string_value not in allowed:
            print(f"Ignore the model, {artifact.properties['model_name'].string_value}")
            continue
        path = mlmd_onnx_model_exporter_root_path.joinpath(
            pathlib.Path(urllib.parse.urlparse(artifact.uri).path).relative_to('/'))
        models.append((artifact, path))
    # Ascending order
    models.sort(key=lambda model: model[1].stat().st_size)
    return list(map(lambda model: model[0], models))


async def read_stream(proc, stream):
    while proc.returncode is None:
        data = await stream.readline()
        if data:
            print(data.decode('utf-8').rstrip(), flush=True)
        else:
            break


async def apply_model(model_names: List[str],
                      claim_root_path: str,
                      mlmd_onnx_model_exporter_root_path: str,
                      mlmd_onnx_model_exporter_commit_hash: str,
                      num_calib_data: int = 1,
                      claim_name: str = None,
                      use_fake_mlmd: bool = False) -> (str, int):
    print(f'Process {len(model_names)} models...')
    run_quantizer = pathlib.Path(__file__).absolute().parent.joinpath("run_quantizer.py")
    cmd = f"python {run_quantizer} export \
        {' '.join(model_names)} \
        --claim-root-path {claim_root_path} \
        --claim-name {claim_name} \
        --mlmd-onnx-model-exporter-root-path {mlmd_onnx_model_exporter_root_path} \
        --mlmd-onnx-model-exporter-commit-hash {mlmd_onnx_model_exporter_commit_hash} \
        --num-calib-data {num_calib_data} \
        {'--use-fake-mlmd' if use_fake_mlmd else ''} \
    "
    process = await asyncio.create_subprocess_shell(cmd,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE,
                                                    cwd=pathlib.Path(__file__).absolute().parent)
    asyncio.create_task(read_stream(process, process.stdout))
    asyncio.create_task(read_stream(process, process.stderr))
    await process.wait()
    return cmd, process.returncode


async def apply_all_models(claim_root_path: str,
                           mlmd_onnx_model_exporter_root_path: str,
                           mlmd_onnx_model_exporter_commit_hash: str,
                           num_calib_data: int = None,
                           claim_name: str = None,
                           use_fake_mlmd: bool = False,
                           max_concurrent_tasks: int = 4):
    artifact_models = all_models(get_store(use_fake_mlmd),
                                 mlmd_onnx_model_exporter_commit_hash,
                                 mlmd_onnx_model_exporter_root_path)
    artifact_models_chunks = [[] for _ in range(max_concurrent_tasks)]
    for idx in range(0, len(artifact_models)):
        artifact_models_chunks[idx % max_concurrent_tasks].append(
            artifact_models[idx].properties["model_name"].string_value)
    print(f'chunks: {list(map(lambda artifact_models: len(artifact_models), artifact_models_chunks))}')

    total = len(artifact_models)
    print(f'{total} models will be processed as onnx model.')
    for artifact_model in artifact_models:
        print(f'[{artifact_model.properties["model_name"].string_value}]: {artifact_model.uri}')

    futures = []
    for model_names in artifact_models_chunks:
        futures.append(
            asyncio.ensure_future(apply_model(model_names,
                                              claim_root_path=claim_root_path,
                                              claim_name=claim_name,
                                              mlmd_onnx_model_exporter_root_path=mlmd_onnx_model_exporter_root_path,
                                              mlmd_onnx_model_exporter_commit_hash=mlmd_onnx_model_exporter_commit_hash,
                                              num_calib_data=num_calib_data,
                                              use_fake_mlmd=use_fake_mlmd)))
    done, pending = await asyncio.wait(futures, return_when=asyncio.ALL_COMPLETED)
    assert len(pending) == 0
    assert len(done) == len(artifact_models_chunks)
    results = []
    for future in done:
        results.append(future.result())
    for result in results:
        print(f"RETCODE: {result[1]}, CMD: {result[0]}")
    assert len(list(filter(lambda result: result[1] != 0, results))) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSS')
    parser.add_argument('--claim-root-path', type=str, required=True)
    parser.add_argument('--num-calib-data', type=int, default=2)
    parser.add_argument('--max-concurrent-tasks', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--mlmd-onnx-model-exporter-root-path', type=str)
    parser.add_argument('--mlmd-onnx-model-exporter-commit-hash', type=str)
    parser.add_argument('--claim-name', type=str)
    parser.add_argument('--use-fake-mlmd', default=False, action="store_true")
    args = parser.parse_args()
    asyncio.run(apply_all_models(num_calib_data=args.num_calib_data,
                                 mlmd_onnx_model_exporter_root_path=args.mlmd_onnx_model_exporter_root_path,
                                 mlmd_onnx_model_exporter_commit_hash=args.mlmd_onnx_model_exporter_commit_hash,
                                 claim_root_path=args.claim_root_path,
                                 claim_name=args.claim_name,
                                 use_fake_mlmd=args.use_fake_mlmd,
                                 max_concurrent_tasks=args.max_concurrent_tasks))
