# Comment out S3 tests as moto does not support aiobotocore. I don't want to make mock
# server just to verify basic S3 API, so wait for pytest plugin now.

# import boto3
# import pytest
# from furiosa.registry.client.transport import S3Transport
# from moto import mock_s3


# @pytest.fixture
# def bucket():
#     return "BUCKET_NAME"


# @pytest.fixture(autouse=True)
# @mock_s3
# def prepare(artifact_file, model_file, bucket):
#     """
#     Prepare mock s3 server fixture to serve files.
#     """
#     client = boto3.client("s3")

#     with open(model_file, "rb") as data:
#         client.upload_fileobj(data, bucket, artifact_file)

#     with open(artifact_file, "rb") as data:
#         client.upload_fileobj(data, bucket, model_file)

#     yield


# @pytest.fixture(scope="function")
# def transport() -> S3Transport:
#     return S3Transport()


# @pytest.mark.asyncio
# async def test_listing(transport, artifact_file, artifacts, bucket):
#     assert (await transport.listing(f"s3://{bucket}/{artifact_file}")) == artifacts


# @pytest.mark.asyncio
# async def test_download(transport, model_file, bucket, MNISTnet):
#     assert await transport.download(f"s3://{bucket}/{model_file}") == MNISTnet
