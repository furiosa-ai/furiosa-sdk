import os
import time
import unittest
import uuid

from furiosa.config import load_furiosa_config, set_apikey
from furiosa.openapi import ApiClient
from furiosa.openapi.api.account_v1_api import ApiKey, AccountV1Api
from furiosa.openapi.api.compiler_v1_api import CompilerV1Api
from furiosa.openapi.api.version_api import VersionApi
from furiosa.openapi.models import VersionInfo, LoginRequest, ApiKeyRequest, ApiKeyPatch
from furiosa.utils import login_account
from tests import test_data


class TestOpenAPIs(unittest.TestCase):
    client = None
    account_api = None
    version_api = None
    access_key_id= None
    compiler_api= None

    @classmethod
    def setUpClass(cls) -> None:
        load_furiosa_config()
        cls.client = ApiClient()
        if os.getenv('FURIOSA_USERNAME') is not None:
            cls.client = login_account(ApiClient())
        cls.account_api = AccountV1Api(api_client=cls.client)
        cls.version_api = VersionApi(api_client=cls.client)

        request = ApiKeyRequest(name=uuid.uuid4().__str__())
        apikey: ApiKey = cls.account_api.create_api_key(request=request)
        # keep for cleaning
        cls.access_key_id = apikey.access_key_id
        set_apikey(cls.client.configuration, apikey.access_key_id, apikey.secret_access_key)
        cls.compiler_api = CompilerV1Api(api_client=cls.client)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.account_api.patch_api_key(cls.access_key_id,
                                       request=ApiKeyPatch(active=False))

    def test_version(self):
        version: VersionInfo = self.version_api.version()
        self.assertEqual(version.version, "0.2.0")

    def test_login(self):
        request = LoginRequest(email=os.environ['FURIOSA_USERNAME'], password=os.environ['FURIOSA_PASSWORD'])
        authenticated = self.account_api.login(request=request)
        self.assertTrue(authenticated.access_token is not None)
        self.assertTrue(authenticated.expires_at is not None)
        self.assertEqual(authenticated.token_type, 'Bearer')

    def test_apikey_set(self):
        names = []
        for idx in range(0, 5):
            names.append(uuid.uuid4().__str__())

        for name in names:
            name = name.__str__()
            request = ApiKeyRequest(name=name, description=name)
            apikey: ApiKey = self.account_api.create_api_key(request=request)
            self.assertEqual(apikey.name, name)
            self.assertEqual(apikey.description, name)
            self.assertTrue(apikey.active)

        apikey_list: [ApiKey] = self.account_api.list_api_keys()
        apikey_dict = {key.name: key for key in apikey_list}

        for name in names:
            self.assertTrue(name in apikey_dict)

        for name in names:
            apikey = apikey_dict[name]
            self.account_api.patch_api_key(apikey.access_key_id,
                                      request=ApiKeyPatch(active=False))

    def test_compile(self):
        request_id = uuid.uuid4().__str__();
        with open(test_data('MNISTnet_uint8_quant_without_softmax.tflite'), 'rb') as file:
            response = self.compiler_api.create_task(x_request_id=request_id,
                                                     source=file)
            self.assertTrue(response.task_id is not None)
            phase = response.phase;
            while phase == 'Running' or phase == 'Pending':
                time.sleep(1)
                response = self.compiler_api.get_task(task_id=response.task_id)
                phase = response.phase

            self.assertEqual(phase, "Succeeded")
            artifacts = self.compiler_api.list_artifacts(task_id=response.task_id)
            for artifact in artifacts:
                if artifact.name == 'output.enf':
                    self.compiler_api.get_artifact(task_id=response.task_id, name=artifact.name, _preload_content=False)
                else:
                    self.compiler_api.get_artifact(task_id=response.task_id, name=artifact.name)

    def test_error(self):
        request_id = uuid.uuid4().__str__();
        compiler_config = {
            'keep_unsignedness': 1
        }
        with open(test_data('MNISTnet_uint8_quant_without_softmax.tflite'), 'rb') as file:
            response = self.compiler_api.create_task(x_request_id=request_id,
                                                     source=file, compiler_config=compiler_config)
            self.assertTrue(response.task_id is not None)
            print(response.task_id)
            phase = response.phase;
            while phase == 'Running' or phase == 'Pending':
                time.sleep(1)
                response = self.compiler_api.get_task(task_id=response.task_id)
                phase = response.phase

            self.assertEqual(phase, "Failed")
            self.assertTrue(response.error_message is not None)
            print(response.error_message)


if __name__ == '__main__':
    if os.getenv('FURIOSA_USERNAME') is not None:
        unittest.main()
    else:
        pass
