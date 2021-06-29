import logging as log
import os
import pkgutil

from furiosa.openapi import ApiClient, Configuration
from furiosa.openapi.api.account_v1_api import AccountV1Api
from furiosa.openapi.model.login_request import LoginRequest


def get_sdk_version(module):
    """Returns the git commit hash representing the current version of the application."""
    git_version = None
    try:
        git_version = str(pkgutil.get_data(module, 'git_version'), encoding="UTF-8")
    except Exception as e:  # pylint: disable=broad-except
        log.debug(e)

    return git_version


def login_account(client: ApiClient):
    request = LoginRequest(email=os.environ['FURIOSA_USERNAME'],
                           password=os.environ['FURIOSA_PASSWORD'])
    account_api = AccountV1Api()
    auth = account_api.login(request=request)
    client.configuration.access_token = auth.access_token
    Configuration.set_default(client.configuration)
    return client
