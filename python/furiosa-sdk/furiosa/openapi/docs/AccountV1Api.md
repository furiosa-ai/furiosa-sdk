# furiosa.openapi.AccountV1Api

All URIs are relative to *https://api.furiosa.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_api_key**](AccountV1Api.md#create_api_key) | **POST** /api/account/v1alpha1/apikeys | Generate new API key
[**get_account**](AccountV1Api.md#get_account) | **GET** /api/account/v1alpha1/me | Get my account information
[**get_api_key**](AccountV1Api.md#get_api_key) | **GET** /api/account/v1alpha1/apikeys/{access_key_id} | Get a API key
[**list_api_keys**](AccountV1Api.md#list_api_keys) | **GET** /api/account/v1alpha1/apikeys | List generated API keys
[**login**](AccountV1Api.md#login) | **POST** /api/account/v1alpha1/login | Login
[**patch_api_key**](AccountV1Api.md#patch_api_key) | **PATCH** /api/account/v1alpha1/apikeys/{access_key_id} | Update a API key
[**update_account**](AccountV1Api.md#update_account) | **PUT** /api/account/v1alpha1/me | Update my account information
[**update_password**](AccountV1Api.md#update_password) | **PUT** /api/account/v1alpha1/me/password | Change my account password


# **create_api_key**
> ApiKey create_api_key()

Generate new API key

Generate a new API key

### Example

* Bearer (JWT) Authentication (BearerAuth):
```python
import time
import furiosa.openapi
from furiosa.openapi.api import account_v1_api
from furiosa.openapi.model.api_key_request import ApiKeyRequest
from furiosa.openapi.model.api_key import ApiKey
from pprint import pprint
# Defining the host is optional and defaults to https://api.furiosa.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = furiosa.openapi.Configuration(
    host = "https://api.furiosa.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = furiosa.openapi.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with furiosa.openapi.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = account_v1_api.AccountV1Api(api_client)
    request = ApiKeyRequest(
        name="name_example",
        description="description_example",
    ) # ApiKeyRequest |  (optional)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Generate new API key
        api_response = api_instance.create_api_key(request=request)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling AccountV1Api->create_api_key: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **request** | [**ApiKeyRequest**](ApiKeyRequest.md)|  | [optional]

### Return type

[**ApiKey**](ApiKey.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successfully new API key has been generated. |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_account**
> InlineResponse200 get_account()

Get my account information

### Example

* Bearer (JWT) Authentication (BearerAuth):
```python
import time
import furiosa.openapi
from furiosa.openapi.api import account_v1_api
from furiosa.openapi.model.inline_response200 import InlineResponse200
from pprint import pprint
# Defining the host is optional and defaults to https://api.furiosa.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = furiosa.openapi.Configuration(
    host = "https://api.furiosa.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = furiosa.openapi.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with furiosa.openapi.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = account_v1_api.AccountV1Api(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Get my account information
        api_response = api_instance.get_account()
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling AccountV1Api->get_account: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**InlineResponse200**](InlineResponse200.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Retrieved successfully |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_api_key**
> ApiKey get_api_key(access_key_id)

Get a API key

### Example

* Bearer (JWT) Authentication (BearerAuth):
```python
import time
import furiosa.openapi
from furiosa.openapi.api import account_v1_api
from furiosa.openapi.model.api_response import ApiResponse
from furiosa.openapi.model.api_key import ApiKey
from pprint import pprint
# Defining the host is optional and defaults to https://api.furiosa.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = furiosa.openapi.Configuration(
    host = "https://api.furiosa.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = furiosa.openapi.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with furiosa.openapi.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = account_v1_api.AccountV1Api(api_client)
    access_key_id = "access_key_id_example" # str | API key ID to get

    # example passing only required values which don't have defaults set
    try:
        # Get a API key
        api_response = api_instance.get_api_key(access_key_id)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling AccountV1Api->get_api_key: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **access_key_id** | **str**| API key ID to get |

### Return type

[**ApiKey**](ApiKey.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get a API key |  -  |
**404** | API key is not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_api_keys**
> [ApiKey] list_api_keys()

List generated API keys

List all generated API keys

### Example

* Bearer (JWT) Authentication (BearerAuth):
```python
import time
import furiosa.openapi
from furiosa.openapi.api import account_v1_api
from furiosa.openapi.model.api_key import ApiKey
from pprint import pprint
# Defining the host is optional and defaults to https://api.furiosa.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = furiosa.openapi.Configuration(
    host = "https://api.furiosa.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = furiosa.openapi.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with furiosa.openapi.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = account_v1_api.AccountV1Api(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # List generated API keys
        api_response = api_instance.list_api_keys()
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling AccountV1Api->list_api_keys: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**[ApiKey]**](ApiKey.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | List all API keys |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **login**
> LoginOutput login()

Login

### Example

```python
import time
import furiosa.openapi
from furiosa.openapi.api import account_v1_api
from furiosa.openapi.model.login_request import LoginRequest
from furiosa.openapi.model.login_output import LoginOutput
from pprint import pprint
# Defining the host is optional and defaults to https://api.furiosa.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = furiosa.openapi.Configuration(
    host = "https://api.furiosa.ai"
)


# Enter a context with an instance of the API client
with furiosa.openapi.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = account_v1_api.AccountV1Api(api_client)
    request = LoginRequest(
        email="email_example",
        password="password_example",
        remember_me=True,
    ) # LoginRequest |  (optional)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Login
        api_response = api_instance.login(request=request)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling AccountV1Api->login: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **request** | [**LoginRequest**](LoginRequest.md)|  | [optional]

### Return type

[**LoginOutput**](LoginOutput.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Logged in successfully |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **patch_api_key**
> ApiResponse patch_api_key(access_key_id)

Update a API key

### Example

* Bearer (JWT) Authentication (BearerAuth):
```python
import time
import furiosa.openapi
from furiosa.openapi.api import account_v1_api
from furiosa.openapi.model.api_response import ApiResponse
from furiosa.openapi.model.api_key_patch import ApiKeyPatch
from pprint import pprint
# Defining the host is optional and defaults to https://api.furiosa.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = furiosa.openapi.Configuration(
    host = "https://api.furiosa.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = furiosa.openapi.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with furiosa.openapi.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = account_v1_api.AccountV1Api(api_client)
    access_key_id = "access_key_id_example" # str | API key to be patched
    request = ApiKeyPatch(
        active=True,
    ) # ApiKeyPatch |  (optional)

    # example passing only required values which don't have defaults set
    try:
        # Update a API key
        api_response = api_instance.patch_api_key(access_key_id)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling AccountV1Api->patch_api_key: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Update a API key
        api_response = api_instance.patch_api_key(access_key_id, request=request)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling AccountV1Api->patch_api_key: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **access_key_id** | **str**| API key to be patched |
 **request** | [**ApiKeyPatch**](ApiKeyPatch.md)|  | [optional]

### Return type

[**ApiResponse**](ApiResponse.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | the API key has been updated successfully |  -  |
**404** | API key is not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_account**
> ApiResponse update_account()

Update my account information

### Example

* Bearer (JWT) Authentication (BearerAuth):
```python
import time
import furiosa.openapi
from furiosa.openapi.api import account_v1_api
from furiosa.openapi.model.api_response import ApiResponse
from furiosa.openapi.model.inline_object import InlineObject
from pprint import pprint
# Defining the host is optional and defaults to https://api.furiosa.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = furiosa.openapi.Configuration(
    host = "https://api.furiosa.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = furiosa.openapi.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with furiosa.openapi.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = account_v1_api.AccountV1Api(api_client)
    inline_object = InlineObject(
        surname="surname_example",
        given_name="given_name_example",
        email="email_example",
        password="password_example",
        company="company_example",
        phone="phone_example",
    ) # InlineObject |  (optional)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Update my account information
        api_response = api_instance.update_account(inline_object=inline_object)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling AccountV1Api->update_account: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **inline_object** | [**InlineObject**](InlineObject.md)|  | [optional]

### Return type

[**ApiResponse**](ApiResponse.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | My account has been updated successfully |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_password**
> ApiResponse update_password()

Change my account password

### Example

* Bearer (JWT) Authentication (BearerAuth):
```python
import time
import furiosa.openapi
from furiosa.openapi.api import account_v1_api
from furiosa.openapi.model.api_response import ApiResponse
from furiosa.openapi.model.inline_object1 import InlineObject1
from pprint import pprint
# Defining the host is optional and defaults to https://api.furiosa.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = furiosa.openapi.Configuration(
    host = "https://api.furiosa.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = furiosa.openapi.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with furiosa.openapi.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = account_v1_api.AccountV1Api(api_client)
    request = InlineObject1(
        old_password="old_password_example",
        new_password="new_password_example",
    ) # InlineObject1 |  (optional)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Change my account password
        api_response = api_instance.update_password(request=request)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling AccountV1Api->update_password: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **request** | [**InlineObject1**](InlineObject1.md)|  | [optional]

### Return type

[**ApiResponse**](ApiResponse.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | My account password has been updated successfully |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

