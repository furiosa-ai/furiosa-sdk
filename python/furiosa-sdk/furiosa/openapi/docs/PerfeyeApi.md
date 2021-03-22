# furiosa.openapi.PerfeyeApi

All URIs are relative to *https://api.furiosa.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**api_profiler_v1alpha1_perfeye_post**](PerfeyeApi.md#api_profiler_v1alpha1_perfeye_post) | **POST** /api/profiler/v1alpha1/perfeye | Generate a visualized performance estimation


# **api_profiler_v1alpha1_perfeye_post**
> str api_profiler_v1alpha1_perfeye_post(x_request_id, source)

Generate a visualized performance estimation

It will generate a single HTML containing a DAG with estimated performance results

### Example

* Api Key Authentication (AccessKeyIdAuth):
* Api Key Authentication (SecretAccessKeyAuth):
```python
import time
import furiosa.openapi
from furiosa.openapi.api import perfeye_api
from furiosa.openapi.model.api_response import ApiResponse
from furiosa.openapi.model.kv_config import KvConfig
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

# Configure API key authorization: AccessKeyIdAuth
configuration.api_key['AccessKeyIdAuth'] = 'YOUR_API_KEY'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['AccessKeyIdAuth'] = 'Bearer'

# Configure API key authorization: SecretAccessKeyAuth
configuration.api_key['SecretAccessKeyAuth'] = 'YOUR_API_KEY'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['SecretAccessKeyAuth'] = 'Bearer'

# Enter a context with an instance of the API client
with furiosa.openapi.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = perfeye_api.PerfeyeApi(api_client)
    x_request_id = "X-Request-ID_example" # str | 
    source = open('/path/to/file', 'rb') # file_type | a byte array of a source image
    target_npu_spec = KvConfig(
        key=None,
    ) # KvConfig |  (optional)
    compiler_config = KvConfig(
        key=None,
    ) # KvConfig |  (optional)

    # example passing only required values which don't have defaults set
    try:
        # Generate a visualized performance estimation
        api_response = api_instance.api_profiler_v1alpha1_perfeye_post(x_request_id, source)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling PerfeyeApi->api_profiler_v1alpha1_perfeye_post: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Generate a visualized performance estimation
        api_response = api_instance.api_profiler_v1alpha1_perfeye_post(x_request_id, source, target_npu_spec=target_npu_spec, compiler_config=compiler_config)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling PerfeyeApi->api_profiler_v1alpha1_perfeye_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **x_request_id** | **str**|  |
 **source** | **file_type**| a byte array of a source image |
 **target_npu_spec** | [**KvConfig**](KvConfig.md)|  | [optional]
 **compiler_config** | [**KvConfig**](KvConfig.md)|  | [optional]

### Return type

**str**

### Authorization

[AccessKeyIdAuth](../README.md#AccessKeyIdAuth), [SecretAccessKeyAuth](../README.md#SecretAccessKeyAuth)

### HTTP request headers

 - **Content-Type**: multipart/form-data
 - **Accept**: text/html, application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successfully compiled |  -  |
**400** | Bad request |  -  |
**401** | Unauthorized. It can happen when no API key is set or the given API key is invalid. |  -  |
**500** | Internal error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

