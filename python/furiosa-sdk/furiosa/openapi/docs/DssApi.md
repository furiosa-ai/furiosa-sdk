# furiosa.openapi.DssApi

All URIs are relative to *https://api.furiosa.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**api_dss_v1alpha1_build_calibration_model_post**](DssApi.md#api_dss_v1alpha1_build_calibration_model_post) | **POST** /api/dss/v1alpha1/build-calibration-model | Calibrate a model and return the calibrated one
[**api_dss_v1alpha1_quantize_post**](DssApi.md#api_dss_v1alpha1_quantize_post) | **POST** /api/dss/v1alpha1/quantize | Calibrate a model and return the calibrated one


# **api_dss_v1alpha1_build_calibration_model_post**
> file_type api_dss_v1alpha1_build_calibration_model_post(x_request_id, source, input_tensors)

Calibrate a model and return the calibrated one

Calibrate specific input tensors of a given model and return the calibrated model

### Example

* Api Key Authentication (AccessKeyIdAuth):
* Api Key Authentication (SecretAccessKeyAuth):
```python
import time
import furiosa.openapi
from furiosa.openapi.api import dss_api
from furiosa.openapi.model.api_response import ApiResponse
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
    api_instance = dss_api.DssApi(api_client)
    x_request_id = "X-Request-ID_example" # str | 
    source = open('/path/to/file', 'rb') # file_type | a byte array of a source image
    input_tensors = "input_tensors_example" # str | an array of input tensor names

    # example passing only required values which don't have defaults set
    try:
        # Calibrate a model and return the calibrated one
        api_response = api_instance.api_dss_v1alpha1_build_calibration_model_post(x_request_id, source, input_tensors)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling DssApi->api_dss_v1alpha1_build_calibration_model_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **x_request_id** | **str**|  |
 **source** | **file_type**| a byte array of a source image |
 **input_tensors** | **str**| an array of input tensor names |

### Return type

**file_type**

### Authorization

[AccessKeyIdAuth](../README.md#AccessKeyIdAuth), [SecretAccessKeyAuth](../README.md#SecretAccessKeyAuth)

### HTTP request headers

 - **Content-Type**: multipart/form-data
 - **Accept**: application/octet-stream, application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successfully calibrated |  -  |
**400** | Bad request |  -  |
**401** | Unauthorized. It can happen when no API key is set or the given API key is invalid. |  -  |
**500** | Internal error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **api_dss_v1alpha1_quantize_post**
> file_type api_dss_v1alpha1_quantize_post(x_request_id, source, input_tensors)

Calibrate a model and return the calibrated one

Calibrate specific input tensors of a given model and return the calibrated model

### Example

* Api Key Authentication (AccessKeyIdAuth):
* Api Key Authentication (SecretAccessKeyAuth):
```python
import time
import furiosa.openapi
from furiosa.openapi.api import dss_api
from furiosa.openapi.model.api_response import ApiResponse
from furiosa.openapi.model.dynamic_range import DynamicRange
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
    api_instance = dss_api.DssApi(api_client)
    x_request_id = "X-Request-ID_example" # str | 
    source = open('/path/to/file', 'rb') # file_type | a byte array of a source image
    input_tensors = "input_tensors_example" # str | an array of input tensor names
    dynamic_ranges = DynamicRange(
        key={
            key="key_example",
            value=[
                1,
            ],
        },
    ) # DynamicRange |  (optional)

    # example passing only required values which don't have defaults set
    try:
        # Calibrate a model and return the calibrated one
        api_response = api_instance.api_dss_v1alpha1_quantize_post(x_request_id, source, input_tensors)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling DssApi->api_dss_v1alpha1_quantize_post: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Calibrate a model and return the calibrated one
        api_response = api_instance.api_dss_v1alpha1_quantize_post(x_request_id, source, input_tensors, dynamic_ranges=dynamic_ranges)
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling DssApi->api_dss_v1alpha1_quantize_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **x_request_id** | **str**|  |
 **source** | **file_type**| a byte array of a source image |
 **input_tensors** | **str**| an array of input tensor names |
 **dynamic_ranges** | [**DynamicRange**](DynamicRange.md)|  | [optional]

### Return type

**file_type**

### Authorization

[AccessKeyIdAuth](../README.md#AccessKeyIdAuth), [SecretAccessKeyAuth](../README.md#SecretAccessKeyAuth)

### HTTP request headers

 - **Content-Type**: multipart/form-data
 - **Accept**: application/octet-stream, application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successfully calibrated |  -  |
**400** | Bad request |  -  |
**401** | Unauthorized. It can happen when no API key is set or the given API key is invalid. |  -  |
**500** | Internal error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

