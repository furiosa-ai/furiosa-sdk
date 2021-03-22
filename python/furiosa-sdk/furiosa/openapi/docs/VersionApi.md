# furiosa.openapi.VersionApi

All URIs are relative to *https://api.furiosa.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**version**](VersionApi.md#version) | **GET** /version | 


# **version**
> VersionInfo version()



Get API Server version

### Example

```python
import time
import furiosa.openapi
from furiosa.openapi.api import version_api
from furiosa.openapi.model.version_info import VersionInfo
from pprint import pprint
# Defining the host is optional and defaults to https://api.furiosa.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = furiosa.openapi.Configuration(
    host = "https://api.furiosa.ai"
)


# Enter a context with an instance of the API client
with furiosa.openapi.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = version_api.VersionApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        api_response = api_instance.version()
        pprint(api_response)
    except furiosa.openapi.ApiException as e:
        print("Exception when calling VersionApi->version: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**VersionInfo**](VersionInfo.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Return the server version information |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

