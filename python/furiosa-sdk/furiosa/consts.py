"""It contains constant values commonly used in the SDK"""

# End points
PRODUCTION_API_ENDPOINT = 'https://api.furiosa.ai'
DEFAULT_API_ENDPOINT = PRODUCTION_API_ENDPOINT

# Shell environment variable list
FURIOSA_API_ENDPOINT_ENV = 'FURIOSA_API_ENDPOINT'
FURIOSA_ACCESS_KEY_ID_ENV = 'FURIOSA_ACCESS_KEY_ID'
SECRET_ACCESS_KEY_ENV = 'FURIOSA_SECRET_ACCESS_KEY'

# HTTP header keys
REQUEST_ID_HTTP_HEADER = 'X-Request-Id'
FURIOSA_API_VERSION_HEADER = 'X-FuriosaAI-API-Version'
FURIOSA_API_VERSION_VALUE = '2'
FURIOSA_SDK_VERSION_HEADER = 'X-FuriosaAI-SDK-Version'
FURIOSA_SDK_VERSION_VALUE = '0.2.1'
ACCESS_KEY_ID_HTTP_HEADER = 'X-FuriosaAI-Access-Key-ID'
SECRET_ACCESS_KEY_HTTP_HEADER = 'X-FuriosaAI-Secret-Access-KEY'

SUPPORT_TARGET_IRS = {'dfg', 'cdfg', 'ldfg', 'gir', 'lir', 'enf'}
