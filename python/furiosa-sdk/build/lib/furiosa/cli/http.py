import sys

from furiosa import consts, __version__

DEFAULT_HEADERS = {
    'User-Agent': 'FuriosaCli %s (Python %s.%s.%s)' % (__version__,
                                                       sys.version_info.major, sys.version_info.minor,
                                                       sys.version_info.micro),
    consts.FURIOSA_API_VERSION_HEADER: consts.FURIOSA_API_VERSION_VALUE,  # version 2
    consts.FURIOSA_SDK_VERSION_HEADER: consts.FURIOSA_SDK_VERSION_VALUE
}
