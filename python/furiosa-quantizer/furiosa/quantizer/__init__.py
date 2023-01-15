"""FuriosaAI qunatizer"""

import furiosa_quantizer_impl
from furiosa_quantizer_impl import *

from furiosa.common.utils import get_sdk_version

__version__ = get_sdk_version(__name__)

__all__ = furiosa_quantizer_impl.__all__

del furiosa_quantizer_impl
