"""FuriosaAI qunatizer"""

from furiosa.common.utils import get_sdk_version
import furiosa_quantizer_impl
from furiosa_quantizer_impl import *

__version__ = get_sdk_version(__name__)

__all__ = furiosa_quantizer_impl.__all__

del furiosa_quantizer_impl
