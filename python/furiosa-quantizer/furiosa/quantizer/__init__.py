"""FuriosaAI qunatizer"""

from furiosa_quantizer_impl import CalibrationMethod, Calibrator, quantize

from furiosa.common.utils import get_sdk_version

__version__ = get_sdk_version(__name__)

del get_sdk_version

__all__ = [
    "CalibrationMethod",
    "Calibrator",
    "quantize",
]
