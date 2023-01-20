"""FuriosaAI qunatizer"""

from furiosa_quantizer_impl import CalibrationMethod, Calibrator, quantize

import furiosa.common.utils

__version__ = furiosa.common.utils.get_sdk_version(__name__)

__all__ = [
    "CalibrationMethod",
    "Calibrator",
    "quantize",
]
