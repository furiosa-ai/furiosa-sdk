"""FuriosaAI qunatizer"""

from furiosa.common.utils import get_sdk_version
from furiosa.optimizer.frontend.onnx import optimize_model

__version__ = get_sdk_version(__name__)

__all__ = ["frontend", "interfaces"]
