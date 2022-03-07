import onnx
from onnx import checker
import onnxruntime as ort


def check_model(model: onnx.ModelProto, check_runnable: bool = True) -> None:
    """
    Check if model's well-defined and executable on onnxruntime
    """
    # TODO After collecting possible errors,
    #  pass through only if all error messages are "No opset import for domain 'com.microsoft'".
    #  The code below is only to see the first error encountered.
    acceptable_error_msg = [
        "No opset import for domain 'com.microsoft'",
        'No Op registered for LayerNormalization with domain_version of 12',
    ]
    try:
        checker.check_model(model)
    except checker.ValidationError as e:
        if str(e).split("==>", maxsplit=1)[0].rstrip() in acceptable_error_msg:
            pass
        else:
            checker.check_model(model)

    if check_runnable:
        ort.set_default_logger_severity(3)
        ort.InferenceSession(model.SerializeToString())
