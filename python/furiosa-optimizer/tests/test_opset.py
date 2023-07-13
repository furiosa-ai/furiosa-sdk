from furiosa.optimizer.frontend.onnx import optimize_model


def test_opset_version(make_test_model):
    model = make_test_model(opset_version=11)
    opset = 12
    model = optimize_model(model, opset_version=opset)

    assert model.opset_import[0].version, opset


def test_opset_version_1(make_test_model):
    model = make_test_model(opset_version=12)
    opset = 13
    model = optimize_model(model, opset_version=opset)

    assert model.opset_import[0].version, opset
