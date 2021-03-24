# Furiosa Quantizer 
Static per-channel post-training quantization tool for fp onnx model.

# Requirement

* load submodules
```git submodule update --init```

* install packages
```pip install -r requirements.txt```
# Test model aws link
- [download link](https://s3.console.aws.amazon.com/s3/buckets/furiosa-private-artifacts?region=ap-northeast-2&prefix=onnx-model-exporter/target-dir/&showversions=false)

# Export Spec
- example code: spec을 export하기 전에 항상 optimize_model 함수를 호출해야합니다. 
    ```python
    from quantizer.frontend.onnx import export_spec, optimize_model
    
    model = onnx.load_model(path-to-onnx-model)
    
    model = optimize_model(model)
    export_spec(model, open(path-to-write-spec, 'w'))
    ```
- example code: `torch`에서 `torch.onnx.export`을 사용하여 모델을 직접 export하는 경우에 `opset_version=12` 옵션을 주어야 합니다.
    ``` python
    import onnx
    import torch
    
    from efficientnet_pytorch import EfficientNet
    from quantizer.frontend.onnx import export_spec, optimize_model
  
  
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model.set_swish(memory_efficient=False)
    dummy_input = torch.randn(10, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "test-b0.onnx", opset_version=12)
    
    
    model = onnx.load_model('test-b0.onnx')
    
    model = optimize_model(model)
    export_spec(model, open('test-b0.spec', 'w'))
    ```

# Quantize all test models
1. [test models](https://s3.console.aws.amazon.com/s3/buckets/furiosa-private-artifacts?region=ap-northeast-2&prefix=onnx-model-exporter/target-dir/&showversions=false)를 로컬에 다운 받습니다.

2. 아래의 python code를 실행시킵니다. 이 때, `MODLE_ROOT`와 `SAVE_ROOT`를 specify합니다.
```python
import onnx

import os
import pathlib

from quantizer.frontend.onnx import post_training_quantization_with_random_calibration
from quantizer.frontend.onnx.quantizer.utils import QuantizationMode

MODEL_ROOT = path-to-model-dir
SAVE_ROOT = path-to-save-dir
model_paths = []

for root, _, files in os.walk(MODEL_ROOT):
    for filename in files:
        if '.onnx' not in filename:
            continue
        model_paths.append(os.path.join(root, filename))

for path in model_paths:
    model_name = os.path.basename(path)
    print('quantize %s' % model_name)
    quant_model = post_training_quantization_with_random_calibration(model=onnx.load_model(path),
                                                                     per_channel=True,
                                                                     static=True,
                                                                     mode=QuantizationMode.dfg,
                                                                     num_data=10)
    save_path = os.path.join(SAVE_ROOT, '[dfg_importable]%s' % model_name)
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    onnx.save_model(quant_model, save_path)
    print('done\n')
```