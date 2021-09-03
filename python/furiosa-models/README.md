Furiosa Models
--------------

Furiosa Models provide ready-to-use ML models running on FuriosaAI runtime.

## Available models

- MLCommons_ResNet50_V1_5
- MLCommons_MobileNetV1
- MLCommons_SSD1200_ResNet34


## Usage

```python
from furiosa.registry import Model
from furiosa.models.vision import MLCommons_ResNet50_V1_5


model: Model = MLCommons_ResNet50_V1_5()
```
