Furiosa Models
==============

Furiosa models provides ready to use deeplearning models provided by FuriosaAI.

## Available models

- MLCommonsResNet50
- MLCommonsMobileNet
- MLCommonsSSDResNet34

## Usage

```python
import asyncio

from furiosa.registry import Model
from furiosa.models.vision import MLCommonsResNet50


model: Model = asyncio.run(MLCommonsResNet50())
```
