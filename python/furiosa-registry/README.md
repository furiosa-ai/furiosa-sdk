Furiosa Registry
----------------

Furiosa Registry defines `Model` which allows client to communicate with several types of registries.

## Getting started

**Install**

```sh
pip install furiosa-registry
```

**List up available models**

```python
import furiosa.common.thread as synchronous
import furiosa.registry as registry


# Repository where to load models from.
repository = "https://github.com/furiosa-ai/furiosa-artifacts"

# List up the available artifacts.
models: List[registry.Model] = synchronous(registry.list)(repository)

for model in models:
    print(model.name)
```

**Load models**

```python
import furiosa.common.thread as synchronous
import furiosa.registry as registry


# Repository where to load models from.
repository = "https://github.com/furiosa-ai/furiosa-artifacts"

models: registry.Model = synchronous(registry.load)(uri=repository, name="MLCommonsResNet50")

# Access the model
print(model.name)
print(model.version)
print(model.metadata.description)
```

**Print documentation about a model**

```python
import furiosa.common.thread as synchronous
import furiosa.registry as registry


# Repository where to load models from.
repository = "https://github.com/furiosa-ai/furiosa-artifacts"

print(synchronous(registry.help)(repository, "MLCommonsResNet50"))
```

## Development

**Generate artfiact JSON schema from pydantic model definition.**

When you changed model schema, you can generate modified schema via

```sh
python -c 'from furiosa.registry import Model;\
            print(Model.schema_json(indent=2), file=open("model_schema.json", "w"))'
```

See `model_schema.json`
