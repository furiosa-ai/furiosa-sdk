from abc import ABC, abstractmethod
from functools import wraps
import inspect
from typing import Any, Callable

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class Processor(ABC):
    # TODO: Any other graceful way to provide pre/postprocess API?
    # Now, type is entirely useless and client requires type ignore to pass validator.
    @abstractmethod
    async def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        return (*args, kwargs) if kwargs else args

    @abstractmethod
    async def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        return (*args, kwargs) if kwargs else args

    def __call__(self, func: Callable):
        """Return decorator which will call preprocess and postprocess.

        Note that the function signatures (preprocess, infer, postprocess) must be
        compatible to make the pipelines correctly.
        """

        @wraps(
            self.preprocess,
            # Note that we don't have any assigned value. See below
            assigned=(),
        )
        async def decorator(*args: Any, **kwargs: Any) -> Any:
            name = self.__class__.__name__

            # Preprocess
            with tracer.start_as_current_span("{}:preprocess".format(name)):
                output = await self.preprocess(*args, **kwargs)

            # Infer
            with tracer.start_as_current_span("{}:inference".format(name)):
                response = await func(output)

            # Postprocess
            with tracer.start_as_current_span("{}:postprocess".format(name)):
                annotation = inspect.signature(func).return_annotation
                if hasattr(annotation, '__origin__') and annotation.__origin__ is tuple:
                    # Unpack return variables if there are more than two (tuple case)
                    return await self.postprocess(*response)
                else:
                    return await self.postprocess(response)

        # # Extract parameter annotation from preprocess
        params = {
            key: value for key, value in self.preprocess.__annotations__.items() if key != "return"
        }
        # Extract return annotation from postprocess
        returns = {"return": self.postprocess.__annotations__["return"]}

        # Replace original function's annotation with new signature
        decorator.__annotations__ = dict(params, **returns)

        # Assign original attribute to decorator. Do not use wrap() as FastAPI will unwrap it
        for attr in ("__module__", "__name__", "__qualname__"):
            setattr(decorator, attr, getattr(func, attr))

        # Integrate docstring
        decorator.__doc__ = "".join(
            (
                doc
                for doc in (
                    self.preprocess.__doc__,
                    func.__doc__,
                    self.postprocess.__doc__,
                )
                if doc
            )
        )

        return decorator
