from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable


class Processor(ABC):
    @abstractmethod
    async def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    async def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __call__(self, func: Callable):
        """
        Rerturn decorator which will call preprocess and postprocess.

        Note that the function signatures (preprocess, infer, postprocess) must be
        compatible to make the pipelines correctly.
        """

        @wraps(
            self.preprocess,
            # Note that we don't have any assigned value. See below
            assigned=(),
        )
        async def decorator(*args: Any, **kwargs: Any) -> Any:
            # Preprocess
            output = await self.preprocess(*args, **kwargs)
            # Infer
            response = await func(output)
            # Postprocess
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
