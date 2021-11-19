from typing import Generic, TypeVar

T = TypeVar('T')


class Transformer(Generic[T]):
    def transform(self, model: T) -> T:
        raise NotImplementedError()
