from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .base import Transport


class TransportError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class TransportNotFound(TransportError):
    def __init__(self, uri: str, transports: List["Transport"]):
        msg = f"Transport for {uri} not found. Available transports:\n\n"

        msg += "\n".join(
            f"{type(transport).__name__}: {transport.__doc__}" for transport in transports
        )

        super().__init__(msg)
