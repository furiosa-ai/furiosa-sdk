from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .transport.base import Transport


class RegistryError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class TransportNotFound(RegistryError):
    def __init__(self, uri: str, transports: List["Transport"]):
        msg = f"Transport for {uri} not found. Available transport:\n\n"

        msg += "\n".join(
            f"{type(transport).__name__}: {transport.__doc__}"
            for transport in transports
        )

        super().__init__(msg)


class URINotFound(RegistryError):
    def __init__(self, uri: str):
        msg = f"{uri} not found. Check the URI is valid."

        super().__init__(msg)
