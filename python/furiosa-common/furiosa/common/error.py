import ctypes
import typing


class FuriosaError(Exception):
    """General exception caused by Furiosa Runtime"""

    def __init__(self, message: str):
        self._message = message

    def __repr__(self):
        return '{}'.format(self._message)

    def __str__(self):
        return self.__repr__()


def is_ok(err: typing.Union[ctypes.c_int, int]) -> bool:
    return (err.value if isinstance(err, ctypes.c_int) else err) == 0


def is_err(err: typing.Union[ctypes.c_int, int]) -> bool:
    return not is_ok(err)
