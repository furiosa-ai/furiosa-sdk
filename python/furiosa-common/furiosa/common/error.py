import ctypes
import typing


class FuriosaError(Exception):
    """general exception caused by Furiosa Runtime"""

    def __init__(self, message: str):
        self._message = message

    @property
    def message(self) -> str:
        """Error message"""
        return self._message

    def __repr__(self):
        return '{}'.format(self._message)

    def __str__(self):
        return self.__repr__()


def is_ok(err: typing.Union[ctypes.c_int, int]) -> bool:
    """True if NuxErr is SUCCESS, or False"""
    # FIXME (@hyunsik): C APIs defined in ctypes returns c_int, or int value.
    #   There's no way to make the behavior deterministic.
    if isinstance(err, ctypes.c_int):
        err = err.value
    elif isinstance(err, int):
        pass

    return err == 0


def is_err(err: typing.Union[ctypes.c_int, int]) -> bool:
    """True if NuxErr is not SUCCESS, or False"""
    # FIXME (@hyunsik): C APIs defined in ctypes returns c_int, or int value.
    #   There's no way to make the behavior deterministic.
    if isinstance(err, ctypes.c_int):
        err = err.value
    elif isinstance(err, int):
        pass

    return err != 0
