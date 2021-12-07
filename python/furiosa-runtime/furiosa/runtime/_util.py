"""Utility module"""
import os
import sys
from typing import Optional

from furiosa.common.native import find_native_lib_path
from furiosa.runtime import __version__, runtime_version


def list_to_dict(items: list) -> dict:
    """Transform a list to a dict with keys derived from indexes"""
    dict_values = {}
    for idx, value in enumerate(items):
        dict_values[idx] = value

    return dict_values


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def dump_info(log_path: Optional[str] = None):
    width = 100  # default
    try:
        width, _ = os.get_terminal_size()
    except:
        pass

    eprint("=" * width)
    eprint("Information Dump")
    eprint("=" * width)
    eprint("- Python version: " + sys.version.replace('\n', ' '))
    eprint(f"- furiosa-libnux path: {find_native_lib_path('nux')}")
    eprint(f"- furiosa-libnux version: {runtime_version()}")
    eprint(f"- furiosa-compiler version: {runtime_version()}")
    eprint(f"- furiosa-runtime version: {__version__}")
    if log_path:
        eprint(
            f"\nPlease check the compiler log at {log_path}.\n"
            f"If you have a problem, please report the log file to https://furiosa-ai.atlassian.net/servicedesk/customer/portals\n"
            f"with the information dumped above."
        )
    eprint("=" * width)
