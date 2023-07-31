import re
import unittest

from furiosa import runtime
from furiosa.runtime import __full_version__


def test_version():
    assert len(runtime.__version__.version) > 0
    assert runtime.__version__.stage in ("dev", "release")
    assert len(runtime.__version__.hash) > 0


def test_full_version():
    # Furiosa SDK Runtime 0.10.0-dev (rev: 705853b9) (libnux 0.10.0-dev c3412f038 2023-07-11T03:15:45Z) # noqa: E501
    # Furiosa SDK Runtime 0.10.0-dev (rev: 705853b9) (furiosa-rt 0.1.0-dev 2bd01c5a9 2023-07-14T01:01:24Z) # noqa: E501
    matcher = re.compile(r"Furiosa SDK Runtime (\S+) \(rev: (\S+)\) \((\S+ \S+ \S+ \S+)\)")
    assert matcher.match(runtime.__full_version__)
