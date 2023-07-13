import re
import unittest

from furiosa import runtime
from furiosa.runtime import __full_version__


class TestVersion(unittest.TestCase):
    def test_version(self):
        self.assertTrue(len(runtime.__version__.version) > 0)
        self.assertTrue(runtime.__version__.stage in "dev", "release")
        self.assertTrue(len(runtime.__version__.hash) > 0)

    def test_full_version(self):
        # Furiosa SDK Runtime 0.10.0-dev (rev: 705853b9) (libnux 0.10.0-dev c3412f038 2023-07-11T03:15:45Z)
        # Furiosa SDK Runtime 0.10.0-dev (rev: 705853b9) (furiosa-rt 0.1.0-dev 2bd01c5a9 2023-07-14T01:01:24Z)
        matcher = re.compile(r'Furiosa SDK Runtime (\S+) \(rev: (\S+)\) \((\S+ \S+ \S+ \S+)\)')
        self.assertTrue(matcher.match(runtime.__full_version__))


if __name__ == '__main__':
    unittest.main()
