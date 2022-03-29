import re
import unittest

from furiosa import runtime
from furiosa.runtime import LIBNUX

versionPattern = r'\d+(=?\.(\d+(=?\.(\d+)*)*)*)*'
regexMatcher = re.compile(versionPattern)


class TestTensor(unittest.TestCase):
    def test_version(self):
        self.assertTrue(len(runtime.__version__.version) > 0)
        self.assertTrue(regexMatcher.match(LIBNUX.version().decode('utf-8')))
        self.assertTrue(len(str(LIBNUX.git_short_hash())) >= 9)
        self.assertEqual(len(str(LIBNUX.build_timestamp())), 22)


if __name__ == '__main__':
    unittest.main()
