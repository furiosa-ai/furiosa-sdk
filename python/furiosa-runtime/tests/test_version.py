import re
import unittest

from furiosa import runtime
from furiosa.runtime import LIBNUX

versionPattern = r'\d+(=?\.(\d+(=?\.(\d+)*)*)*)*'
regexMatcher = re.compile(versionPattern)


class TestVersion(unittest.TestCase):
    def test_version(self):
        self.assertTrue(len(runtime.__version__.version) > 0)
        self.assertTrue(regexMatcher.match(LIBNUX.version().decode('utf-8')))
        self.assertTrue(len(LIBNUX.git_short_hash()) >= 9)
        self.assertEqual(20, len(LIBNUX.build_timestamp()))


if __name__ == '__main__':
    unittest.main()
