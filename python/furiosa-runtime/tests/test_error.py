import re
import unittest

from furiosa.runtime.errors import DeviceBusy

versionPattern = r'\d+(=?\.(\d+(=?\.(\d+)*)*)*)*'
regexMatcher = re.compile(versionPattern)


class TestFuriosaError(unittest.TestCase):
    def test_error(self):
        error = DeviceBusy()
        self.assertEqual(error.__repr__(), "NPU device busy (native error code: 23)")
        self.assertEqual(error.__str__(), "NPU device busy (native error code: 23)")
        self.assertEqual(error._native_err, 23)


if __name__ == '__main__':
    unittest.main()
