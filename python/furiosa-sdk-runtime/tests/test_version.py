import unittest

from furiosa.runtime import LIBNUX


class TestTensor(unittest.TestCase):
    def test_version(self):
        self.assertEqual(LIBNUX.version(), b"0.2.2")
        self.assertTrue(len(str(LIBNUX.git_short_hash())) >= 9)
        self.assertEqual(len(str(LIBNUX.build_timestamp())), 22)


if __name__ == '__main__':
    unittest.main()
