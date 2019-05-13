import sys
sys.path.append("..")

from lib import constants

import unittest

class TestConstants(unittest.TestCase):
    def test_framerate(self):
        self.assertEqual(constants.t_frame, 0.1)

class SampleTestCase(unittest.TestCase):
    def test_python_runs(self):
        for _ in range(10000):
            self.assertEqual(2+2, 4)

if __name__ == '__main__':
    unittest.main()