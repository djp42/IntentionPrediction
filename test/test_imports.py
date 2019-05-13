import unittest

class SampleTestCase(unittest.TestCase):
    def test_python_runs(self):
        for _ in range(100):
            self.assertEqual(2+2, 4)
