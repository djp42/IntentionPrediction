import unittest

class SampleTestCase(unittest.TestCase):
    def test_python_runs(self):
        for _ in range(10000):
            self.assertEqual(2+2, 4)

if __name__ == '__main__':
    unittest.main()