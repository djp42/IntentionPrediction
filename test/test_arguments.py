import os
import sys
sys.path.append(os.environ["INTENTPRED_PATH"])

import unittest

from utils import argument_utils, defaults

class TestArguments(unittest.TestCase):
    def test_defaults(self):
        args = argument_utils.parse_args(parse_args=False)
        self.assertEqual(args.mode, defaults.MODE)
        self.assertEqual(args.filenames, defaults.FILENAME)
        self.assertEqual(args.featurize_type, defaults.FEATURIZE_TYPE)
        self.assertEqual(args.models, defaults.MODELS)
        self.assertEqual(args.test_nums, defaults.TEST_NUMS)
        self.assertEqual(args.test_intersections, defaults.TEST_INTERSECTIONS)

    def test_non_defaults(self):
        mode = "evaluate"
        filename = "testfile.txt"
        featurize = "n"
        models = "lstm1"
        testnums = "000,001,111"
        test_intersections = "1,2,3,4"

        nondefault_arglist = [
            mode,
            "--filenames", filename,
            "--featurize", featurize,
            "--models", models,
            "--test_nums", testnums,
            "--test_intersections", test_intersections
        ]

        args = argument_utils.parse_args(parse_args=False, arglist=nondefault_arglist)
        self.assertEqual(args.mode, mode)
        self.assertEqual(args.filenames, [filename])
        self.assertEqual(args.featurize_type, featurize)
        self.assertEqual(args.models, models)
        self.assertEqual(args.test_nums, testnums)
        self.assertEqual(args.test_intersections, test_intersections)

    def test_multiple_filenames(self):
        args = argument_utils.parse_args(
            parse_args=False, arglist=["a", "--filenames", "testfile1.txt", "testfile2.txt"])
        self.assertEqual(args.filenames, ["testfile1.txt", "testfile2.txt"])

    def test_flags(self):
        flag_shorthand_arglist = ["evaluate", "-d", "-s", "-q", "-l", "-x"]
        args = argument_utils.parse_args(parse_args=False, arglist=flag_shorthand_arglist)
        self.assertTrue(args.make_distance_histograms)
        self.assertTrue(args.save_scores)
        self.assertTrue(args.quiet_eval)
        self.assertTrue(args.load_scores)
        self.assertTrue(args.excel)

        flag_longhand_arglist = [
            "evaluate", 
            "--make_distance_histograms", 
            "--save_scores", 
            "--quiet_eval", 
            "--load_scores", 
            "--excel"]

        args = argument_utils.parse_args(parse_args=False, arglist=flag_longhand_arglist)
        self.assertTrue(args.make_distance_histograms)
        self.assertTrue(args.save_scores)
        self.assertTrue(args.quiet_eval)
        self.assertTrue(args.load_scores)
        self.assertTrue(args.excel)
    
    #def test_invalid_options(self):
    
    def test_expand_mode(self):
        for key, value in argument_utils.PROGRAM_MODES.items():
            self.assertEqual(argument_utils.expand_mode(key), value)
            self.assertEqual(argument_utils.expand_mode(value), value)
        with self.assertRaises(ValueError):
            argument_utils.expand_mode("DOESNOTEXIST1234132")

if __name__ == '__main__':
    unittest.main()