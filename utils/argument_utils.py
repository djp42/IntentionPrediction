import argparse

import os
import sys
sys.path.append(os.environ["INTENTPRED_PATH"])
from utils import defaults as D

# for argparsing
def str2bool(v):
    return v.lower() == "true"

def add_required_arguments(parser):
    """
    Specifies the required arguments.
    """
    modes = ["combine", "augment", "featurize", "train", "evaluate", "experimental"]
    mode_shorthands = ["c", "a", "f", "t", "e", "x"]
    parser.add_argument("mode", type=str.lower, choices=modes+mode_shorthands,
        help="Indicate what process to run. Options are: \n - {}".format("\n - ".join(modes)))

def add_data_processing_arguments(parser):
    """
    Specifies arguments specific to data processing.
    """
    parser.add_argument("--filename", type=str, default=D.FILENAME, 
        help="When augmenting, what is the filename to augment?")
    parser.add_argument("--featurize_type", type=str.lower, default=D.FEATURIZE_TYPE, choices = ["s", "i", "n"], 
        help="When featurizing:\n s -- save featurized data \n i -- save by intersection, \n n -- no saving")
    
def add_eval_flags(parser):
    """
    Specifies arguments specific to the evaluation process.
    """
    parser.add_argument("-d", "--make_distance_histograms", action="store_true",
        help="Flag whether or not to create distance histograms when evaluating.")
    parser.add_argument("-s", "--save_scores", action="store_true",
        help="Flag whether or not to save scores when evaluating.")
    parser.add_argument("-q", "--quiet_eval", action="store_true",
        help="Flag whether or not to only print the final outputs when evaluating.")
    parser.add_argument("-l", "--load_scores", action="store_true",
        help="Flag whether or not to load scores when evaluating.")
    parser.add_argument("-x", "--excel", action="store_true",
        help="Flag whether or not to save to excel when evaluating. Forces '-s' and '-l' as well.")

def add_general_arguments(parser):
    """
    Arguments that are shared across multiple different processes.
    """
    parser.add_argument("--test_nums", type=str, default=D.TEST_NUMS,
        help="Enter comma separated list of the test types to use, which indicate which features to use. (e.g. 000,001,010).")
    parser.add_argument("--test_intersections", type=str, default=D.TEST_INTERSECTIONS,
        help="Enter comma separated list of the intersections to produce test results on.")


"""
parse_args() creates a parser object, adds the arguments and default arguments,
  and concludes by returning the parsed arugments.

Arguments:
    parse_args 
        - a boolean about whether or not to parse.
        - Set to false to use in unit tests (does not try to get arguments from commandline.)
    arglist
        - a list of the arguments, pased to the parser to overwrite default values if parse_args=False.

Returns:
  args: an object with all of the function arguments. 
"""
def parse_args(parse_args=True, arglist=[D.MODE]):
    parser = argparse.ArgumentParser()
    add_required_arguments(parser)
    add_data_processing_arguments(parser)
    add_eval_flags(parser)
    add_general_arguments(parser)
    if parse_args:
        return parser.parse_args()
    else:
        return parser.parse_args(arglist)
        
