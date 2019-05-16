import argparse

import os
import sys
sys.path.append(os.environ["INTENTPRED_PATH"])
from utils import defaults as D

# for argparsing
def str2bool(v):
    return v.lower() == "true"

### HANDLING PROGRAM MODES ###
PROGRAM_MODES = {
    "c" : "combine", 
    "a" : "augment", 
    "f" : "featurize", 
    "t" : "train", 
    "e" : "evaluate", 
    "x" : "experimental",
}

def expand_mode(mode):
    if mode in PROGRAM_MODES: 
        return PROGRAM_MODES[mode]
    if mode in PROGRAM_MODES.values():
        return mode
    raise ValueError("Invalid mode, should be caught by argparser.")

def all_mode_choices(just_full_name=False):
    if just_full_name:
        return sorted(list(PROGRAM_MODES.values()))
    return list(PROGRAM_MODES.keys()) + list(PROGRAM_MODES.values())

def add_required_arguments(parser):
    """
    Specifies the required arguments.
    """
    parser.add_argument("mode", type=str.lower, choices=all_mode_choices(),
        help="Indicate what process to run. Options are: \n - {}".format("\n - ".join(all_mode_choices(True))))

def add_data_processing_arguments(parser):
    """
    Specifies arguments specific to data processing.
    """
    parser.add_argument("--filenames", default=D.FILENAME, nargs='+',
        help="When augmenting, what is the filename to augment? Separate multiple files with spaces.")
    parser.add_argument("--featurize_type", type=str.lower, default=D.FEATURIZE_TYPE, choices=["s", "i", "n"], 
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
    parser.add_argument("--models", type=str, default=D.MODELS, choices=sorted(model_choices.keys()),
        help="Indicate which set of models to use.")
    parser.add_argument("--test_nums", type=str, default=D.TEST_NUMS,
        help="Enter comma separated list of the test types to use, which indicate which features to use. (e.g. 000,001,010).")
    parser.add_argument("--test_intersections", type=str, default=D.TEST_INTERSECTIONS,
        help="Enter comma separated list of the intersections to produce test results on.")


### HANDLING MODELS ###
nonLSTMs = ["SVM", "BN", "nb", "DNN"]
LSTMs = ["LSTM_128x2", "LSTM_128x3", "LSTM_256x2"]
baselines = ["Marginal", "nb"] 
#nb is naive bayes
model_choices = {
    "all": ["Marginal", "SVM", "BN", "nb", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "bases": baselines,
    "nb": ["nb"],
    "lstms": LSTMs,
    "svm": ["SVM"],
    "bn": ["BN"],
    "svmbn": ["SVM", "BN"],
    "dnn": ["DNN"],
    "nlstms": nonLSTMs,
    "nns": ["DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "nn": ["DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "lstm1": ["LSTM_128x2"],
    "lstm2": ["LSTM_128x3"],
    "lstm3": ["LSTM_256x2"],
    "lstmtest": ["LSTM_test1"],
    "nsvmbn": ["Marginal", "nb", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "nsvm": ["Marginal", "BN", "nb", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "nbasesbn": ["SVM", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "nbasessvm": ["BN", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "nbases": ["SVM", "BN", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "dnnlstm1": ["DNN", "LSTM_128x2"],
    "notnns": ["Marginal", "SVM", "BN", "nb"],
    "test": ["Marginal", "LSTM_128x2"],
}


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
        
