import os
import sys
sys.path.append(os.environ["INTENTPRED_PATH"])

import unittest
import subprocess

import program
from utils import constants, argument_utils


class TestNoErrorsFullSystem(unittest.TestCase):
    # TODO: make a more variable path system so that I 
    # can create a testing directory for the data and then delete.\
    # TODO: create test dataset once I can change the path easier.

    def test_program_full_stack(self):
        # testing combine and augment together because order matters. 
        # augment depends on combine.
        peachtree, lankershim = program.main(
            argument_utils.parse_args(False, ["c"])
        )
        os.remove(peachtree)
        os.remove(lankershim)
        peachtree, lankershim = program.main(
            argument_utils.parse_args(False, ["c"])
        )
        self.assertTrue(os.path.exists(peachtree))
        self.assertTrue(os.path.exists(lankershim))
        
        # done with combine

        program.main(argument_utils.parse_args(
            False, 
            ["a", "--filenames", "trajectories-peachtree.txt", "trajectories-lankershim.txt"]
        ))
        self.assertTrue(os.path.exists(
            os.path.join(constants.PATH_TO_RESOURCES, "Peachtree", "AUGv2_trajectories-peachtree.txt")
        ))
        self.assertTrue(os.path.exists(
            os.path.join(constants.PATH_TO_RESOURCES, "Lankershim", "AUGv2_trajectories-lankershim.txt")
        ))
        
        # done with augment
        
        program.main(argument_utils.parse_args(
            False, 
            ["f", "--featurize_type", "i", "--test_nums", "000"]
        ))
        for intersection in range(1,10):
            path_to_features_for_intersection = os.path.join(
                constants.PATH_TO_RESULTS, "ByIntersection", "000", str(intersection))
            self.assertTrue(os.path.exists(path_to_features_for_intersection))
            self.assertTrue(os.path.exists(
                os.path.join(path_to_features_for_intersection, "featuresAndTargets")
            ))
            self.assertTrue(os.path.exists(
                os.path.join(path_to_features_for_intersection, "LSTM_Formatted_featuresAndTargets.npy")
            ))
        
        # done with featureizing

        program.main(argument_utils.parse_args(
            False, 
            ["t", "--models", "test", "--test_nums", "000", "--test_intersections", "1"]
        ))

        # done training some models

        program.main(argument_utils.parse_args(
            False, 
            ["e", "--models", "test", "--test_nums", "000", "--test_intersections", "1"]
        ))

        # done with some evaluation



if __name__ == '__main__':
    unittest.main()