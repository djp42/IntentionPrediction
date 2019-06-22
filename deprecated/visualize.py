import os
import sys
sys.path.append(os.environ["INTENTPRED_PATH"])
import numpy as np
from utils import frame_util as futil

def main(argv):
	frameDict = futil.LoadDictFromTxt("res/Lankershim/AUGv2_trajectories-lankershim.txt", 'frame')
	futil.AnimateFrames(frameDict)

if __name__ == "__main__":
    main(sys.argv)
