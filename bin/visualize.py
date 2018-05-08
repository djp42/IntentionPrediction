import numpy as np
from lib import frame_util as futil
import sys, os


def main(argv):
	frameDict = futil.LoadDictFromTxt("res/Lankershim/aug_trajectories-0750am-0805am.txt", 'frame')
	futil.AnimateFrames(frameDict)

if __name__ == "__main__":
    main(sys.argv)
