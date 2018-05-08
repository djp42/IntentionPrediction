import numpy as np
from lib import frame_util as futil
from lib import merger_methods as mm
from lib import constants

import sys, os

#Creates augmented trajectory files with Vy and Ay data
def augmentTrajectories():
    for subdir, dirs, files in os.walk(constants.PATH_TO_RESOURCES):
        print (files)
        for file in files:
            filepath = subdir + os.sep + file
            if not filepath.endswith(".txt") or not file[:3] == 'tra': 
                continue     
            futil.processBaseFile(filepath)

def findMergers():
    print("Finding merger vehicle information...")
    for (filepath, filename) in constants.paths:
        open(filepath+'.txt', 'r')
        print("Starting merge stuff for:", filepath+'.txt')
        #mm.doMinRangesAndStartForMerges(filepath, constants.LaneID, constants.VehicleID,
        #                                constants.FrameID, constants.TotFrames) #MergeLane=7
        if 'compressed' in filename:
                mm.doRangesAndStartForMerges(filepath, 9, constants.VehicleID,
                                        constants.FrameID, constants.TotFrames,
                                        filename) #MergeLane=7
        else:
            mm.doRangesAndStartForMerges(filepath, constants.LaneID, constants.VehicleID,
                                        constants.FrameID, constants.TotFrames,
                                        filename) #MergeLane=7

def main(argv):
    augmentTrajectories()
    #findMergers()

if __name__ == "__main__":
    main(sys.argv)
