import os, sys

#frames are 0.1 seconds
t_frame = 0.1

#year, month, day of lankershim
Year = 2005
Month = 6
Day = 16

#columns that item corresponds to in data
#These are for lankershim and peachstreet
VehicleID = 0
FrameID = 1
TotFrames = 2
GlobalT = 3
LocalX = 4
LocalY = 5
GlobalX = 6
GlobalY = 7
Len = 8
Wid = 9
Class = 10
Vel = 11
Accel = 12
LaneID = 13
Origin = 14
Dest = 15
Intersect = 16
Section = 17
Dir = 18
Movement = 19
Preceding = 20
Following = 21
SpaceHdwy = 22
TimeHdwy = 23
augVx = 24
augAx = 25
orientation = 26
destLane = 27
goalx = 28
goaly = 29
laneType = 30
nextMove = 31
Distance = 32

#path101 = os.environ["INTENTPRED_PATH"]()+'/res/101_trajectories/'
#path80 = os.environ["INTENTPRED_PATH"]()+'/res/80_trajectories/'
#
#file101_1 = 'aug_trajectories-0750am-0805am'
#file101_2 = 'aug_trajectories-0805am-0820am'
#file101_3 = 'aug_trajectories-0820am-0835am'
#file101 = '101_full_trajectories_compressed'
#file80_1 = 'aug_trajectories-0400-0415'
#file80_2 = 'aug_trajectories-0500-0515'
#file80_3 = 'aug_trajectories-0515-0530'
#paths=[#(path101+file101_1[4:],'res/101_trajectories/'+file101_1+'.txt'), 
#       #(path101+file101_2[4:],'res/101_trajectories/'+file101_2+'.txt'),
#       #(path101+file101_3[4:],'res/101_trajectories/'+file101_3+'.txt'),
#       (path101+file101,'res/101_trajectories/'+file101+'.txt')]
#       #,path80+file80_1, path80+file80_2, path80+file80_3]


PATH_TO_ROOT = None
PATH_TO_RESOURCES = os.path.join(os.environ["INTENTPRED_PATH"], "res")
PATH_TO_RESULTS = os.path.join(os.environ["INTENTPRED_PATH"], "results")
PATH_TO_SCORES = os.path.join(os.environ["INTENTPRED_PATH"], "scores")
PATH_TO_EXECUTABLES = None
PATH_TO_LIBRARIES = None
EXECUTABLES = None
EXE_ARG_POS = None
DEFAULT_EXE_CHOICE = None
MAX_X = 70
MAX_Y = 2250

X_DIV = 10
Y_DIV = 60
MIN_GRID_X = 0
MAX_GRID_X = 40
MIN_GRID_Y = 0
MAX_GRID_Y = 1500
X_STEP = float((MAX_GRID_X - MIN_GRID_X)/X_DIV)
Y_STEP = float((MAX_GRID_Y - MIN_GRID_Y)/Y_DIV)

'''
Moving direction of the vehicle. 
1 - east-bound (EB), 
2 - north-bound (NB), 
3 - west-bound (WB), 
4 - south-bound (SB).

LaneTypes (generalizes to any road):
# 0 is any
# 1 is straight only
# 2 is left only
# 3 is straight or left
# 4 is right only
# 5 is straight or right
# 6 is right only BAY  (bay may affect likelihood of not following laws)
# 7 is left only BAY
'''

# lanetypes to enumerate if lane is right only, left only, etc
# dict of section-dir-lane to type
lanetypes = {
    # start
    '1-4-1':1, '1-4-2':1, '1-4-3':1,
    # intersection 1
    '1-2-1':1, '1-2-2':1,
    '1-3-1':2, '1-3-2':4, '1-3-3':4,
    '2-4-1':1, '2-4-2':1, '2-4-3':1,
    # intersection 2    
    '2-1-1':2, '2-1-2':1, '2-1-3':5,
    '2-2-1':1, '2-2-2':1, '2-2-3':5, '2-2-4':4, '2-2-11':7,
    '2-3-1':3, '2-3-2':1, '2-3-3':5,
    '3-4-1':1, '3-4-2':1, '3-4-3':1, '3-4-11':7, '3-4-12':2, '3-4-31':6, 
    # intersection 3
    '3-1-1':2, '3-1-2':5,
    '3-2-1':1, '3-2-2':1, '3-2-3':1, '3-2-11':7, '3-2-31':6, 
    '3-3-1':2, '3-3-2':5, 
    '4-4-1':1, '4-4-2':1, '4-4-3':1, '4-4-11':7, '4-4-4':4,
    # intersection 4
    '4-1-1':2, '4-1-2':3, '4-1-3':4,
    '4-2-1':1, '4-2-2':1, '4-2-3':1, '4-2-11':7, '4-2-4':4, 
    '4-3-1':3, '4-3-2':5, 
    '5-4-1':1, '5-4-2':1, '5-4-3':1, '5-4-11':7, '5-4-4':5,
    #end
    '5-2-1':1, '5-2-2':1, '5-2-3':1, '5-2-4':1
}

# dict of section-dir-lane to type
peachLaneTypes = {
    # start
    '1-4-1':1, '1-4-2':1,
    # intersection 1
    '1-1-1':2, '1-1-2':1, '1-1-3':1, '1-1-4':4,
    '1-2-1':5, '1-2-2':1, '1-2-11':7,
    '1-3-1':2, '1-3-2':1, '1-3-3':5,
    '2-4-1':1, '2-4-2':5, '2-4-11':7,
    # intersection 2    
    '2-1-1':3, '2-1-2':4,
    '2-2-1':1, '2-2-2':5, 
    '2-3-1':0, 
    '3-4-1':1, '3-4-2':5, '3-4-11':7, 
    # intersection 3
    '3-1-1':2, '3-1-2':5,
    '3-2-1':1, '3-2-2':5, '3-2-11':7, 
    '3-3-1':0,
    '4-4-1':1, '4-4-2':5, '4-4-11':7,
    # intersection 4
    '4-1-1':0,
    '4-2-1':1, '4-2-2':5, '4-2-11':7, 
    '4-3-1':0,
    '5-4-1':3, '5-4-2':1,
    # intersection 5
    '5-1-1':2, '5-1-2':5,
    '5-2-1':1, '5-2-2':5, '5-2-11':7, 
    '5-3-1':2, '5-3-2':1, '5-3-3':5,
    '6-4-1':1, '6-4-2':5, '6-4-11':7,
    #end
    '6-2-1':1, '6-2-2':2,
}

# _ _ _ _
# left allowed, straight allowed, right allowed, bay?
def laneTypeEncoding(laneType):
    laneEncodeDict = {
    0:[1,1,1,0],
    1:[0,1,0,0],
    2:[1,0,0,0],
    3:[1,1,0,0],
    4:[0,0,1,0],
    5:[0,1,1,0],
    6:[0,0,1,1],
    7:[1,0,0,1],
    }
    return laneEncodeDict[int(laneType)]

def getLanesToSidesLankershim(sect, direction, lane, distance):
    if sect == 1:
        if direction == 2:
            return (lane - 1), (2-lane)
        else:
            return (lane - 1), (3-lane)
    elif sect == 2:
        if direction == 2:
            if distance > 150:
                return (lane - 1), (4-lane)
            else:
                curLane = lane
                if curLane == 11:
                    curLane = 0
                return (curLane), (4-curLane)
        else:
            return (lane - 1), (3-lane)
    elif sect == 3:
        if direction == 2:
            if distance > 115:
                return (lane - 1), (3-lane)
            else:
                curLane = lane
                if curLane == 11:
                    curLane = 0
                if curLane == 31:
                    curLane = 4
                return (curLane), (4-curLane)
        elif direction == 4:
            if distance > 320:
                return (lane - 1), (3-lane)
            else:
                curLane = lane
                if curLane == 11:
                    curLane = -1
                if curLane == 12:
                    curLane = 0
                if curLane == 31:
                    curLane = 4
                curLane = curLane + 1
                return (curLane), (5-curLane)
        else:
            return (lane - 1), (2-lane)
    elif sect == 4:
        if direction == 2:
            if distance > 90:
                return (lane - 1), (4-lane)
            else:
                curLane = lane
                if curLane == 11:
                    curLane = 0
                return (curLane), (4-curLane)
        elif direction == 4:
            curLane = lane
            if curLane > 5:
                curLane = 0
            return (curLane, 4-curLane)
        else:
            return (lane - 1), (3-lane)
    elif sect == 5:
        if direction == 2:
            return (lane - 1), (4-lane)
        elif direction == 4:
            curLane = lane
            if curLane == 11:
                curLane = 0
            return (curLane), (4-curLane)
        else:
            return (lane - 1), (3-lane)
    return (-1, -1)
          
def getLanesToSidesPeach(sect, direction, lane, distance):
    maxLanes = 0
    if sect == 1:
        maxLanes = 3
        if direction == 2:
            if lane == 11: lane = 0
            if distance < 115: 
                lane += 1
            else:
                maxLanes -= 1
        elif direction == 1: 
            maxLanes += 1
    elif sect == 2:
        maxLanes = 3
        if direction==4:
            if lane == 11: lane = 0
            lane += 1
        elif direction == 2 or direction == 1:
            maxLanes -= 1
        else:
            maxLanes = 1
    elif sect == 3:
        maxLanes = 2
        if direction == 2:
            if distance < 130:
                maxLanes = 3
                if lane == 11: lane = 0
                lane += 1
        elif direction == 3:
            maxLanes = 1
        elif direction == 4:
            if distance < 120:
                maxLanes = 3
                if lane == 11: lane = 0
                lane += 1
    elif sect == 4:
        maxLanes = 1
        if direction == 2:
            maxLanes = 2
            if distance < 140:
                maxLanes = 3
                if lane == 11: lane = 0
                lane += 1
        elif direction == 4:
            maxLanes = 2
            if distance < 100:
                maxLanes = 3
                if lane == 11: lane = 0
                lane += 1
    elif sect == 5:
        maxLanes = 2
        if direction == 2:
            maxLanes = 3
            if lane == 11:
                lane = 0
            lane += 1
        elif direction == 3:
            maxLanes = 3
    elif sect == 6:
        maxLanes = 2
        if direction == 4:
            maxLanes = 3
            if lane == 11:
                lane = 0
            lane += 1
    
    return (lane - 1), (maxLanes - lane)

#lanesToMedian, lanesToCurb
def getLanesToSides(sect, direction, lane, distance, bool_peach):
    if bool_peach == False:
        return getLanesToSidesLankershim(sect, direction, lane, distance)
    return getLanesToSidesPeach(sect, direction, lane, distance)
        

# X_DIV = 35
# Y_DIV = 200
# X_STEP = float(MAX_X/X_DIV)
# Y_STEP = float(MAX_Y/Y_DIV)

##
# Function: init
# -------------------
# Because of some weird bug with __file and absolute vs. relative paths,
# constants.py must have a separate function to actually configure the
# global constants, using a path_to_root variable fed to it by __main__.py.
#
# Example path_to_root on my computer:
#   "/Users/Alex Lin/Documents/CS229/Project/cs229_merging"
##
def init(pathToRoot):
    global PATH_TO_ROOT
    global PATH_TO_RESOURCES
    global PATH_TO_EXECUTABLES
    global PATH_TO_LIBRARIES
    global EXECUTABLES
    global EXE_ARG_POS
    global DEFAULT_EXE_CHOICE
    global GRID_X
    global GRID_Y
    global X_DIVS
    global Y_DIVS
    global X_STEP
    global Y_STEP

    # This is the absolute path of the recipe_writer folder on your computer.
    PATH_TO_ROOT = pathToRoot

    # The full paths of the folders holding various important things
    PATH_TO_RESOURCES = os.path.join(PATH_TO_ROOT, "res")
    PATH_TO_EXECUTABLES = os.path.join(PATH_TO_ROOT, "bin")
    PATH_TO_LIBRARIES = os.path.join(PATH_TO_ROOT, "lib")

    # Used by:
    #  - main.py to direct execution to the proper executable in bin/
    EXECUTABLES = ["setup", "sandbox", "visualize"]
    EXE_ARG_POS = 1

