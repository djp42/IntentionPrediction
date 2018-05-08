##
# File: /lib/frame_util.py
# ------------------
# Commonly used functions to load frames/dta
##

import collections, itertools, copy
from copy import deepcopy
import scipy, math, random
import numpy as np
import os, sys, time, importlib
import tokenize, re, string
import json, unicodedata
from lib import constants as c
#import matplotlib.pyplot as plt
#from IPython import display
from lib import vehicleclass as v
from lib import data_class as dd

FRAME_TIME = 0.1

# This has heuristics to speed up coding, also can only be done with these 
# two files anyway...
def combineTrajFiles(filepath1, filepath2, overlapStartFrame=10201, maxVid1=1438):
    newFilename = 'trajectories-0830am-0900am.txt'
    filename1 = os.path.basename(filepath1)
    outpath = filepath1[:-len(filename1)]+newFilename
    infile2 = open(filepath2, 'r')
    with open(outpath, 'w') as outFile:
        with open(filepath1) as infile1:
            outFile.write(infile1.read())
        with open(filepath2) as infile2:
            for line in infile2:
                arr = line.split()
                newline = arr
                newline[c.VehicleID] = str(int(arr[c.VehicleID]) + maxVid1)
                newline[c.FrameID] = str(int(arr[c.FrameID]) + overlapStartFrame)
                outFile.write(arrToStrForSave(newline))
    return outpath 

def combineTrajFilesNoOverlap(filepath1, filepath2, skipFront=500, skipEnd=1000):
    newFilename = 'trajectories-peachtree.txt'
    filename1 = os.path.basename(filepath1)
    outpath = filepath1[:-len(filename1)]+newFilename
    #infile2 = open(filepath2, 'r')
    maxVid1 = 0
    maxFid1 = max(int((line.split())[c.FrameID]) for line in open(filepath1))
    with open(outpath, 'w') as outFile:
        with open(filepath1) as infile1:
            for line in infile1:
                arr = line.split()
                if int(arr[c.FrameID]) > maxFid1: break
                outFile.write(line)
                if int(arr[c.VehicleID]) > maxVid1:
                    maxVid1 = int(arr[c.VehicleID])
        with open(filepath2) as infile2:
            for line in infile2:
                arr = line.split()
                if int(arr[c.FrameID]) <= skipFront: continue
                newline = arr
                newline[c.VehicleID] = str(int(arr[c.VehicleID]) + maxVid1)
                newline[c.FrameID] = str(int(arr[c.FrameID]) + maxFid1)
                outFile.write(arrToStrForSave(newline))
    return outpath 
 

def arrToStrForSave(arr):
    writeArray = [str(item) for item in arr]
    writeString = ' '.join(writeArray) + "\n"
    return writeString

def saveJuliaFD(filepath, frameDict):
    filename = os.path.basename(filepath)
    outpath = filepath[:-len(filename)]+'JULIA_'+filename
    outFile = open(outpath, 'w')
    frames = list(frameDict.keys())
    frames.sort()
    for frame in frames:
        vids = list(frameDict[frame].keys())
        vids.sort()
        for vid in vids:
            arr = frameDict[frame][vid]
            outFile.write(arrToStrForSave(arr))
    outFile.close()

def saveFrameDict(filepath, frameDict, compressed):
    filename = os.path.basename(filepath)
    if compressed:
        outpath = filepath[:-len(filename)]+'driv_compr_'+filename
    else:
        outpath = filepath
    outFile = open(outpath, 'w')
    frames = list(frameDict.keys())
    frames.sort()
    for frame in frames:
        vids = list(frameDict[frame].keys())
        vids.sort()
        for vid in vids:
            veh = v.vehicle(frameDict[frame][vid], compressed)
            '''if not len(frameDict[frame][vid]) == 30:
                print(len(frameDict[frame][vid]))                
            if abs(veh.getOrientation()) > math.pi:
                print("Problem, yaw too large.")
                print(vid)
                print(veh.getOrientation())'''
            arr = veh.returnArrayInfo()
            #if len(arr) < 21:
            #    print("Issue with array length in saving: ", len(arr))
            outFile.write(arrToStrForSave(arr))
    outFile.close()

'''Add directional velocity/acceleration to base trajectories'''
def augment(filepath):
    filename = os.path.basename(filepath)
    outpath = filepath[:-len(filename)]+'aug_'+filename
    AddVxAx(filepath, outpath)
    return outpath

'''Compress a file that has already been augmented'''
def compress(aug_filepath):
    filename = os.path.basename(aug_filepath)
    trajectoryFile = open(aug_filepath, 'r')
    outpath = aug_filepath[:-len(filename)]+'compr_'+filename
    outFile = open(outpath, 'w')
    lines = trajectoryFile.readlines()
    numLines = len(lines)
    lineCounter = 0
    for line in lines:
        if lineCounter % 30000 == 0:
            print("Read line ", lineCounter, "/", numLines)
        augArray = line.split()
        veh = v.vehicle(augArray, False)
        compAugArr = veh.returnCompressedArray()  
        writeArray = [str(item) for item in compAugArr]
        writeString = ' '.join(writeArray) + "\n"
        outFile.write(writeString)
        lineCounter += 1
    trajectoryFile.close()
    outFile.close()
    return

def processBaseFile(filepath):
    aug_path = augment(filepath)
    compress(aug_path)
    return
  
def AddVxAx2(inFilename,outFilename):
    trajectoryFile = open(inFilename, 'r')
    outFile = open(outFilename, 'w')
    lines = trajectoryFile.readlines()
    numLines = len(lines)
    lineCounter = 0
    prevLine = []
    for line in lines:
        if lineCounter % 30000 == 0:
            print("Read line ", lineCounter, "/", numLines)
        if lineCounter == 0:
            prevLine = line
            continue
        curArray = line.split()
        curVID = int(curArray[c.VehicleID])
        curFrame = int(curArray[c.FrameID])
        curArray.append(Vx)
        curArray.append(Ax)
        lineCounter += 1

"""
Function: AddVxAx:
Essentially used once to initialize the Vy and Ay values

"""
def AddVxAx(inFilename, outFilename):
    trajectoryFile = open(inFilename, 'r')
    outFile = open(outFilename, 'w')
    lineNum = 0
    lastVID = 0
    frameCounter = 0 #number of frames the car is in?
    vidDict = {}
    lines = trajectoryFile.readlines()
    numLines = len(lines)
    lineCounter = 0
    for line in lines:
        if lineCounter % 30000 == 0:
            print("Read line ", lineCounter, "/", numLines)
        curArray = line.split()
        curVID = int(curArray[c.VehicleID])
        curFrame = int(curArray[c.FrameID])
        Vx = 0
        Ax = 0
        if lastVID != curVID:
            frameCounter = 0
            vidDict[curVID] = {}
        if frameCounter > 0:
            curY = float(curArray[c.LocalX])
            preVx = float(vidDict[curVID][curFrame - 1][c.LocalX])
            Vx = float(curY - preVx)/FRAME_TIME
        if frameCounter > 1:
            curVx = float(Vx)
            prevVx = float(vidDict[curVID][curFrame - 1][c.augVx])
            Ax = float(curVx - float(prevVx))/float(FRAME_TIME)
        curArray.append(Vx)
        curArray.append(Ax)
        vidDict[curVID][curFrame] = curArray
        writeArray = [str(item) for item in curArray]
        writeString = ' '.join(writeArray) + "\n"
        outFile.write(writeString)
        frameCounter += 1
        lastVID = curVID
        lineCounter += 1
    trajectoryFile.close()
    outFile.close()
    return vidDict

"""
Function: VIDToFrameDicts:
Converts a dictionary based on VID keys to a dictionary based
on FrameID keys.

"""

def VIDToFrameDicts(vidDict):
    frameDict = {}
    vidDictLen = len(vidDict)
    counter = 0
    print(vidDictLen, " entries to convert")
    for elem in vidDict:
        elem = int(elem)
        for vid in vidDict[elem]:
            entry = vidDict[elem][vid]
            curVID = int(entry[0])
            frameID = int(entry[1])
            if frameID not in frameDict:
                frameDict[frameID] = {}
            frameDict[frameID][curVID] = deepcopy(entry)
        if counter % 200 == 0:
            print("Processing.... ", counter, " / ", vidDictLen)
        counter += 1
    return frameDict

"""
Function: LoadDictFromTxt
Params: filename, dictType

Takes a full-path filename of an entry file and loads the info into memory
as a dictionary.  The dictType param determines whether the returned dictionary
is keyed by frameID or VID
"""

def LoadDictFromTxt(filename, dictType):
    trajectoryFile = open(filename, 'r')
    outDict = {}
    for line in trajectoryFile.readlines():
        curArray = np.array(line.split()).astype(float)
        curVID = int(curArray[0])
        curFrame = int(curArray[1])
        if dictType == 'vid':
            if curVID not in outDict:
                outDict[curVID] = {}
            outDict[curVID][curFrame] = curArray
        if dictType == 'frame':
            if curFrame not in outDict:
                outDict[curFrame] = {}
            outDict[curFrame][curVID] = curArray
            
    trajectoryFile.close()
    return outDict

"""
Function: GetGridsFromFrameDict

Takes a dictionary based on frameIDs, converts to a dictionary
of grids based on frameID.  Calls FrameToGrid, which does most
of the work

"""

#each vehicle has its full entry in the dict.
def GetGridsFromFrameDict(frameDict, mean_centered, compressed=False):
    gridDict = {}
    counter = 0
    for i in frameDict:
        frame = frameDict[i]
        grid = FrameToGrid(frame, mean_centered, compressed)
        gridDict[i] = deepcopy(grid)
        counter += 1
        if counter % 500 == 0:
            print("Processed ", counter, " frames.")
    return gridDict

"""
Function: GetGridIndices

Takes in an x and a y, and returns the indies in the currently dimensioned
grid.

"""

def GetGridIndices(givenX, givenY):
    gridX = int((givenX - c.MIN_GRID_X) / c.X_STEP)
    gridY = int((givenY - c.MIN_GRID_Y) / c.Y_STEP)
    return gridX, gridY


"""
Function: getGridMeans

Returns the means of each column vector at each point in the grid, over
the number of vehicles.

"""
def getGridMeans(grid):
    sum1 = np.sum(grid,1)
    sum2 = np.sum(sum1, 0)
    numVehicles = sum2[0]
    means = sum2 / numVehicles
    if numVehicles == 0:
        print(numVehicles)
        print(sum2)
        print(sum1)
        return [0]*len(means)
    return means


"""
Function: MeanCenterGrid

Takes in a grid, and subtracts the mean of all values besides #vehicles.

"""

def MeanCenterGrid(grid):
    x, y, z = grid.shape
    means = getGridMeans(grid)
    means[0] = 0
    for i in range(x):
        for j in range(y):
            if grid[i][j][0] != 0:
                grid[i][j] -= means

    # means = np.zeros(z)
    # for i in range(x):
    #     for j in range(y):
    #         numVehicles = grid[i][j][0]
    #         scaledVals = grid[i][j]*numVehicles
    #         scaledVals[0] = numVehicles
    #         means += scaledVals
    # means /= means[0]
    # means[0] = 0
    # for i in range(x):
    #     for j in range(y):
    #         if grid[i][j][0] != 0:
    #             grid[i][j] -= means
    return

    

"""
Function: FrameToGrid

Converts info from a frame (dict of VIDs for all cars in a particular
frame) into a grid populated with entries if a vehicle is present at
that location.

"""

def FrameToGrid(frame, mean_centered, compressed=False):
    #Creates grid determined by DIV numbers in constants.py
    started = False    
    grid = np.zeros((c.X_DIV +1, c.Y_DIV +1, 3))#len(dummyVehicle.GridInfo)) # is number of elems in trajectory info
    for vid in frame:
        vehicleData = frame[vid]
        veh = v.vehicle(vehicleData, compressed)
        if not started:
            grid = np.zeros((c.X_DIV +1, c.Y_DIV +1 , len(veh.GridInfo)))
            started = True
        if not InGridBounds(veh.getX(), veh.getY()):
            continue
        # Scales the grid into the desired window - check constants.py
        # to edit MIN/MAX_GRID values.
        gridX, gridY = GetGridIndices(veh.getX(), veh.getY())
        grid[gridX][gridY] += veh.getGridInfo()
        if mean_centered == 1:
            MeanCenterGrid(grid)
    return grid

"""
Function: InGridBounds

Checks if a given x and y are within the constant bounds
of the desired grid.  Returns True if true, False if false.

"""

def InGridBounds(givenX, givenY):
    if givenX < c.MIN_GRID_X or givenX > c.MAX_GRID_X:
        return False
    if givenY < c.MIN_GRID_Y or givenY > c.MAX_GRID_Y:
        return False
    return True

"""
Function: GetGridPoints

This is basically just used for animation, but tests a given grid
for nonzero indices, then calculates their positions in the original
frame (does not decompress), for display purposes

"""

def GetGridPoints(grid):
    gflat = np.sum(grid, axis=2)
    nz = np.nonzero(gflat)
    nzx = nz[0]*c.X_STEP
    nzy = nz[1]*c.Y_STEP
    return nzx, nzy


def AnimateData(data_object):
    inputDict = data_object.getFrameDict()    
    fig_size = plt.rcParams["figure.figsize"]
    
    print("Current size:", fig_size)
     
    # Set figure width to 12 and height to 9
    fig_size[0] = 3
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.figure(1)
    portionToDisplay = 1
    for frameid in inputDict.keys():
        if not frameid % portionToDisplay == 0: 
            continue
        curFrame = inputDict[frameid]
        plotFrame(curFrame, frameid, data_object.getCompressed())
        plt.clf()
"""
Function: Animate Frames

Given a dictionary and an input type, animates the dictionary in time order

"""

def AnimateFrames(inputDict, inputType='frame'):
    #With a loaded frameDict, animates frames.
    fig_size = plt.rcParams["figure.figsize"]
     
    print("Current size:", fig_size)
     
    # Set figure width to 12 and height to 9
    fig_size[0] = 3
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.figure(1)
    numFrames = len(inputDict.keys())
    for frameid in range(int(numFrames/5)):
        curFrame = inputDict[str(frameid*5)] #i think the 100+ gets rid of boring first ones
        if inputType == 'frame':
            plotFrame(curFrame, frameid)
        if inputType == 'grid':
            plotGrid(curFrame, frameid)
        plt.clf()

"""
Function: plotFrame

Helper to plot a given frame

"""
def plotFrame(curFrame, fid, signals=None, compressed=False):
    cars_x = []
    cars_y = []
    #bikes_x = []
    #bikes_y = []
    #trucks_x = []
    #trucks_y = []    
    for vid in curFrame.keys():
        veh = v.vehicle(curFrame[vid],compressed)
        x=float(veh.getX())
        y=float(veh.getY())
        '''if veh.getClass() == '2':
            bikes_x.append(x)
            bikes_y.append(y)
        elif veh.getClass() == '3':
            cars_x.append(x)
            cars_y.append(y)
        else:
            trucks_x.append(x)
            trucks_y.append(y)'''
        cars_x.append(x)
        cars_y.append(y)
    #x,y = getFramePoints(curFrame)        
    #plt.plot(bikes_x,bikes_y, 'ko')
    plt.plot(cars_x,cars_y, 'bo')
    #plt.plot(trucks_x,trucks_y, 'bo')
    plt.title("t = " + str(fid))
    plt.axis([-100, 100, -250, 1750])
    if signals:
        plt.plot(signals['red']['x'], signals['red']['y'], 'ko')
        plt.plot(signals['green']['x'], signals['green']['y'], 'go')
    #display.clear_output(wait=False)
    display.display(plt.gcf())

"""
Function: plotGrid

Helper to plot a given grid

"""

def plotGrid(curGrid, fid):
    gflat = np.sum(curGrid, axis=2)
    nz = np.nonzero(gflat)
    nzx = nz[0]*c.X_STEP
    nzy = nz[1]*c.Y_STEP
    plt.title("t = " + str(fid))
    plt.axis([0, 2250, -70, 0])
    plt.plot(nzy, -nzx, 'ro')
    display.clear_output(wait=True)
    display.display(plt.gcf())

"""
Function: getFramePoints

Plots a given 

"""
def getFramePoints(curFrame):
    x = np.array([0]*len(curFrame))
    y = np.array([0]*len(curFrame))
    entryCounter = 0
    for entry in curFrame:
        x[entryCounter] = float(curFrame[entry][4])
        y[entryCounter] = float(curFrame[entry][5])
        entryCounter += 1
    return x,y

