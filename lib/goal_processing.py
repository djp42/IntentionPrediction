# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 14:22:06 2016

@author: LordPhillips
"""

from lib import constants as c
from lib import frame_util as futil
from lib import vehicleclass as v
from lib import data_util as dd
from lib import driver_util as dutil
import numpy as np
import re
import matplotlib.pyplot as plt
from IPython import display
import random
import seaborn as sns
import os

plotFolder = 'Plots/'

def loadFeatureGoals(featurepath):
    filepath = featurepath
    trajectoryFile = open(filepath, 'r')
    lines = trajectoryFile.readlines()
    goals = False
    goal_to_cords = {} 
    #will be a dict that maps dest_num: (avg_x, avg_y), n
    #key = (dest, destLane)
    #d[key]=(x,y),n -- d[key][0]=(x,y) -- d[key][1] = n -- d[key][0][0] = x
    goalNum = 0
    for line in lines:
        if 'GOALS TO CORDS' in line:
            goals = True
            continue
        if 'CONSTRAINTS' in line:
            goals = False
            continue
        array = list(filter(None, re.split("[, (\n)]",line)))
        if goals:
            dest = int(array[0])
            destLane = int(array[1])
            destX = float("{0:.2f}".format(float(array[2])))
            destY = float("{0:.2f}".format(float(array[3])))
            goal_to_cords[(dest,destLane)] = (destX,destY),goalNum-1
            goalNum = goalNum + 1
    return goal_to_cords

def getVid(array, data_object, wrong):
    if not wrong:
        return int(array[0])
    wrong_vid = int(array[0])
    fid = int(array[1])
    frame = data_object.getFrame(fid)
    sortedKeys = list(frame.keys())
    sortedKeys.sort()
    if wrong:
        real_vid = sortedKeys[wrong_vid]
    else:
        real_vid = wrong_vid
    return real_vid

def isGoalLine(array):
    return not "(" in array[2]

def readGoalAndVelFile(filepath, numGoals, data_object, wrong=True):
    file = open(filepath, 'r')
    lines = file.readlines()
    numLines = len(lines)
    lineCounter = 0
    vidToGoals = {}
    vidToVels = {}
    for line in lines:
        #first are velocities
        #next are goal distributions
        array = line.split()
        vid = getVid(array, data_object, wrong)
        fid = int(array[1])
        if lineCounter % 10000 == 0:
            print("Processed line for goal dict: ", lineCounter, "/", numLines)
        lineCounter = lineCounter + 1
        if isGoalLine(array):
            posteriors = [0] * numGoals
            for goal in range(0, numGoals):
                p = float(array[goal+2])
                posteriors[goal] = p
            if vid in vidToGoals.keys():  
                vidToGoals[vid][fid] = posteriors
            else:
                vidToGoals[vid] = {fid: posteriors}
        else:
            vels = [0] * (numGoals * 2)
            for i in range(0,numGoals*2):
                if i % 2 == 0:
                    vel = float(array[i+2][1:-1])  #goal lines go vel
                else:
                    vel = float(array[i+2][:-1])
                vels[i] = vel 
            if vid in vidToVels.keys():  
                vidToVels[vid][fid] = vels
            else:
                vidToVels[vid] = {fid: vels}
    return vidToGoals, vidToVels

def readGoalFile(filepath, numGoals, data_object, wrong=True):
    goalFile = open(filepath, 'r')
    lines = goalFile.readlines()
    numLines = len(lines)
    lineCounter = 0
    vidDict = {}
    #vid is key, inner is dict of fids to arrays of goal posteriors
    for line in lines:
        if lineCounter % 10000 == 0:
            print("Processed line for goal dict: ", lineCounter, "/", numLines)
        array = line.split()
        vid = getVid(array, data_object, wrong)
        fid = int(array[1])
        posteriors = [0] * numGoals
        for goal in range(0,numGoals):
            p = float(array[goal+2])
            posteriors[goal] = p
        if vid in vidDict.keys():  
            vidDict[vid][fid] = posteriors
        else:
            vidDict[vid] = {fid: posteriors}
        lineCounter += 1
    return vidDict    

def plotPostVsFrame(vehicleStuff, goal, fileout, stopAt = None):
    frames = list(vehicleStuff.keys())
    posteriors = {}
    for frame in frames:
        all_goals = vehicleStuff[frame]
        this_posterior = all_goals[goal]
        posteriors[frame] = this_posterior
    #display.clear_output(wait=False)
    plt.figure()
    sns.set(style="darkgrid", font='Source Sans Pro')
    sns.set_context("talk")
    x = [i for i in range(0,len(posteriors.values()))]
    y = list(posteriors.values())
    if stopAt:
        x = x[0:stopAt]
        y = y[0:stopAt]
    plt.plot(x, y)
    plt.savefig(plotFolder+'PostVsTme'+fileout)
    
def plotForOneVIDFrame(vehicleStuff, fileout, frame=-1):
    if frame == -1:
        frame = random.choice(list(vehicleStuff.keys()))
    posteriors = vehicleStuff[frame]
    print(posteriors)
    plt.figure()
    sns.set(style="dark", font='Source Sans Pro')
    sns.set_context("talk")
    sns.barplot([i for i in range(0,len(posteriors))], posteriors)
    sns.plt.savefig(plotFolder+'GoalPosts'+fileout)
    
'''line1 = vid fid
   line2 - line n-1 = goal goal ... goalvid fid
       goalvid = goal[12345678...] 8... = vid of variable length
   line n = goal goal ... goal

'''
def fixFormat(wrongFormatFilePath, outpath, numGoals, data_object):
    wrongFile = open(wrongFormatFilePath, 'r')
    rightOutFile = open(outpath, 'w')
    lines = wrongFile.readlines()
    numLines = len(lines)
    lineCounter = 0
    #vid is key, inner is dict of fids to arrays of goal posteriors
    vidfid = lines[0].split()
    for line in lines[1:]:
        if lineCounter % 10000 == 0:
            print("Fixed line for goal dict: ", lineCounter, "/", numLines)
        fake_vid = int(vidfid[0])
        fid = int(vidfid[1])
        frame = data_object.getFrame(fid)
        sortedKeys = list(frame.keys())
        sortedKeys.sort()
        real_vid = sortedKeys[fake_vid]
        nextVidLen = len(str(fake_vid))  
        if fake_vid in [9,99,999]:
            nextVidLen = nextVidLen + 1
        if fake_vid == len(sortedKeys)-1:#data_object.vidIsLastInFrame(vid, fid):
            nextVid = 0
            nextVidLen = len(str(nextVid))
        writeThings = [real_vid,fid]
        arrayWrong = line.split()
        leng = (len(arrayWrong))
        last = leng - 2
        for goal in range(0,last):
            p = float(arrayWrong[goal])
            writeThings.append(p)
        desLen = len(arrayWrong[last]) - nextVidLen
        lastPost = float(arrayWrong[last][0:desLen])
        if lineCounter < numLines - 2:
            vidfid[0] = int(arrayWrong[last][desLen:len(arrayWrong[last])])
            vidfid[1] = int(arrayWrong[last+1])
        writeThings.append(lastPost)
        rightOutFile.write(futil.arrToStrForSave(writeThings))
        lineCounter += 1

def printGoals(goal_to_cords_dict):
    print("Goal coordinates")
    keys = list(goal_to_cords_dict.keys())
    keys.sort() #to make things nicer
    for d in keys:
        print("Destination (section, lane) =", d, "Coords:",goal_to_cords_dict[d][0])

def processJustGoals(vid, fid, driver_data, vidDict, goal_dict, stopAt):
    print("doing for vid:", vid," and fid:",fid)
    dest, destLane = driver_data.getDest(fid,vid)
    goal = goal_dict[(dest, destLane)][1]
    fileout = 'vidOf'+str(vid)+'_fidOf'+str(fid)+'_goal'+str(goal)+'.png'
    vehicleStuff = vidDict[vid]
    plotForOneVIDFrame(vehicleStuff, fileout, fid)
    print(goal)
    plotPostVsFrame(vehicleStuff, goal, fileout, stopAt)
    fileout = 'vidOf'+str(vid)+'_fidOf'+str(fid)+'_goal'+str(9)+'.png'
    plotPostVsFrame(vehicleStuff, 9, fileout, stopAt)

def findMaxPosterior(goalList):
    maxPost = 0
    index = -1
    for i in range (0,len(goalList)):
        if goalList[i] > maxPost:
            index = i
            maxPost = goalList[i]
    return index

def processVelsAndGoals(vid, fid, driver_data, vidDict, goal_dict, stopAt, vidToVels):
    predGoal = findMaxPosterior(vidDict[vid][fid])
    predVelX = vidToVels[vid][fid][predGoal * 2]
    predVelY = vidToVels[vid][fid][(predGoal * 2) + 1]
    veh = driver_data.getVehicle(vid,fid)
    actualVelX = veh.getVx()
    actualVelY = veh.getVy()
    #if fid % 100 == 0:
    #    print("for vid", vid, "and fid:", fid, "predicted goal:", predGoal)
    #    print("predicted velocity: (",predVelX,",", predVelY, "), actual velocity: (", actualVelX, ",", actualVelY, ")")
    return actualVelX, actualVelY, predVelX, predVelY
    '''dest, destLane = driver_data.getDest(fid,vid)
    goal = goal_dict[(dest, destLane)][1]
    fileout = 'vidOf'+str(vid)+'_fidOf'+str(fid)+'_goal'+str(goal)+'.png'
    vehicleStuff = vidDict[vid]
    plotForOneVIDFrame(vehicleStuff, fileout, fid)
    print(goal)
    plotPostVsFrame(vehicleStuff, goal, fileout, stopAt)
    fileout = 'vidOf'+str(vid)+'_fidOf'+str(fid)+'_goal'+str(9)+'.png'
    plotPostVsFrame(vehicleStuff, 9, fileout, stopAt)'''

def doGoalStuff(filepath, isCompressed, goalFile, fake=True, vels=False):
    print(filepath)
    filename = os.path.basename(filepath)
    featurepath = dutil.findFeaturePath(filename)
    print("constructing driver data...")
    driver_data = dd.data(filepath, isCompressed, featurepath)
    print("constructing goal dict...")
    goal_dict = loadFeatureGoals(featurepath)
    #will be a dict that maps dest_num: (avg_x, avg_y), goalNum
    #key = (dest, destLane)
    #d[key]=(x,y),n -- d[key][0]=(x,y) -- d[key][1] = goalNums
    numGoals = len(list(goal_dict.keys())) - 1
    print("constructing vidDIct")
    if 'format_wrong' in goalFile:
        wrongPath = dutil.findPathForFile(goalFile)
        outfile = goalFile[0:(len(goalFile)-len('_format_wrong.txt'))]+'.txt'
        outpath = dutil.findPathForFile(outfile)
        print("fixing format")
        fixFormat(wrongPath, outpath, numGoals, driver_data)
        if not vels:
            vidDict = readGoalFile(outpath, numGoals, driver_data, False)
        else:
            vidDict, vidToVels = readGoalAndVelFile(outpath, numGoals, driver_data, False)
    else:
        if not vels:            
            vidDict = readGoalFile(dutil.findPathForFile(goalFile), numGoals, driver_data, fake)
        else:
            vidDict, vidToVels = readGoalAndVelFile(dutil.findPathForFile(goalFile), numGoals, driver_data, fake)

    #pick a random vehicle
    #vid = driver_data.getRandVid()
    #while not vid in vidDict.keys():
    #    vid = driver_data.getRandVid()
    stopAt = 900
    vids = [2, 5, 9, 14, 24]
    '''stopAt = 899
    print(vid)
    fidFirst = list(vidDict[vid].keys())[0]
    fidLast = 925#list(vidDict[vid].keys())[len(list(vidDict[vid].keys()))-1]
    print(fidFirst, fidLast)
    fids = [fidLast]
    vidsToFids = {vid:fids}
    for vid in vidsToFids.keys():'''
    #for vid in vids: 
    for vid in vidDict.keys():
        if not driver_data.hasVID(vid):
            continue
        stopAt = min(driver_data.lastFrameForVID(vid), list(vidDict[vid].keys())[-1])
        frames = []
        ActX = []
        ActY = []
        PredX = []
        PredY = []
        #for fid in vidsToFids[vid]:
        for fid in vidDict[vid].keys():
            if not fid in (driver_data.getFrameDict()).keys():
                continue
            if vid not in (driver_data.getFrameDict())[fid]:
                continue
            if not vels:    
                processJustGoals(vid, fid, driver_data, vidDict, goal_dict, stopAt)
            else:
                aX, aY, pX, pY = processVelsAndGoals(vid, fid, driver_data, vidDict, goal_dict, stopAt, vidToVels)
                ActX.append(aX)
                ActY.append(aY)
                PredX.append(pX)
                PredY.append(pY)
                frames.append(fid)
        plt.figure()
        sns.set(style="darkgrid", font='Source Sans Pro')
        sns.set_context("talk")
        x = frames
        XAline, = plt.plot(x, ActX, label='Actual X Vel')
        YAline, = plt.plot(x, ActY, label='Actual Y Vel')
        XPline, = plt.plot(x, PredX, '--', label='Pred X Vel')
        YPline, = plt.plot(x, PredY, '--', label='Pred Y Vel')
        title = "Pred and Actual Vel Comparison for Vehicle " + str(vid)
        plt.title(title)
        plt.legend(loc=0)
        fileout = 'TrajComparisonVidX&Y'+str(vid)+'.png'
        plt.savefig(plotFolder+fileout)


                

