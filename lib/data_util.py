# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:58:14 2016

This is designed to be a streamlines and upgraded way to augment the initial
trajectory files with all features needed in a coherent fashion
@author: LordPhillips
"""

from lib import constants as c
from lib import frame_util as futil #for arrtostring
from lib import vehicleclass as v
from collections import defaultdict
from collections import Counter
import numpy as np
import math
import os

def getFeaturesLSTM(load_folder, testnum, intersections, q=False): #loadf is results/ByIntersection/testnum/
    LSTM_these = {}
    for intersection in intersections:
        LSTM_these[intersection] = np.load(
            os.path.join(load_folder, str(intersection), "LSTM_Formatted_featuresAndTargets.npy")
        )
        if not q: print("LSTM features loaded from inter", intersection)
    return convertToFeatures_lstm(LSTM_these, testnum, q)

def getFeaturesnonLSTM(load_folder, testnum, intersections, all=True, q=False):
    norm_test = {}
    for intersection in intersections:
        filepath = os.path.join(load_folder, str(intersection), "featuresAndTargets")
        norm_test[intersection] = np.loadtxt(filepath)
        if not q: print("NonLSTM features loaded from inter", intersection)
    return convertToFeatures_normal(norm_test, testnum, all, q)

def convertToFeatures_normal(norm_all, testnum, all=False, q=False): #input is dict of inter_id to features with nextmove, fid, vid appended
            #all is there because BayesNets test on all the features, does not remove
    dist_ind = 7
    if testnum[0] == "1":
        dist_ind = 11
    removed = 0
    targets = defaultdict(list)
    features = defaultdict(list)
    intersections = sorted(list(norm_all.keys()))
    for inter_id in intersections:
        for i in range(len(norm_all[inter_id])):
            if not all and norm_all[inter_id][i][dist_ind] == 0:
                removed += 1
                continue
            features[inter_id].append(norm_all[inter_id][i][:-3])
            targets[inter_id].append(int(norm_all[inter_id][i][-1])-1)
    X = np.ascontiguousarray(np.concatenate(([features[j] for j in intersections])))
    Y = np.ascontiguousarray(np.concatenate(([targets[j] for j in intersections])))
    if not q:
        print("removed:", removed, "from intersection.", len(Y), "remain")
        print(X.shape, Y.shape)
    return X, Y

#lstm_all of shape intersectionid => features
            #what features are is (numInputs, numFramesInTrajectory, numFeatures)
def convertToFeatures_lstm(lstm_all, testnum, q=False): #input is dict of inter_id to features with nextmove, fid, vid appended
    #ways to get rid of vehicles in intersection:
        #flatten by trajectory, if a trajectory has more than x in intersection, discard
            #remake into 3D matrix
          #not fully flattened, (trajLen, numFeatures) shape
    targets = defaultdict(list)
    features = defaultdict(list)
    dist_ind = 7
    if testnum[0] == "1":
        dist_ind = 11
    numTrajRemoved = 0
    numTrajKept = 0
    numTraj = len(lstm_all[list(lstm_all.keys())[0]][0])
    for inter_id in lstm_all.keys():
        for input_i in range(len(lstm_all[inter_id])):
            #lstm_all[inter_id][i] has desired shape
            #lstm_all[inter_id] is shape (x, traj_len, num_features + |fid, vid, nextmove|)
            invalid = False#0
            #nextMove = 0
            traj_moves = Counter([int(lstm_all[inter_id][input_i][i][-1]) for i in range(numTraj)])
            if len(traj_moves) > 1 or 0 in traj_moves:
                m, n = traj_moves.most_common(1)[0]
                if m == 0: 
                    print("most common was 0", traj_moves)
                    invalid = True
                for ii in range(numTraj):
                    lstm_all[inter_id][input_i][ii][-1] = m
            #for traj_i in range(numTraj): #also known as traj len
            #    if lstm_all[inter_id][input_i][traj_i][dist_ind] == 0:
            #        invalid += 1
            #    t = lstm_all[inter_id][input_i][traj_i][-1]
            #    if nextMove > 0 and t != nextMove:
            #        print("this shouldnt happen,", inter_id, nextMove, t)
            #    if nextMove == 0:
            #        nextMove = t
            if not invalid: #True:# made change for scoring, this is already done on feature creation
                #invalid <= numTraj/2 and nextMove > 0:
                numTrajKept += 1
                features[inter_id].append([lstm_all[inter_id][input_i][i][:-3] for i in range(numTraj)])
                targets[inter_id].append([int(lstm_all[inter_id][input_i][i][-1]-1) for i in range(numTraj)])
            else:
                numTrajRemoved += 1
    if not q:
        print("Num trajectories removed:", numTrajRemoved)
        print("Num trajectories kept:", numTrajKept)
    intersections = sorted(list(features.keys()))
    X = np.ascontiguousarray(np.concatenate(([features[j] for j in intersections])))
    Y = np.ascontiguousarray(np.concatenate(([targets[j] for j in intersections])))
    Y = Y.reshape((Y.shape[0], Y.shape[1], 1))
    if not q:
        print(X.shape, Y.shape)
    return X, Y

    
def convertFeaturesToTrainTestable(features, targets, dist_ind=8):        #distance index is 8, unless using lanetype, then 12
    intersections = sorted(list(features.keys()))
    X = np.ascontiguousarray(np.concatenate(([features[j] for j in intersections])))
    Y = np.ascontiguousarray(np.concatenate(([targets[j] for j in intersections])))
    ind_to_remove = []  
    for ind in range(len(X)):
        if X[ind][dist_ind] == 0:
            ind_to_remove.append(ind)
    Y = np.delete(Y, (ind_to_remove), axis=0)
    X = np.delete(X, (ind_to_remove), axis=0)
    print("removed:", len(ind_to_remove), "from intersection.", len(Y), "remain")
    return X, Y
        
        
def loadFeaturesAndTargets(filepath, load_folder, fid_folder=None, verbose=True):
    if not fid_folder:
        fid_folder = load_folder
    if not os.path.exists(load_folder):
        print("ERROR: folder does not exist:", load_folder)
        return
    allFeatures = {}
    allTargets = {}
    allFids = []
    secFids = {}
    examined_sfs = []
    for subdir, dirs, files in os.walk(load_folder):
        for sub_folder in dirs:
            if len(sub_folder) > 1 or sub_folder in examined_sfs:
                continue
            examined_sfs.append(sub_folder)
            i = int(sub_folder)
            fullpath = load_folder + sub_folder + '/'
            if 'LSTM' in load_folder:
                allFeatures[i] = np.load(fullpath + "featureSetv2.npy")
                allTargets[i] = np.load(fullpath + "targetSetv2.npy")
                secFids[i] = np.loadtxt(fid_folder[:-len("LSTM_formatted/")] + sub_folder + "/Fids")
            else:
                allFeatures[i] = np.loadtxt(fullpath + "featureSet")
                allTargets[i] = np.loadtxt(fullpath + "targetSet")
                secFids[i] = np.loadtxt(fid_folder + sub_folder + "/Fids")
            allFids.extend(secFids[i])
            if verbose:
              print("Features and Targets loaded from:", load_folder + sub_folder)
    return allFeatures, allTargets, len(allFids)

def get_norm_params_from_file(filepath, folder, cvs=True):
    allFeatures, _, _ = loadFeaturesAndTargets(filepath, folder, verbose=False)
    for i in sorted(list(allFeatures.keys())):
        ids = sorted(list(set(allFeatures.keys())))# - set([i])))
        Xtrain = np.ascontiguousarray(np.concatenate(([allFeatures[j] for j in ids])))
        mean, std = normalize_get_params(Xtrain)
        return mean, std  #just return one of them since norming all


#pass in numpy array, get mean, stddev for each column
def normalize_get_params(X):
    shape = X.shape
    numFeatures = shape[-1]
    means = np.zeros((numFeatures,))
    stddevs = np.zeros((numFeatures,))
    for i in range(numFeatures):
        if len(shape) == 3:
            a = X[:,:,i]
        elif len(shape) == 2:
            a = X[:,i]
        else:
            print("Unhandled shape:", shape)
            return [0], [1]
        means[i] = np.mean(a)
        stddevs[i] = np.std(a)
    return means, stddevs
            
def unnormalize(x, mean, stddev):
    return (x * stddev) + mean
    
def normalize(X, means, stddevs):
    for i in range(len(stddevs)):
        if stddevs[i] == 0.0:
            stddevs[i] = 1.0
    return (X-means)/stddevs

def normalize_wrapper(Xtrain, Xtest):
    means, stddevs = normalize_get_params(Xtrain)
    Xtrain_normed = normalize(Xtrain, means, stddevs)
    Xtest_normed = normalize(Xtest, means, stddevs)
    return Xtrain_normed, Xtest_normed

'''the big function for this'''
def augOrigData(filepath):
    #create framedict for file
    fd = makeFrameDictFromBase(filepath)
    
    keys = sorted(list(fd.keys()))
    print("Calculating Vx and Ax for all entries...")
    VxAxDict = calcVxAx(fd) #fid:{vid:[Vx,Ax]}
    print("Appending Vx and Ax...")
    #append Vx, Ax
    for fid in keys:
        for vid in fd[fid].keys():
            fd[fid][vid].extend(VxAxDict[fid][vid])
    print("Calculating and appending yaw...")
    #append Vx, Ax, orientation and calculate destination stuff
    numFids = len(keys)
    goal_to_cords = {} 
    #will be a dict that maps dest_num: (avg_x, avg_y), n
    #key = (dest, destLane)
    #d[key]=(x,y),n -- d[key][0]=(x,y) -- d[key][1] = n -- d[key][0][0] = x
    dest_lanes = {} #vid:destlane
    for fid in keys:
        if fid % int(numFids/10) == 0:
            print("Done", fid, "/", numFids, "frames...")
        curFrame = fd[fid]
        if fid + 1 in keys:
            nextFrame = fd[fid + 1]
        else:
            nextFrame = None
        for vid in curFrame.keys():
            curVehArr = curFrame[vid]
            if nextFrame and vid in nextFrame.keys():
                nextVehArr = nextFrame[vid]
                if fid - 1 in keys and vid in fd[fid-1].keys():
                    prevVehArr = fd[fid-1][vid]
                    orientation = calcOri(curVehArr, nextVehArr, prevVehArr)
                else:
                    dx = float(nextVehArr[c.LocalX]) - float(curVehArr[c.LocalX])
                    dy = float(nextVehArr[c.LocalY]) - float(curVehArr[c.LocalY])
                    orientation = math.atan2(dy,dx)
            else: # this means cur is the last time the vehicle appears
                # since appending as we go, can just get previous ones
                if fid - 1 in keys and vid in fd[fid-1].keys():
                    prevVehArr = fd[fid-1][vid]
                    orientation = float(prevVehArr[c.orientation])
                else:
                    orientation = 0
                dest, destLane, newGoal = goalCalcs(curVehArr, goal_to_cords)
                goal_to_cords[(dest, destLane)] = newGoal
                dest_lanes[vid] = destLane
            fd[fid][vid].append(float("{0:.4f}".format(float(orientation))))
    print("Appending destination info and lane types...")
    #append destLane, goalx, goaly
    #append LaneType
    laneTypes = c.lanetypes
    if "peach" in filepath:
        laneTypes = c.peachLaneTypes
    s = '-' #sep
    for fid in keys:
        if fid % int(numFids/10) == 0:
            print("Done", fid, "/", numFids, "frames...")
        for vid in fd[fid].keys():
            vehArr = fd[fid][vid]
            dest = int(vehArr[c.Dest])
            destLane = int(dest_lanes[vid])
            if not (dest, destLane) in goal_to_cords.keys():
                print(goal_to_cords.keys())
            destX = float("{0:.2f}".format(goal_to_cords[(dest, destLane)][0][0]))
            destY = float("{0:.2f}".format(goal_to_cords[(dest, destLane)][0][1]))
            appendArr = [destLane, destX, destY]
            lanetypeindex = str(vehArr[c.Section])+s+str(vehArr[c.Dir])+s+str(vehArr[c.LaneID])
            if lanetypeindex in laneTypes.keys():
                appendArr.append(int(laneTypes[lanetypeindex]))
            else:
                appendArr.append(0) # if not one of those anything can happen
            fd[fid][vid].extend(appendArr)
    #append nextMove -- vid-fid : move at next intersection
    #append distanceToIntersect -- vid-fid : distance to next intersection
    print("Finding distances and next moves...")
    nextMoves, distances = calcMovesDists(fd)
    print("Appending distances and next moves...")
    for fid in keys:
        if fid % int(numFids/10) == 0:
            print("Done", fid, "/", numFids, "frames...")
        for vid in fd[fid].keys():
            mv_dis_indx = str(vid)+s+str(fid)
            toAppend = [0,0]
            if mv_dis_indx in nextMoves.keys():
                toAppend[0] = int(nextMoves[mv_dis_indx])
            if mv_dis_indx in distances.keys():
                toAppend[1] = float("{0:.2f}".format((distances[mv_dis_indx])))
            fd[fid][vid].extend(toAppend)
    specialSave(filepath, fd)

''' Makes a special name that is distinct from the previous ones and saves'''    
def specialSave(filepath, frameDict):
    filename = os.path.basename(filepath)
    outpath = filepath[:-len(filename)]+'AUGv2_'+filename
    outFile = open(outpath, 'w')
    frames = list(frameDict.keys())
    frames.sort()
    for frame in frames:
        vids = list(frameDict[frame].keys())
        vids.sort()
        for vid in vids:
            arr = frameDict[frame][vid]
            outFile.write(futil.arrToStrForSave(arr))
    outFile.close()


'''make the frame dict without relying on anything else'''
def makeFrameDictFromBase(filepath):
    trajectoryFile = open(filepath, 'r')
    lines = trajectoryFile.readlines()
    numLines = len(lines)
    numBreak = int(numLines / 12)
    lineCounter = 0
    frameDict = {}
    for line in lines:
        if lineCounter % numBreak == 0:
            print("Processed line for dict: ", lineCounter, "/", numLines)
        array = line.split()
        vid = int(array[c.VehicleID])
        fid = int(array[c.FrameID])
        if fid in frameDict.keys():  
            #frameDict[fid] is a dict of vid:vehicledata
            frameDict[fid][vid] = array
        else:
            frameDict[fid] = {vid: array}
        lineCounter += 1
    return frameDict

def calcVxAx(fd):
    vxDict = calcVxForAll(fd)
    VxAxDict = calcAxForAll(vxDict)
    return VxAxDict
    
def calcVxForAll(fd):
    keys = sorted(list(fd.keys()))
    FidToVidToVx = {}
    for fid in keys:
        FidToVidToVx[fid] = {}
    numFids = len(keys)
    for fid in keys:
        if fid % int(numFids/10) == 0:
            print("Done", fid, "/", numFids, "frames in Vx calcs...")
        curFrame = fd[fid]
        if fid + 1 in keys:
            nextFrame = fd[fid + 1]
        else:
            nextFrame = None
        for vid in curFrame.keys():
            curVehArr = curFrame[vid]
            if nextFrame and vid in nextFrame.keys():
                Vx = calcVx(curVehArr, nextFrame[vid])
            else: # this means cur is the last time the vehicle appears
                # since appending as we go, can just get previous ones
                if fid - 1 in keys and vid in fd[fid-1].keys():
                    Vx = FidToVidToVx[fid-1][vid][0]
                else:
                    Vx = 0
            FidToVidToVx[fid][vid] = [float("{0:.4f}".format(Vx))]
    return FidToVidToVx

def calcAxForAll(vxDict):
    keys = sorted(list(vxDict.keys()))
    numFids = len(keys)
    for fid in keys:
        if fid % int(numFids/10) == 0:
            print("Done", fid, "/", numFids, "frames in Ax calcs...")
        curFrame = vxDict[fid]
        if fid + 1 in keys:
            nextFrame = vxDict[fid + 1]
        else:
            nextFrame = None
        for vid in curFrame.keys():
            curVx = curFrame[vid][0]
            if nextFrame and vid in nextFrame.keys():
                nextVx = nextFrame[vid][0]
                if type(nextVx) is list or type(curVx) is list:
                    print(vid, fid, curVx, nextVx)
                Ax = calcAx(curVx, nextVx)
            else: 
                if fid - 1 in keys and vid in vxDict[fid-1].keys():
                    Ax = vxDict[fid-1][vid][1]
                else:
                    Ax = 0
            vxDict[fid][vid].append(float("{0:.4f}".format(Ax)))
    return vxDict

def calcVx(curVehArr, nextVehArr):
    Vx = (float(nextVehArr[c.LocalX])-float(curVehArr[c.LocalX]))/c.t_frame
    return Vx

def calcAx(curVx, nextVx):
    Ax = (float(curVx) - float(nextVx))/c.t_frame
    return Ax
    
def calcOri(curVehArr, nextVehArr, prevVehArr):
    threshold = 0.5  #if the car moves a very small amount dont update
    max_diff = math.pi/4 #if there is too much rotation ignore.
    numToAverage = 3  #smooth orientation over the last 3 orientations
    prevOr = float(prevVehArr[c.orientation])
    dx = float(nextVehArr[c.LocalX]) - float(curVehArr[c.LocalX])
    dy = float(nextVehArr[c.LocalY]) - float(curVehArr[c.LocalY])
    yaw = math.atan2(dy,dx)
    if abs(dx) < threshold or abs(dy) < threshold:
        if abs(yaw - prevOr) > max_diff:
            yaw = prevOr
    prev_tot = prevOr * numToAverage
    new_avg = ((prev_tot - prevOr) + yaw) / numToAverage
    return new_avg

def goalCalcs(vehArr, goal_to_cords):
    dest = int(vehArr[c.Dest])
    destLane = int(vehArr[c.LaneID])
    x = float(vehArr[c.LocalX])
    y = float(vehArr[c.LocalY])
    if (dest, destLane) in goal_to_cords.keys():
        val = goal_to_cords[(dest,destLane)]
        prev_n = val[1]
        prev_avgX = val[0][0]
        prev_avgY = val[0][1]
    else:
        prev_n, prev_avgX, prev_avgY = [0]*3
    new_avgX = float((prev_avgX*prev_n) + x)/(prev_n+1)
    new_avgY = float((prev_avgY*prev_n) + y)/(prev_n+1)
    newGoal = (new_avgX, new_avgY), prev_n+1
    return dest, destLane, newGoal

def calcMovesDists(frameDict):
    nextMoves = {} #vid-fid : move at next intersection
    distances = {} #same, except distance to intersection
    vidfidstoupdate = {}
    s = '-' #this is sep
    sortedFrames = sorted(list(frameDict.keys()), reverse=False)
    framesDone = 0
    numFids = len(sortedFrames)
    for fid in sortedFrames:
        if fid % int(numFids/10) == 0:
            print("Done", fid, "/", numFids, "frames...")
        framesDone = framesDone + 1
        frame = frameDict[fid]
        for vid in frame.keys():
            veh = frame[vid]
            if int(veh[c.Section]) > 0:
                posX = float(veh[c.LocalX])
                posY = float(veh[c.LocalY])
                if vid in vidfidstoupdate.keys():
                    vidfidstoupdate[vid][fid]=[posX, posY]
                else:
                    vidfidstoupdate[vid] = {fid:[posX, posY]}
            else:
                move = int(veh[c.Movement])
                posX = float(veh[c.LocalX])
                posY = float(veh[c.LocalY])
                if vid in vidfidstoupdate.keys():
                    for fid2 in vidfidstoupdate[vid].keys():
                        indx = str(vid)+s+str(fid2)
                        nextMoves[indx] = move
                        dx = posX - vidfidstoupdate[vid][fid2][0]
                        dy = posY - vidfidstoupdate[vid][fid2][1]
                        distances[indx] = math.hypot(dx, dy)
                vidfidstoupdate[vid] = {}
                nextMoves[str(vid)+s+str(fid)] = move #dont forget this one...
                distances[str(vid)+s+str(fid)] = 0  # in intersection
    return nextMoves, distances

def getNeighbors(frame, veh):
    left_front_vid, left_back_vid, right_front_vid, right_back_vid = [0]*4
    if veh.getSection() == 0:
        return getXClosest(frame, veh, 4, list(frame.keys()))  #because in intersection
    lanesToMedian, lanesToCurb = veh.getLanesToSides()
    curLane = veh.getLane()
    leftLane = 0
    rightLane = 0
    if lanesToMedian > 0:
        if curLane == 1:
            if lanesToMedian == 1:
                leftLane = 11
            else:
                leftLane = 12
        elif curLane == 12:
            leftLane = 11
        elif curLane == 31:
            leftLane = 3
        else:
            leftLane = curLane - 1
    if lanesToCurb > 0:
        if curLane == 12:
            rightLane = 1
        elif curLane == 11:
            if veh.getSection() == 3:
                rightLane = 12
            else:
                rightLane = 1
        else:
            rightLane = curLane + 1
    leftVids = []
    rightVids = []
    for vid in frame.keys():
        veh2 = v.vehicle(frame[vid])
        if veh2.getSection() == veh.getSection():
            if leftLane > 0 and veh2.getLane() == leftLane:
                leftVids.append(vid)
            elif rightLane > 0 and veh2.getLane() == rightLane:
                rightVids.append(vid)
    left_front_vid, left_back_vid = getSide_FrontAndBack(frame, veh, leftVids)
    right_front_vid, right_back_vid = getSide_FrontAndBack(frame, veh, rightVids)
    return left_front_vid, left_back_vid, right_front_vid, right_back_vid

def getSide_FrontAndBack(frame, veh, sidevidlist):
    closest2 = getXClosest(frame, veh, 2, sidevidlist)
    vidlist = closest2
    if 0 in vidlist:
        return [0]*2
    elif -1 in vidlist:
        vid = [i for i in vidlist if not i == -1][0]
    else:
        vid = vidlist[0]
    veh2 = v.vehicle(frame[vid])
    if veh2.getPreceding() in set(vidlist) - set([vid]):
        back_vid = vid
        front_vid = veh2.getPreceding()
    else:
        back_vid = veh2.getPreceding()
        front_vid = vid
    return front_vid, back_vid
    
def getXClosest(frame, veh, X, fullvidlist):
    closest = {-i:99999-i for i in range(0,X)}
    for vid in fullvidlist:
        furthestClosestKey = find_furthest(closest)
        veh2 = v.vehicle(frame[vid])
        dist = math.sqrt(math.pow((veh.getX()-veh2.getX()),2) + math.pow((veh.getY()-veh2.getY()),2))
        if dist < closest[furthestClosestKey]:
            closest.pop(furthestClosestKey)
            closest[vid] = dist
    return list(closest.keys())

def find_furthest(vid_to_dist):
    maxd = 0
    maxv = 0
    for vid in vid_to_dist:
        if vid_to_dist[vid] > maxd:
            maxd = vid_to_dist[vid]
            maxv = vid
    return maxv
    
# run after getting other neighbors, pass list so that do not use them
# nextIntersection from frameDict
def getCarAtNextIntersection(frame, veh, nextIntersection, otherNeighborsVIDs):
    vids = frame.keys()
    interVIDs = []
    for vid in vids:
        if not vid in otherNeighborsVIDs:            
            if vid in nextIntersection:
                interVIDs.append(vid)
    if len(interVIDs) > 0:
        return (getXClosest(frame, veh, 1, interVIDs))[0]
    return 0


    
    
    
    
    
    
    
    
