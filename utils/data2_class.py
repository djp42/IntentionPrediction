# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:51:26 2016

This is similar to data_class except it is stripped of depricated functions 
after data_util was made and has function designed for the machine learning
@author: LordPhillips
"""

from utils import constants as c
from utils import frame_util as futil
from utils import data_util as dutil
from utils import vehicleclass as v
import numpy as np
import os
import math
import time
import re
import random

class data2:
    '''to be used on file that has already been through the rounds with data_class
    The other option is to use data_util.augOrigData on the base file and use this
    class directly with that output (filename will be AUGv2_...)'''
    def __init__(self, filepath):
        self.nextIntersections = {} #[vid][fid]=> intersection id
        self.maxFid = 0
        self.frameDict = self.makeFrameDict(filepath)
        self.filepath = filepath
        self.frames = self.getFrames()
        self.numFrames = len(self.frames)
    
    '''FrameDict will have entries where every key is a dict of all vehicles
    that are present in that frame framedict = s{frameid: {vid:vehicledata}}'''
    def makeFrameDict(self, filepath):
        trajectoryFile = open(filepath, 'r')
        lines = trajectoryFile.readlines()
        numLines = len(lines)
        lineCounter = 0
        frameDict = {}
        for line in lines:
            if lineCounter % 30000 == 0:
                print("Processed line for dict: ", lineCounter, "/", numLines)
            array = line.split()
            veh = v.vehicle(array)
            fid = veh.getFrame()
            if fid in frameDict.keys():  
                #frameDict[fid] is a dict of vid:vehicledata
                frameDict[fid][veh.getVID()] = array
            else:
                frameDict[fid] = {veh.getVID(): array}
            if fid > self.maxFid:
                self.maxFid = fid
            lineCounter += 1
        return frameDict
    
    #maxfid and n dont matter, havent removed from all calls yet 
    def getNextIntersectionAndRecord(self, veh, fids, maxfid, n):
        vid = veh.getVID()
        fid = veh.getFrame()
        n = len(fids)
        if vid in self.nextIntersections:
            if fid in self.nextIntersections[vid]:
                return self.nextIntersections[vid][fid]
        else:
            self.nextIntersections[vid] = {}
        for nextfid_index in range(fids.index(fid), n): #starting at current fid, iterate over all frames vehicle appears in
            nextfid = fids[nextfid_index]
            if vid not in self.frameDict[nextfid]: 
                print("vid is not in a frame it should be:", vid, fid)
                continue
            veh = v.vehicle(self.frameDict[nextfid][vid])
            if veh.getIntersection() > 0: #if in intersection, find last in intersection
                lastFid_ind_InIntersection = n-1 #if we don't find, it is last index
                for curFrame_ind in range(nextfid_index, n):  #find last frame where vid is in intersection
                    curFrame = fids[curFrame_ind]
                    otherveh = v.vehicle(self.frameDict[curFrame][vid])
                    if otherveh.getIntersection() == 0:
                        lastFid_ind_InIntersection = curFrame_ind - 1
                        break
                intersect = veh.getIntersection()
                for interfid_ind in range(fids.index(fid), lastFid_ind_InIntersection): #from current fid through intersection, fill it in
                    self.nextIntersections[vid][fids[interfid_ind]] = intersect
                return intersect
        return 0
        
    #includes skipping vehicles with next move of 0
    #fids are the reducedFids
    #features[-1] is nextMove
    def getFeatureVectors_intersection(self, fids, intersectionID, laneType=False, useHistory=False, historyFrames=[],
                                       useTraffic=False, bool_peach=False):
        featVecs = np.ascontiguousarray([])
        numinstances = len(fids)
        numDone = 0
        maxfid = max(fids)
        n = numinstances
        for fid in fids:
            frame = self.frameDict[fid]
            numDone = numDone + 1
            if numDone % int(numinstances/10) == 0:
                print("Done making features for", numDone, "/", numinstances, "instances...")
            for vid in sorted(list(frame.keys())):
                veh = v.vehicle(frame[vid])
                nextMove = veh.getNextMove()
                if nextMove == 0: continue
                if self.getNextIntersectionAndRecord(veh, fids, maxfid, n) != intersectionID: continue #absolutely the slowest way to do this, but irrelevant in grand scheme
                fullFeatures = []
                if laneType:
                    fullFeatures.extend(veh.getTrainFeatures1(bool_peach))
                else:
                    fullFeatures.extend(veh.getTrainFeatures2())
                if useHistory:
                    fullFeatures.extend(self.getHistory([], fid, vid, historyFrames))
                if useTraffic:
                    fullFeatures.extend(self.getTrafficFeatures(fid,vid))
                fullFeatures.append(nextMove)
                if featVecs.size == 0:
                    featVecs = np.ascontiguousarray(fullFeatures)
                else:
                    featVecs = np.vstack((featVecs,fullFeatures))
        return np.ascontiguousarray(featVecs)
        
        
    def getAllFeatureVectors_intersection(self, fids, laneType=False, useHistory=False, historyFrames=[],
                                       useTraffic=False, bool_peach=False, ids = [1,2,3,4]):
        features = {}
        for interid in ids:
            features[interid] = np.ascontiguousarray([])
        numinstances = len(fids)
        numDone = 0
        maxfid = max(fids)
        n = numinstances
        for fid in fids:
            frame = self.frameDict[fid]
            numDone += 1
            #if numDone % int(numinstances/10) == 0:
            print("Done making features for", numDone, "/", numinstances, "instances...")
            for vid in sorted(list(frame.keys())):
                veh = v.vehicle(frame[vid])
                nextMove = veh.getNextMove()
                if nextMove == 0: continue
                interid = self.getNextIntersectionAndRecord(veh, fids, maxfid, n)
                if interid == 0: continue 
                fullFeatures = []
                if laneType:
                    fullFeatures.extend(veh.getTrainFeatures1(bool_peach))
                else:
                    fullFeatures.extend(veh.getTrainFeatures2())
                if useHistory:
                    fullFeatures.extend(self.getHistory([], fid, vid, historyFrames))
                if useTraffic:
                    fullFeatures.extend(self.getTrafficFeatures(fid,vid))
                fullFeatures.append(nextMove)
                if features[interid].size == 0:
                    features[interid] = np.ascontiguousarray(fullFeatures)
                else:
                    features[interid] = np.vstack((features[interid],fullFeatures))
        for interid in features:
            features[interid] = np.ascontiguousarray(features[interid])
        return features
    
    def getFeatureVectors(self, fids, laneType=True, useHistory=False, historyFrames=[], 
                          useTraffic=True, bool_peach=False):
        featVecs = np.ascontiguousarray([])
        fids = sorted(list(fids))
        numFids = len(fids)
        numDone = 0
        for fid in fids:
            fid = int(fid)
            numDone = numDone + 1
            if numDone % int(numFids/10) == 0:
                print("Done making features for", numDone, "/", numFids, "frames...")
            frame = self.frameDict[fid]
            for vid in sorted(list(frame.keys())):
                veh = v.vehicle(frame[vid])
                fullFeatures = []
                if laneType:
                    fullFeatures.extend(veh.getTrainFeatures1(bool_peach))
                else:
                    fullFeatures.extend(veh.getTrainFeatures2())
                if useHistory:
                    fullFeatures.extend(self.getHistory(fids, fid, vid, historyFrames))
                if useTraffic:
                    fullFeatures.extend(self.getTrafficFeatures(fid,vid))
                if featVecs.size == 0:
                    featVecs = np.ascontiguousarray(fullFeatures)
                else:
                    featVecs = np.vstack((featVecs,fullFeatures))
        return np.ascontiguousarray(featVecs)

    def getHistory(self, fids, fid, vid, history, bool_peach=False):    
        historyFeatures = []
        fid = int(fid)
        for oldfid in [fid-history[i] for i in range(0,len(history))]:
            if oldfid in self.frameDict.keys():                        
                oldframe = self.frameDict[oldfid]
                if vid in oldframe.keys():
                    oldveh = v.vehicle(oldframe[vid])
                    new = [1]#append a 1 for an indicator
                    new.extend(oldveh.getTrainFeaturesHist(bool_peach))
                    historyFeatures.extend(new)
                else: #vehicle wasnt in scene, append 0s, and a 0 for not existing
                    curveh = v.vehicle(self.frameDict[fid][vid])
                    historyFeatures.extend([0] * (len(curveh.getTrainFeaturesHist(bool_peach))+1))
                    '''firstframewithvid = fid
                    for frameid in range(oldfid, fid): #at max 30 loops
                        oldframe = self.frameDict[frameid]
                        if vid in oldframe.keys():
                            firstframewithvid = frameid
                            break
                    frame = self.frameDict[firstframewithvid]
                    oldveh = v.vehicle(frame[vid])
                    historyFeatures.extend(oldveh.getTrainFeaturesHist())'''
            else: #frame before measurements, append 0s
                curveh = v.vehicle(self.frameDict[fid][vid])
                historyFeatures.extend([0] * (len(curveh.getTrainFeaturesHist(bool_peach))+1))
                '''
                firstframewithvid = fid
                for frameid in range(fids[0], fid): #at max 30 loops
                    oldframe = self.frameDict[frameid]
                    if vid in oldframe.keys():
                        firstframewithvid = frameid
                        break
                frame = self.frameDict[firstframewithvid]
                oldveh = v.vehicle(frame[vid])
                historyFeatures.extend(oldveh.getTrainFeaturesHist())'''
        return historyFeatures
    
    def getTrafficFeatures(self, fid, vid):
        trafficFeatures = []
        frame = self.frameDict[fid]
        veh = v.vehicle(frame[vid])
        car_in_front_vid = veh.getPreceding()
        car_in_rear_vid = veh.getFollowing()
        left_front_vid, left_back_vid, right_front_vid, right_back_vid = dutil.getNeighbors(frame, veh)
        vids = [car_in_front_vid, car_in_rear_vid, left_front_vid, left_back_vid, right_front_vid, right_back_vid]
        numFeatures = (len(["dx","dy"]) + len(["V", "A", "yaw", "hdwyInd", "headway"]))
        for neighbor_vid in vids:
            if neighbor_vid > 0 and neighbor_vid in frame.keys():
                veh2 = v.vehicle(frame[neighbor_vid])
                trafficFeatures.append(1)#indicator
                trafficFeatures.append(veh2.getX() - veh.getX())
                trafficFeatures.append(veh2.getY() - veh.getY())
                trafficFeatures.extend(veh2.getTrainFeaturesTraff())
            else:
                trafficFeatures.extend([0]*(numFeatures+1))
        all_vids = vids
        all_vids.append(vid)
        intersection_vid = self.getIntersectionVID(frame, veh, all_vids)
        if intersection_vid > 0 and intersection_vid in frame.keys():
            veh2 = v.vehicle(frame[intersection_vid])
            trafficFeatures.append(1)#indicator
            trafficFeatures.append(veh2.getX() - veh.getX())
            trafficFeatures.append(veh2.getY() - veh.getY())
            trafficFeatures.extend(veh2.getTrainFeaturesTraff())
        else:
            trafficFeatures.extend([0]*(numFeatures+1))
        return trafficFeatures
        
    def getIntersectionVID(self, frame, veh, otherNeighborsVIDs):
        if veh.getSection == 0:
            notUsedVids = []
            for vid in frame.keys():
                if vid not in otherNeighborsVIDs:
                    notUsedVids.append(vid)
            return (dutil.getXClosest(frame, veh, 1, notUsedVids))[0]
        nextIntersection = self.getNextIntersection(veh)
        vids = frame.keys()
        interVIDs = []
        for vid in vids:
            if not vid in otherNeighborsVIDs: 
                veh2 = v.vehicle(frame[vid])
                if veh2.getIntersection() == nextIntersection:
                    interVIDs.append(vid)
        if len(interVIDs) > 0:
            return (dutil.getXClosest(frame, veh, 1, interVIDs))[0]
        return 0
    
    #check if in an intersection first
    def getNextIntersection(self, veh):
        curFid = veh.getFrame()
        vid = veh.getVID()
        sortedframeIDs = self.getFrames()
        for fid in sortedframeIDs:
            if fid > curFid:
                if not vid in self.frameDict[fid].keys():
                    return 0
                veh2 = v.vehicle(self.frameDict[fid][vid])
                if veh2.getIntersection() > 0:
                    return veh2.getIntersection()
        return 0
    
    def getTargets(self, fids):
        targets = []
        fids = sorted(list(fids))
        numFids = len(fids)
        numDone = 0
        for fid in fids:
            numDone = numDone + 1
            if numDone % int(numFids/10) == 0:
                print("Done making targets for", numDone, "/", numFids, "frames...")
            frame = self.frameDict[fid]
            for vid in sorted(list(frame.keys())):
                veh = v.vehicle(frame[vid])
                targets.append(veh.getNextMove())
        return np.array(targets)
    
    # WARNING!!! this permanently alters the frame dictionary stored in the class
    def reduceDataset(self, newStart, newEnd):
        frames = sorted(list(self.frameDict.keys()))
        numFids = len(frames)
        newfd = {}
        for fid in frames:
            if fid % int(numFids/10) == 0:
                print("Done", fid, "/", numFids, "frames...")
            if fid >= newStart and fid <= newEnd:
                newfd[fid]=self.frameDict[fid]
        self.frameDict = newfd
    #Y = dd2.getNextMoves()
    #X = [LaneID, laneType, Class, Dir, Vel, Accel, augVx, augAx, orientation, 
         #SpaceHdwy, distanceToIntersect]
    
    def countItems(self, fids=None):
        if not fids:
            fids = self.getFrames()
        return sum([len(self.frameDict[fid]) for fid in fids])
        
    def getFrames(self):
        return sorted(list(self.frameDict.keys()))
    
    def getMaxVid(self):
        # brute force cause why not
        maxVid=0
        for fid in self.frameDict.keys():
            for vid in self.frameDict[fid].keys():
                if vid > maxVid:
                    maxVid = vid
        return maxVid
    
    # make it so a batch is of size numVids, and all get passed in at once
    # both newFs and newTs will have an indicator as the first index
    # includes the removal of edge cars (next move indeterminant)
    def convertToLSTMFormat(self, allFeatures_i, allTargets_i, sectionedFids_i):
        maxVid = self.getMaxVid()
        numFids = len(sectionedFids_i)
        numFeatures = len(allFeatures_i[0,:])
        numOutputs = 1#allTargets_i.shape[1]
        newFeatures = np.zeros((numFids, maxVid+1, numFeatures+1))
        newTargets = np.zeros((numFids, maxVid+1, numOutputs))
        j = 0 # j is index into targets
        f_index = 0 # f_index is index into frames, fid - minFid
        num_skipped = 0
        print("Num vids:", maxVid, "numFids:", numFids)
        for fid in sorted(sectionedFids_i):
            vids_in_frame = sorted(list(self.frameDict[fid].keys()))
            for vid in vids_in_frame:
                if allTargets_i[j] > 0:
                    newFeatures[f_index][vid][0] = 1
                    #newTargets[f_index][vid][0] = 1
                    newFeatures[f_index][vid][1:] = allFeatures_i[j]
                    newTargets[f_index][vid] = allTargets_i[j]
                else:
                    num_skipped += 1
                j += 1
            f_index += 1
        print("LSTM convert skipped:", num_skipped)
        return newFeatures, newTargets, maxVid
    
    def convertTo1Hot(features, targets):
        newFeatures = np.zeros(features.shape)
        for feature in features:
            #remove lanetype
            #1-hot lanetype = c.laneTypeEncoding[lanetype]
            #add to features
            newFeatures = features
        newTargets = np.zeros((len(targets), 3))
        for target in targets:
            newTargets = [0,0,0]
            newTargets[target-1] = 1
        return newFeatures, newTargets
        
    def countFeatures(self, fids, laneType=True, useHistory=False, 
                      historyFrames=[], useTraffic=True):
        fid = fids[0]
        frame = self.frameDict[fid]
        vid = list(frame.keys())[0]
        return len(self.getFeaturesForFidVid(fids, fid, vid, laneType, useHistory, 
                                             historyFrames, useTraffic))

    #output is dict interid=>features
    #features will be of shape (numInputs, numFramesInTrajectory, numFeatures)
    # default is (numInputs, 50, 10)
    #output targets will be of same shape with numFeatures = 1
    #changed so it works with intersection ids. for old, see revision history
    #bool peach to get correct lane ids
    def generateTrajFeaturesTargetsForLSTM(self, fids, laneType=True, useHistory=False, 
                                    historyFrames=[], useTraffic=True, 
                                    numFramesToFind=20, intersectionIDs=None, bool_peach=False):
        featVecs = np.ascontiguousarray([])
        features = {}
        targets = np.ascontiguousarray([])
        fids = sorted(list(fids))
        maxfid = fids[-1]
        n = len(fids)
        numDone = 0
        vidToFrameDict = self.makeVidToFrameDict(fids)  
        numvids = len(vidToFrameDict.keys())
        numFeatures = self.countFeatures(fids, laneType, useHistory, 
                                         historyFrames, useTraffic)
        vids = sorted(list(vidToFrameDict.keys()))
        for vid in vids:
            numDone += 1
            if numDone % 10 == 0:#numvids/10 == 0:
                print("Done making features and targets for", numDone, "/", numvids, "vids, (", 
                    int(laneType), int(useHistory), int(useTraffic), ")")

            frames = sorted(vidToFrameDict[vid], reverse=True) #go over this backwards, adding to featureMatrix
            #this limits the number of trajectories that must be removed
            trajFeatsVID = self.generateTrajFeaturesLSTM(vid, frames, laneType, useHistory,
                                                    historyFrames, useTraffic, numFramesToFind, bool_peach) 
                         #trajFeatsVID will be dict (inter=>features), features shape (x, trajLen, numFeatures+3)
            for interID in trajFeatsVID:
                if interID in features:
                    features[interID] = np.vstack((features[interID],trajFeatsVID[interID]))
                else:
                    features[interID] = trajFeatsVID[interID]
        return features

 
    #frames is reverse sorted fids that vid appears in
    #returns dict(intersections => features)
    #    features of shape (x, TrajLen, numFeatures+3 (nextMove, fid, vid))
    # x depends on how many full trajectories can be produced, how long vehicle is in scene.
    def generateTrajFeaturesLSTM(self, vid, frames, laneType, useHistory, historyFrames, useTraffic, TrajLen, bool_peach):
       result = {} 
       curTrajFeatures = None #keep track of one feature thing, when it gets to traj len, add to result
            #will be shape (trajLen, numFeatures+3)
       curInter = None
       for fid in frames:
            veh = v.vehicle(self.frameDict[fid][vid])
            interid = self.getNextIntersectionAndRecord(veh, frames, None, None) #remember, the frames are all the frames the vehicle appears in
            nextMove = veh.getNextMove()
            if nextMove == 0 or interid == 0 or veh.getIntersection() > 0: #if vehicle is not approaching an intersection or is already in one
                curTrajFeatures = None #reset
                curInter = None
                continue
            fidvid = [fid, vid]
            fullFeatures = self.getFeaturesForFidVid([], fid, vid, laneType, 
                                                     useHistory, historyFrames, useTraffic, bool_peach) #this is a list
            fullFeatures.extend(fidvid)
            fullFeatures.append(nextMove)
            if (curInter and interid == curInter) or curInter == None:
                curInter = interid
            else:
                curTrajFeatures = None #reset
                curInter = None
                continue    
            if curTrajFeatures != None:
                #print("Before apend", np.array(curTrajFeatures).shape)
                #curTrajFeatures.append(fullFeatures)
                #print("Before insert", np.array(curTrajFeatures).shape)
                curTrajFeatures.insert(0,fullFeatures)
                #print("After both", np.array(curTrajFeatures).shape)
                #features in front because we are reverse iterating
            else:
                curTrajFeatures = [fullFeatures]
            if len(curTrajFeatures) >= TrajLen: #> for some tests, should not affect actual
                if curInter in result:
                    result[curInter] = np.vstack((result[curInter], np.ascontiguousarray([curTrajFeatures])))
                else:
                    result[curInter] = np.ascontiguousarray([curTrajFeatures])
                curTrajFeatures = None #reset
                curInter = None
       return result

    def getFeaturesForFidVid(self, fids, fid, vid, laneType=True, 
                             useHistory=False, historyFrames=[], useTraffic=True, bool_peach=False):
        frame = self.frameDict[fid]
        veh = v.vehicle(frame[vid])
        fullFeatures = []
        if laneType:
            fullFeatures.extend(veh.getTrainFeatures1(bool_peach))
        else:
            fullFeatures.extend(veh.getTrainFeatures2(bool_peach))
        if useHistory:
            fullFeatures.extend(self.getHistory(fids, fid, vid, historyFrames, bool_peach))
        if useTraffic:
            fullFeatures.extend(self.getTrafficFeatures(fid,vid))
        return fullFeatures
        
        
    def makeVidToFrameDict(self, fids):
        vidToFrameDict = {}
        for fid in fids:
            frame = self.frameDict[fid]
            for vid in frame.keys():
                if vid in vidToFrameDict.keys():
                    vidToFrameDict[vid].append(fid)
                else:
                    vidToFrameDict[vid] = [fid]
        return vidToFrameDict
        

