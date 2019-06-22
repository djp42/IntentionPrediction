# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:02:14 2016

@author: LordPhillips
"""

from utils import constants as c
from utils import frame_util as futil
from utils import vehicleclass as v
import numpy as np
import os
import math
import time
import re
import random
      

'''Returns a list of vids in the prev frame but not in cur frame'''
def getExitedVIDs(prevFrame,curFrame):
    setOfIDs = prevFrame.keys() - curFrame.keys()
    return list(setOfIDs)
        
class data:
    def __init__(self, filepath, compressed, goals=True, constraints=True, featurepath=None):
        self.compressed = compressed
        self.featurepath = featurepath
        self.filepath = filepath
        self.VIDs = []
        self.frameDict = self.makeFrameDict(filepath, self.compressed)
        self.firstFrame = min(self.frameDict.keys())
        self.numFrames = max(self.frameDict.keys()) #last frame, technically
        self.maxVID = self.findMaxVID()
        if constraints:
            self.maxAccel, self.minAccel, self.maxVel, self.minVel = self.findConstraints()
        else:
            self.maxAccel, self.minAccel, self.maxVel, self.minVel = [0]*4
        if goals:
            self.goals = self.findGoalCords()   #also finds dest lanes      
        else:
            self.goals = None
        #self.agentIDs = range(1,self.maxVID)
        #self.agents = self.initAgents()
    
    '''FrameDict will have entries where every key is a dict of all vehicles
    that are present in that frame framedict = s{frameid: {vid:vehicledata}}'''
    def makeFrameDict(self, filepath, compressed):
        trajectoryFile = open(filepath, 'r')
        lines = trajectoryFile.readlines()
        numLines = len(lines)
        lineCounter = 0
        frameDict = {}
        for line in lines:
            if lineCounter % 30000 == 0:
                print("Processed line for dict: ", lineCounter, "/", numLines)
            array = line.split()
            veh = v.vehicle(array, compressed)
            if not veh.getVID() in self.VIDs:
                self.VIDs.append(veh.getVID())
            fid = veh.getFrame()
            if fid in frameDict.keys():  
                #frameDict[fid] is a dict of vid:vehicledata
                frameDict[fid][veh.getVID()] = array
            else:
                frameDict[fid] = {veh.getVID(): array}
            lineCounter += 1
        return frameDict    

    '''Add lane type, movement at next intersection
    To be used after frame dict has been made
    designed for the bayes net stuff im doing in Julia
    Does not keep the new data structure permanently, but does write to
    file that will be read in julia
    #laneid serves as a count of 'closeness' to side, except bays
    '''
    def augmentforJulia(self):
        newFD = {}
        n_frames_skip = 0
        laneTypes = c.lanetypes
        nextMoves = {} #vid-fid : move at next intersection
        distances = {} #same, except distance to intersection
        vidfidstoupdate = {}
        s = '-' #this is sep
        sortedFrames = sorted(list(self.frameDict.keys()), reverse=False)
        framesDone = 0
        for fid in sortedFrames:
            if framesDone % 1000 == 0:
                print("First pass: done", framesDone, "/", len(sortedFrames),"...")
            framesDone = framesDone + 1
            frame = self.frameDict[fid]
            for vid in frame.keys():
                veh = v.vehicle(frame[vid], self.compressed)
                if veh.getSect() > 0:
                    posX = veh.getX()
                    posY = veh.getY() 
                    if vid in vidfidstoupdate.keys():
                        vidfidstoupdate[vid][fid]=[posX,posY]
                    else:
                        vidfidstoupdate[vid] = {fid:[posX,posY]}
                else:
                    move = veh.getMove()
                    posX = veh.getX()
                    posY = veh.getY()
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
        print("First pass done")       
        framesDone = 0
        for fid in sortedFrames:
            if framesDone % 1000 == 0:
                print("First pass: done", framesDone, "/", len(sortedFrames),"...")
            framesDone = framesDone + 1
            if fid < n_frames_skip:
                continue
            #first pass
            frame = self.frameDict[fid]
            for vid in frame.keys():
                veharr = frame[vid]
                veh = v.vehicle(veharr, self.compressed)
                lanetypeindex = str(veh.getSect())+s+str(veh.getDir())+s+str(veh.getLane())
                if lanetypeindex in laneTypes.keys():
                    veharr.append(laneTypes[lanetypeindex])
                else:
                    veharr.append(0) # if not one of those anything can happen
                mv_dis_indx = str(vid)+s+str(fid)
                if mv_dis_indx in nextMoves.keys():
                    veharr.append(nextMoves[mv_dis_indx])
                else:
                    veharr.append(0)
                if mv_dis_indx in distances.keys():
                    veharr.append(distances[mv_dis_indx])
                else:
                    veharr.append(0)
                frame[vid] = veharr
            newFD[fid] = frame
        futil.saveJuliaFD(self.filepath, newFD)

                

    def save(self, constraints=False):
        print("Saving frameDict...")
        futil.saveFrameDict(self.filepath,self.frameDict, self.compressed)
        if constraints:
            print("Saving constraints and goal cords...")
            self.saveConstraintsCords()
            
    def loadFeatureConstraints(self, goalsOrConstraints='goals'):
        filepath = self.featurepath
        trajectoryFile = open(filepath, 'r')
        lines = trajectoryFile.readlines()
        goals = False
        constraints = False
        goal_to_cords = {} 
        constrs = {}
        constrsDone = 0
        #will be a dict that maps dest_num: (avg_x, avg_y), n
        #key = (dest, destLane)
        #d[key]=(x,y),n -- d[key][0]=(x,y) -- d[key][1] = n -- d[key][0][0] = x
        for line in lines:
            if 'GOALS TO CORDS' in line:
                goals = True
                constraints = False
                continue
            if 'CONSTRAINTS' in line:
                constraints = True
                goals = False
                constrsDone = 1
                continue
            array = list(filter(None, re.split("[, (\n)]",line)))
            if goals and 'oal' in goalsOrConstraints:
                dest = int(array[0])
                destLane = int(array[1])
                destX = float("{0:.2f}".format(float(array[2])))
                destY = float("{0:.2f}".format(float(array[3])))
                n = int(array[4])
                goal_to_cords[(dest,destLane)] = (destX,destY),n
            elif constraints and 'onstr' in goalsOrConstraints:
                constrs[constrsDone] = array
                constrsDone = constrsDone + 1
        if 'onstr' in goalsOrConstraints:
            maxa = {1:constrs[1][0], 2:constrs[2][0], 3:constrs[3][0]}
            mina = {1:constrs[1][1], 2:constrs[2][1], 3:constrs[3][1]}
            maxv = {1:constrs[1][2], 2:constrs[2][2], 3:constrs[3][2]}
            minv = {1:constrs[1][3], 2:constrs[2][3], 3:constrs[3][3]}
            return maxa, mina, maxv, minv
        if 'oal' in goalsOrConstraints:
            return goal_to_cords

    def saveConstraintsCords(self):
        filename = os.path.basename(self.filepath)
        outpath = self.filepath[:-len(filename)]+'FEATURES_'+filename
        outFile = open(outpath, 'w')
        goal_to_cords_dict = self.goals
        keys = list(goal_to_cords_dict.keys())
        keys.sort() #to make things nicer
        outFile.write("GOALS TO CORDS\n")
        for key in keys:
            #key = (dest, destLane)
            #d[key]=(x,y),n -- d[key][0]=(x,y) -- d[key][1] = n -- d[key][0][0] = x
            (dest, destLane) = key
            xCord = goal_to_cords_dict[key][0]
            yCord = goal_to_cords_dict[key][1]
            outArr = [dest, destLane, xCord, yCord]
            outFile.write(futil.arrToStrForSave(outArr))
        outFile.write("CONSTRAINTS\n")
        mxA, mnA, mxV, mnV = self.maxAccel, self.minAccel, self.maxVel, self.minVel
        bikeConstrs = [mxA[1],mnA[1],mxV[1],mnV[1]]
        carConstrs = [mxA[2],mnA[2],mxV[2],mnV[2]]
        truckConstrs = [mxA[3],mnA[3],mxV[3],mnV[3]]
        outFile.write(futil.arrToStrForSave(bikeConstrs))
        outFile.write(futil.arrToStrForSave(carConstrs))
        outFile.write(futil.arrToStrForSave(truckConstrs))
        outFile.close()
    
    def findMaxVID(self):
        lastFrame = self.frameDict[self.numFrames]
        maxvid = 0        
        for vid in lastFrame.keys():
            veh = v.vehicle(lastFrame[vid],self.compressed)
            maxvid = max(int(veh.getVID()), maxvid)
        return maxvid

    #find constraints 
    def findConstraints(self):
        #1-moto, 2-auto, 3-truck
        if self.featurepath:
            print("Loading constraints...")
            return self.loadFeatureConstraints('constraints')
        print("Finding constraints...")
        maxaccel = {1:0,2:0,3:0}
        minaccel = {1:0,2:0,3:0}
        maxvel = {1:0,2:0,3:0}
        minvel = {1:0,2:0,3:0}
        for frameid in self.frameDict.keys():
            if int(frameid) % 300 == 0:
                print("on frame", frameid, "...")
            for vid in self.frameDict[frameid].keys():
                veh = v.vehicle(self.frameDict[frameid][vid],self.compressed)
                vclass = veh.getClass()
                if veh.getAx() < 30 and veh.getAy() < 30:
                    maxaccel[vclass] = max(veh.getAx(), veh.getAy(), maxaccel[vclass])
                if veh.getAx() > -30 and veh.getAy() > -30:
                    minaccel[vclass] = min(veh.getAx(), veh.getAy(), minaccel[vclass])
                maxvel[vclass] = max(veh.getVx(), veh.getVy(), maxvel[vclass])
                minvel[vclass] = min(veh.getVx(), veh.getVy(), minvel[vclass])
        return maxaccel, minaccel, maxvel, minvel

    def checkEntryLen(self,desired_length, frameid, vid, errmsg="Error:"):
        if not len(self.frameDict[frameid][vid]) == desired_length:
            print(errmsg, len(self.frameDict[frameid][vid]))
            print(frameid, vid)
            return True
        return False
            
    def checkFrameLen(self, desired_length, frameid, errmsg="Error:", yaw=False):
        for vid in self.frameDict[frameid].keys():
            if self.checkEntryLen(desired_length,frameid,vid,errmsg):
                if yaw:
                    veh = v.vehicle(self.frameDict[frameid][vid], self.compressed)
                    print("YAW", veh.getOrientation())

    def checkDictLen(self, desired_length, errmsg="Error:"):
        for frameid in self.frameDict.keys():
            self.checkFrameLen (desired_length, frameid, errmsg)

    def updateOrientations(self, frameids):
        self.setFirstOrient()
        prevFrame = self.frameDict[self.firstFrame]
        #self.checkFrameLen(28, self.firstFrame, "Error in first orientations")
        for frameid in frameids[1:]:
            curFrame = self.frameDict[frameid]
            self.setOrientations(prevFrame, curFrame, frameid)
            #self.checkFrameLen(28,frameid,"Error in orientation, med", True)
            prevFrame = curFrame
        
    def calcYaw(self, dy, dx):
        yaw = math.atan2(dy,dx)
        return yaw
    
    def setFirstOrient(self):
        frame = self.frameDict[self.firstFrame]
        for vid in frame.keys():
            arr = frame[vid]
            veh = v.vehicle(arr,self.compressed)
            arr.append(veh.getOrientation())
            self.frameDict[self.firstFrame][vid] = arr

    '''to be used with outer function, do after destlane is set'''
    def setOrientations(self, prevFrame, curFrame, curid):
        threshold = 0.5  #if the car moves a very small amount dont update
        max_diff = math.pi/4 #if there is too much rotation ignore.
        numToAverage = 3  #smooth orientation over the last 3 orientations
        for vid in curFrame.keys():
            if vid in prevFrame.keys():
                veh_cur = v.vehicle(curFrame[vid], self.compressed)
                veh_prev = v.vehicle(prevFrame[vid], self.compressed)
                dx = veh_cur.getX()-veh_prev.getX()
                dy = veh_cur.getY()-veh_prev.getY()
                if abs(dx) < threshold or abs(dy) < threshold:
                    if abs(self.calcYaw(dy,dx) - veh_prev.getOrientation()) > max_diff:
                        yaw = veh_prev.getOrientation()
                    else:
                        yaw = self.calcYaw(dy,dx)
                else:
                    yaw = self.calcYaw(dy,dx)
                prev_tot = veh_prev.getOrientation() * numToAverage
                new_avg = ((prev_tot - veh_prev.getOrientation()) + yaw) / numToAverage
                new_veh_data = curFrame[vid]
                new_veh_data.append(new_avg)
                self.frameDict[curid][vid] = new_veh_data           
                #if self.checkEntryLen(28,curid,vid,"Error in orientation, innermost"):
                #    print("Yaw:", yaw)
     
    def setDestLanes(self, destLaneDict):
        for frameid in self.frameDict.keys():
            for vid in self.frameDict[frameid].keys():
                destLane = destLaneDict[vid]
                new_veh_data = self.frameDict[frameid][vid]
                new_veh_data.append(destLane)
                self.frameDict[frameid][vid] = new_veh_data
            #self.checkFrameLen(27,frameid,"Error in dest lanes, innermost")


    def setDestCords(self, goal_to_cords):
        for frameid in self.frameDict.keys():
            for vid in self.frameDict[frameid].keys():
                new_veh_array = self.frameDict[frameid][vid]
                veh = v.vehicle(new_veh_array, self.compressed)
                dest = veh.getDest()
                destLane = veh.getDestLane()
                cords = goal_to_cords[(dest,destLane)][0]   
                new_veh_array.append(cords[0])
                new_veh_array.append(cords[1])
                self.frameDict[frameid][vid] = new_veh_array
            #self.checkFrameLen(30,frameid,"Error in destination cords, innermost")


    def goalCalcs(self, frame, vid, goal_to_cords):
        veh = v.vehicle(frame[vid],self.compressed)
        dest = veh.getDest()
        x = veh.getX()
        y = veh.getY()
        destLane = veh.getLane()
        if (dest,destLane) in goal_to_cords.keys():
            val = goal_to_cords[(dest,destLane)]
            prev_n = val[1]
            prev_avgX = val[0][0]
            prev_avgY = val[0][1]
        else:
            prev_n, prev_avgX, prev_avgY = [0]*3
        new_avgX = float("{0:.2f}".format(float(((prev_avgX*prev_n) + x)/(prev_n+1))))
        new_avgY = float("{0:.2f}".format(float(((prev_avgY*prev_n) + y)/(prev_n+1))))
        newGoal = (new_avgX, new_avgY), prev_n+1
        return dest, destLane, newGoal
    
    '''finds goals by averaging last coordinates of cars that go there
    (and find lanes)
    (and set orientation)'''
    def findGoalCords(self):
        if self.featurepath:
            print("Loading goals...")
            return self.loadFeatureConstraints('goals')
            #do not need to update anything either
        goal_to_cords = {} 
        #will be a dict that maps dest_num: (avg_x, avg_y), n
        #key = (dest, destLane)
        #d[key]=(x,y),n -- d[key][0]=(x,y) -- d[key][1] = n -- d[key][0][0] = x
        dest_lanes = {} #vid:destLane
        frameids = list(self.frameDict.keys())
        prevFrame = self.frameDict[self.firstFrame]
        for frameid in frameids[1:]:
            curFrame = self.frameDict[frameid]
            exited_vids = getExitedVIDs(prevFrame,curFrame)
            for vid in exited_vids:
               dest, destLane, newGoal = self.goalCalcs(prevFrame, vid, goal_to_cords)
               dest_lanes[vid] = destLane
               goal_to_cords[(dest,destLane)] = newGoal
            if frameid == self.numFrames:
                #last frame
                for vid in curFrame.keys():
                    if not vid in dest_lanes.keys():
                        dest, destLane, newGoal = self.goalCalcs(curFrame, vid, goal_to_cords)
                        dest_lanes[vid] = destLane
                        goal_to_cords[(dest,destLane)] = newGoal
            prevFrame = curFrame
        if not self.compressed:
            self.updateDict(dest_lanes, frameids, goal_to_cords)
        return goal_to_cords
    
    def hasVID(self, vid):
        return vid in self.VIDs
    def getRandVid(self):
        randFrame = random.choice(list(self.frameDict.keys()))
        return random.choice(list(self.frameDict[randFrame].keys()))
    def updateDict(self, dest_lanes, frameids, goal_to_cords):
        print("Updating destination lanes...")
        self.setDestLanes(dest_lanes) #sets all dest lanes
        #self.checkDictLen(27,"Error in destination lanes, outer")
        print("Udating orientations...")
        self.updateOrientations(frameids)
        #self.checkDictLen(28,"Error in orientations, outer")
        print("Updating destination coordinates...")
        self.setDestCords(goal_to_cords)
        #self.checkDictLen(30,"Error in dest cords, outer")
        
    def lastFrameForVID(self, vid):
        fd = self.getFrameDict()
        lastF = -1
        if not self.hasVID(vid):
            return -1
        for fid in fd.keys():
            if vid in fd[fid]:
                lastF = fid
            else:
                if lastF > 0:
                    return lastF
        return lastF
    
    def getDest(self, fid, vid):
        arr = self.frameDict[fid][vid]
        veh = v.vehicle(arr, self.compressed)
        return veh.getDest(), veh.getDestLane()
    def vidIsLastInFrame(self, vid, fid):
        return vid == max(self.frameDict[fid].keys())
    def firstVIDinFrame(self, fid):
        return min(list(self.frameDict[fid].keys()))
    def getVIDs(self):
        return self.VIDs
    def getFrameDict(self):
        return self.frameDict
    '''getFrame returns a dictionary for the frame indicated by 'key' '''
    def getFrame(self,key):
        if key > self.numFrames or key < self.firstFrame:
            return None
        return self.frameDict[key]
    def getFirstFrame(self):
        return self.firstFrame
    def getNumFrames(self):
        return self.numFrames
    def getCompressed(self):
        return self.compressed
    def getMaxVID(self):
        return self.maxVID
    def getConstraints(self):
        return self.maxAccel, self.minAccel, self.maxVel, self.minVel 
    def getGoals(self):
        return self.goals
    def getPath(self):
        return self.filepath
    def getVehicle(self, vid, fid):
        return v.vehicle(self.frameDict[fid][vid], self.compressed)

