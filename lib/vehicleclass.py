import collections, itertools, copy
import numpy, scipy, math, random
import os, sys, time, importlib
import tokenize, re, string
import json, unicodedata
#import lib.constants as c
from lib import constants as c
import math

# All variables are being referenced by their index instead of their original
# names.
class vehicle:
    def __init__(self, augArray, compressed=False):
        if not compressed:
            self.vid = int(augArray[c.VehicleID])
            self.fid = int(augArray[c.FrameID])
            self.numFrames = int(augArray[c.TotFrames])
            self.globalT = augArray[c.GlobalT]
            self.x = float(augArray[c.LocalX])
            self.y = float(augArray[c.LocalY])
            self.Vy = float("{0:.2f}".format(float(augArray[c.Vel])))
            self.Ay = float("{0:.2f}".format(float(augArray[c.Accel])))
            self.V = self.Vy
            self.A = self.Ay
            self.Vx = float("{0:.2f}".format(float(augArray[c.augVx])))
            self.Ax = float("{0:.2f}".format(float(augArray[c.augAx])))
            self.org = int(augArray[c.Origin])
            self.dest = int(augArray[c.Dest])
            self.dir = int(augArray[c.Dir])
            self.act= int(augArray[c.Movement])
            self.inter = int(augArray[c.Intersect])
            self.sect = int(augArray[c.Section])
            self.lane = int(augArray[c.LaneID])
            self.preceding = int(augArray[c.Preceding])
            self.following = int(augArray[c.Following])
            self.length = augArray[c.Len]
            self.width = augArray[c.Wid]
            self.type = int(augArray[c.Class])
            self.timeHeadway = augArray[c.TimeHdwy]
            self.spaceHeadway = float("{0:.2f}".format(float(augArray[c.SpaceHdwy])))
            
            self.destLane = 0 #not computed yet
            if len(augArray) > c.destLane:
                self.destLane = int(augArray[c.destLane])

            if len(augArray) > c.orientation and abs(float(augArray[c.orientation])) <= 2*math.pi:
                self.orientation = float(augArray[c.orientation])
            else:
                self.orientation = self.getOrientation()#yaw - Due East = 0 radians

            if len(augArray) > c.goaly:
                self.goalx = float(augArray[c.goalx])
                self.goaly = float(augArray[c.goaly])
            else:
                self.goalx = 0
                self.goaly = 0
            
            if len(augArray) >= c.Distance:
                self.laneType = int(augArray[c.laneType])
                self.nextMove = int(augArray[c.nextMove])
                self.Distance = float(augArray[c.Distance])
            else:
                self.laneType, self.nextMove, self.Distance = [0]*3
            


        else:
            self.vid = int(augArray[0])
            self.fid = int(augArray[1])
            self.x = float(augArray[2])
            self.y = float(augArray[3])
            self.Vx = float(augArray[4])
            self.Ax = float(augArray[5])
            self.Vy = float(augArray[6])
            self.Ay = float(augArray[7])
            self.lane = int(augArray[8])
            self.preceding = int(augArray[9])
            self.following = int(augArray[10])
            self.spaceHeadway = float(augArray[11])
            self.dest = int(augArray[12])
            self.destLane = int(augArray[13])
            self.dir = int(augArray[14])
            self.type = int(augArray[15])
            self.length = float(augArray[16])
            self.width = float(augArray[17])
            self.orientation = float(augArray[18])
            self.goalx = float(augArray[19])
            self.goaly = float(augArray[20])

        """
        Uncomment out depending on which model we're running
        This is what you should change to change what's included in the grid.
        """
        self.GridInfo = [1, self.Vx, self.Ax]#, self.Vy, self.Ay]#, self.spaceHeadway]
        #self.GridInfo = [1, self.Vx, self.Ax, self.spaceHeadway, self.timeHeadway]
        #, self.Vy, self.Ay]#, self.spaceHeadway]
        #idk what timeHeadway and spaceHeadway are, used both
        #self.GridInfo = [1, self.Vx, self.Ax]

    #lane should be an int
    def setDestLane(self, lane):
        self.destLane = lane
    def getDestLane(self):
        if self.destLane > 3 or self.destLane < 1:
            return 0
        return self.destLane
        
    def setOrientation(self, yaw_in_rads):
        self.orientation = yaw_in_rads
    def getOrientation(self):
        yaw = self.orientation
        if abs(yaw) > math.pi:
            if (self.dir == 1): #East
                yaw = 0
            elif (self.dir == 2): # North
                yaw = (0.5*math.pi)
            elif (self.dir == 3): # West
                yaw = (math.pi)
            elif (self.dir == 4): # South
                yaw = (-0.5*math.pi)
            else:
                print("Invalid Direction for vehicle", self.getVID())
                return None 
        if abs(yaw) < 0.0001:
            yaw = 0
        return yaw
    def getLaneBasedYaw(self, yaw=None):
        if not yaw:
            yaw = self.orientation
        return yaw - (self.dir - 1) * (0.5 * math.pi)
        #
        if (self.dir == 1): #East
            yaw += 0
        elif (self.dir == 2): # North
            yaw -= (0.5*math.pi)
        elif (self.dir == 3): # West
            yaw -= (math.pi)
        elif (self.dir == 4): # South
            yaw -= (1.5*math.pi)
        return yaw


    def getClusteringFeatures(self):
        return [self.Vx, self.Ax, self.Vy, self.Ay, self.timeHeadway, self.spaceHeadway]

    def getGoalCords(self):
        return self.goalx, self.goaly
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getFrame(self):
        return self.fid
    def getVID(self):
        return self.vid
    def getVx(self):
        return self.Vx
    def getAx(self):
        return self.Ax
    def getVy(self):
        return self.Vy
    def getAy(self):
        return self.Ay
    def getMove(self):
        return self.act
    def getLane(self):
        return self.lane
    def getSect(self):
        return self.sect
    def getSection(self):
        return self.sect
    def getIntersection(self):
        return self.inter
    def getPreceding(self):
        return self.preceding
    def getFollowing(self):
        return self.following
    def getTimeHeadway(self):
        return self.timeHeadway
    def getSpaceHeadway(self):
        return self.spaceHeadway
    def getTrajectory(self):
        return [self.x, self.y, self.Vx, self.Vy, self.Ax, self.Ay]
    def getDest(self):
        return self.dest
    def getDir(self):
        return self.dir
    def getGridInfo(self):
        return self.GridInfo
    def getGridInfoLen(self):
        return len(self.GridInfo)
    def getWidth(self):
        return self.width
    def getLength(self):
        return self.length
    def getClass(self):
        return self.type
    def getGlobalT(self):
        return self.globalT
    def getNextMove(self):
        return self.nextMove
    def returnCompressedArray(self):
        return [self.vid, self.fid, self.numFrames, self.x, self.y, self.Vy, 
                self.Ay, self.Vx, self.Ax, self.lane, self.dest, self.dir] #, 
  #              self.timeHeadway, self.spaceHeadway]
    def returnArrayInfo(self):
        goalx, goaly = self.getGoalCords()
        arr = []
        arr.extend([self.getVID(),self.getFrame(),self.getX(),self.getY()])
        arr.extend([self.getVx(),self.getAx(),self.getVy(),self.getAy(), self.getLane()])
        arr.extend([self.getPreceding(),self.getFollowing(),self.getSpaceHeadway()])
        arr.extend([self.getDest(),self.getDestLane(),self.getDir(), self.getClass()])
        arr.extend([self.getLength(),self.getWidth(),self.getOrientation()])
        arr.extend([goalx, goaly])
        return arr
    
    
    #[LaneID, laneType, Class, Dir, Vel, Accel, augVx, augAx, orientation, 
         #SpaceHdwy, distanceToIntersect]
    
    def getLanesToSidesInSection(self, bool_peach=False):
        return c.getLanesToSides(self.sect, self.dir, self.lane, self.Distance, 
                                 bool_peach)
        
                

    def getLanesToSidesInIntersection(self):
        return -1, -1
        
    def getLanesToSides(self, bool_peach=False):
        lanesToMedian = 0
        lanesToCurb = 0
        if self.sect > 0:
            lanesToMedian, lanesToCurb = self.getLanesToSidesInSection(bool_peach)
        else:
            lanesToMedian, lanesToCurb = self.getLanesToSidesInIntersection()
        if lanesToMedian < -1:
            lanesToMedian = -1
        if lanesToMedian >= 7:
            lanesToMedian = -1
        if lanesToCurb >= 7:
            lanesToCurb = -1
        if lanesToCurb < -1:
            lanesToCurb = -1
        return lanesToMedian, lanesToCurb

    def getLaneType(self):
        return c.laneTypeEncoding(self.laneType)
    
    def getTrainFeaturesHist(self, bool_peach=False):
        return self.getTrainFeatures2(bool_peach)
        #lanesToMedian, lanesToCurb = self.getLanesToSides()
        #return [lanesToMedian, lanesToCurb, self.Vy, self.Ay, self.Vx, self.Ax, self.orientation, self.spaceHeadway]

    def getTrainFeaturesTraff(self):
        return [self.V, self.A, self.getLaneBasedYaw(), int(self.spaceHeadway > 0.0), self.spaceHeadway]
        #return [self.Vy, self.Ay, self.Vx, self.Ax, self.orientation, self.spaceHeadway]
        
    def getTrainFeaturesTraffPreceding(self):
        return [self.Vy, self.Ay, self.Vx, self.Ax, self.getLaneBasedYaw(), self.spaceHeadway]
    
    def getTrainFeaturesTraffFollowing(self):
        return [self.Vy, self.Ay, self.Vx, self.Ax, self.getLaneBasedYaw()]
    
    def getTrainFeaturesTraffIntersect(self):
        return [self.Vy, self.Ay, self.Vx, self.Ax, self.getLaneBasedYaw(), self.spaceHeadway]

    def getTrainFeatures1(self, bool_peach=False):
        lanesToMedian, lanesToCurb = self.getLanesToSides(bool_peach)
        laneType = self.getLaneType() #1-hot encoding, len==4
        #laneType = self.laneType  #integer
        features = [lanesToMedian, lanesToCurb]
        features.extend(laneType)
        features.extend([self.V, self.A, self.getLaneBasedYaw(), int(self.spaceHeadway > 0.0), self.spaceHeadway, self.Distance])
        #features.extend([self.Vy, self.Ay, self.Vx, self.Ax, self.orientation, self.spaceHeadway, self.Distance])
        return features 
    
    def getTrainFeatures2(self, bool_peach=False):
        lanesToMedian, lanesToCurb = self.getLanesToSides(bool_peach)
        return [lanesToMedian, lanesToCurb, self.V, self.A,
                self.getLaneBasedYaw(), int(self.spaceHeadway > 0.0), self.spaceHeadway, self.Distance]
        #return [lanesToMedian, lanesToCurb, self.Vy, self.Ay,
        #        self.Vx, self.Ax, self.orientation, self.spaceHeadway, self.Distance]
    
