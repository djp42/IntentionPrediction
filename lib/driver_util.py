# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:27:58 2016

@author: LordPhillips
"""
import os
import datetime
from lib import constants as c
from lib import vehicleclass as v
from lib import frame_util as futil

'''The ngsim datetime is wrong. this converts.
example input:      1118936700200
example conversion: "1118936700.200"
example output:     datetime.datetime.fromtimestamp(1118936700.200)'''
def convertNGSIMtimestampToDateTime(timestamp):
    if not len(str(timestamp)) == len("1118936700200"):
        print("Error in converting timestamp, original lengths don't match")
        return None
    string = str(timestamp)
    conversion = string[:-3] + "." + string[-3:]
    return datetime.datetime.fromtimestamp(float(conversion))

'''Helper to isolate deciding if a situation is interesting'''
#broken down so easy to add more cases
def inInterestingScenario(vehicle):
    if vehicle.getSpaceHeadway() > 10:
        return False
    if vehicle.getVx() < 5:
        return False
    return True
    
'''keeps track of a dictionary of interesting frameIDs and position ranges'''
def find_interesting_parts(data):
    interesting_parts = {}
    for frameid in range(data.getFirstFrame(), data.getNumFrames()+1):
        frame = data.getFrame(frameid)
        interestingVehiclesInFrame = []
        for vid in frame.keys():
            veh = v.vehicle(frame[vid], data.getCompressed())
            if inInterestingScenario(veh):
                interestingVehiclesInFrame.append(vid)
        interesting_parts[frameid]=interestingVehiclesInFrame
    return interesting_parts
    
def test_goal_predictions(data):
    cur_frame = 0
    max_frame = data.getNumFrames()
    predictions = {}
    while cur_frame <= max_frame:
        sim_step(data, cur_frame)
        for agent in agents:
            if not agent.ignore: 
                goal = agent.predict_goal #not right
                predictions[agent].append(goal)
        cur_frame += 1
    evaluate_pred_results(predictions, actuals)

''' essentially 'loads' scenario for current frame'''
def sim_step(data, frame): 
    dict_of_cars = data.get_dict(frame)
    agents = range(1,data.getMaxVID())
    for agent in agents:  #Here agent.id is the vehicle id
        if agent in dict_of_cars: 
            agent.ignore = False  #Indicates agent is not in frame 
                                  #(use this if just having all agents stored)
            agent.vel.x = dict_of_cars(agent.id).vel.x
            agent.vel.y = dict_of_cars(agent.id).vel.y
            agent.pos.x = dict_of_cars(agent.id).pos.x
            agent.pos.y = dict_of_cars(agent.id).pos.y
        else:
            agent.ignore = True
      
def evaluate_pred_results(pred, act):
    return pred - act

def visualize(dataobject):
    futil.AnimateData(dataobject)


#finds the path to the file in RESOURCES that matches
def findPathForFile(filename):
    for subdir, dirs, files in os.walk(c.PATH_TO_RESOURCES):
        for file in files:
            filepath = subdir + os.sep + file
            if not filepath.endswith(".txt"): 
                continue     
            Thisfilename = os.path.basename(filepath)
            if filename == Thisfilename:
                return filepath
    print("No matching file in RESOURCES")
    return None

#assumes filename starts with 'driver_compr_'
def findFeaturePath(filename):
    featurefile = 'FEATURES_' + filename[len('driv_compr_'):]
    print(featurefile)
    return findPathForFile(featurefile)
    
def printConstraints(maxa, mina, maxv, minv):    
    print("Constraints:")
    print("Max Acceleration = ", maxa)
    print("Min Acceleration = ", mina)
    print("Max Velocity = ", maxv)
    print("Min Velocity = ", minv)
    
def findPrintAvg(frameDict, isCompressed):
    print("Finding averages")
    sumAx = 0
    sumAy = 0
    sumVx = 0
    sumVy = 0
    n = 0
    for frame in frameDict.keys():
        for vid in frameDict[frame].keys():
            veh = v.vehicle(frameDict[frame][vid], isCompressed)
            sumAx = sumAx + abs(veh.getAx())
            sumAy = sumAy + abs(veh.getAy())
            sumVx = sumVx + abs(veh.getVx())
            sumVy = sumVy + abs(veh.getVy())
            n = n + 1
    avg = [sumAx/n, sumAy/n, sumVx/n, sumVy/n]
    print(avg)
