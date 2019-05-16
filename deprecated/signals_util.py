# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:06:07 2016

@author: LordPhillips

traffic signal utilities

"""

from utils import vehicleclass as v
from utils import driver_util as dru
from utils import constants as c
from utils import data2_class as dd2
from utils import frame_util as fut
import matplotlib.pyplot as plt
import datetime

'''the NGSIM data had the intersections labelled as 87-90 for the signal
timing sheets, but for the trajectory data they are 1-4. This converts either way'''
def convertIntersection(number):
    if number > 10:
        return number-86
    else:
        return number+86

'''The signals have a special timestamp that is just the hour:minute:second,
so I have to add the constants.year,month,day to make a datetime object'''
def convertTimeStamp(stringTime):
    hour = int(stringTime[:2])
    minute = int(stringTime[3:5])
    second = int(stringTime[6:8])
    return datetime.datetime(c.Year,c.Month,c.Day,hour,minute,second)
    
'''read my handmade time splits. 
returns dictionary:
            {intersection: {time: status} }
    each intersection has an entry for every second and 
    the status is a list of what 'directions' are green 
        (need to look at signals constants to decode)
DISCLAIMER -- lots of file reading heuristics specific to my file format, sorry...'''
def readSignalTimeSplit(filepath):
    signalFile = open(filepath, 'r')
    lines = signalFile.readlines()
    numLines = len(lines)
    lineCounter = 0
    signalDict = {}
    intersectionStarted = False
    timeChk = convertTimeStamp("00:00:00")
    curInter = 0
    counter2 = 0
    for line in lines:
        lineCounter = lineCounter + 1
        if lineCounter % int(numLines/10) == 0:
            print("Done", lineCounter, "/", numLines, "lines...")
        if "Intersection" in line:
            intersectionStarted = True
            arr = line.split()
            interNum = convertIntersection(int(arr[1]))
            signalDict[interNum] = {}
            curInter = interNum
            continue
        elif not intersectionStarted or "TIME" in line:
            continue
        elif line in ['\n', '\r\n'] or line == '':
            print("Done with intersection:",curInter)
            intersectionStarted = False
            timeChk = convertTimeStamp("00:00:00")
            continue
        elif not intersectionStarted:
            continue
        elif line[0] == "#":  # a comment or something
            continue
        if line[0] == "+":  #at a time split
            arr = line.split()
            duration = int(arr[0][1:])
            for dt in range(0,duration):
                if timeChk.second == 59:
                    if timeChk.minute == 59:
                        timeChk = timeChk.replace(hour=timeChk.hour+1,minute=0,second=0)
                    else:
                        timeChk = timeChk.replace(minute=timeChk.minute+1,second=0)
                else:
                    timeChk = timeChk.replace(second=timeChk.second+1)
                if len(arr) > 1:  #if nothing, means just delay
                    signalDict[curInter][timeChk] = arr[1].split(',')
                    counter2 = counter2 + 1
        else:               #at a cycle of 100 sec start => update time checkpoint
            arr = line.split()
            giventime = arr[0]
            timeChk = convertTimeStamp(giventime) #dont update, make new
    print(counter2)
    return signalDict
    
def convertDirToNum(curInter, green):
    if curInter == 1:
        if green == 'SB':
            return 2
        elif green == 'NB':
            return 6
        elif green == 'WB':
            return 4
        return 0
    elif curInter == 2:
        if green == 'SB':
            return 6
        elif green == 'NB':
            return 2
        elif green == 'WB':
            return 3
        elif green == 'EB':
            return 4
        elif green == "SBLT":
            return 1
        elif green == "NBLT":
            return 5
        return 0
    elif curInter == 3 or curInter == 4:
        if green == 'SB':
            return 2
        elif green == 'NB':
            return 6
        elif green == 'WB':
            return 3
        elif green == 'EB':
            return 4
        elif green == "SBLT":
            return 5
        return 0
    return -1
    
    
'''read my handmade time splits in format 2. 
returns dictionary:
            {intersection: {time: status} }
        the file has "intersection" then a line for time and then a line for 
        what direction is currently green
DISCLAIMER -- lots of file reading heuristics specific to my file format, sorry...'''
def readSignalTimeSplit2(filepath):
    signalFile = open(filepath, 'r')
    lines = signalFile.readlines()
    numLines = len(lines)
    lineCounter = 0
    signalDict = {}
    intersectionStarted = False
    timeChk = convertTimeStamp("00:00:00")
    curInter = 0
    counter2 = 0
    line = ''
    for next_line in lines:
        lineCounter = lineCounter + 1
        if lineCounter % int(numLines/10) == 0:
            print("Done", lineCounter, "/", numLines, "lines...")
        if next_line == '' or len(next_line.split()) == 0: #last entry is fluff
            print("Done with intersection:",curInter)
            intersectionStarted = False
            timeChk = convertTimeStamp("00:00:00")
            curInter = 0
        elif "Intersection" in line:
            intersectionStarted = True
            arr = line.split()
            interNum = int(arr[1])
            signalDict[interNum] = {}
            curInter = interNum
        elif intersectionStarted and not line[0] == '#': #line.split()[0] = time
            arr = line.split()
            giventime = arr[0]
            timeChk = convertTimeStamp(giventime) #dont update, make new
            nextarr = next_line.split()
            nextime = nextarr[0]
            nextdtime = convertTimeStamp(nextime)
            duration = int((nextdtime-timeChk).total_seconds())
            for dt in range(0,duration-1): #1 second delay seems fine
                if timeChk.second == 59:
                    if timeChk.minute == 59:
                        timeChk = timeChk.replace(hour=timeChk.hour+1,minute=0,second=0)
                    else:
                        timeChk = timeChk.replace(minute=timeChk.minute+1,second=0)
                else:
                    timeChk = timeChk.replace(second=timeChk.second+1)
                if len(arr) > 1:  #if nothing, means just delay
                    greens = arr[1].split(',')
                    newgreens = []
                    for green in greens:
                        newgreens.append(convertDirToNum(curInter, green))
                    signalDict[curInter][timeChk] = newgreens
                    counter2 = counter2 + 1
        line = next_line
    print(counter2)
    return signalDict
    
    
'''will visualize the data with the status of each light.
For validation purposes does not display frame until user enters new line.
Does every 5th frame, aka every half second. This allows for less frames. '''
def visualizeForValidation(filepath, signalFilepath=None, formattype=1):
    driver_data = dd2.data2(filepath)
    if formattype == 1:
        signalDict = readSignalTimeSplit(signalFilepath)
    elif formattype == 2:
        signalDict = readSignalTimeSplit2(signalFilepath)
    else:
        signalDict = {}
    frames = driver_data.getFrames()
    useFrames = frames[0::10]
    wrongFrames = {}
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 3
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    plt.figure(1)
    index = 0
    while (True):
        fid = useFrames[index]
        timemark = visualize_frame_and_signals(driver_data.frameDict[fid], signalDict, fid)
        command = input("Press Enter to continue, q to quit, m to mark frame for review: ")
        if 'goto' in command:
            gofid = int(command.split()[1])
            if gofid in useFrames:
                index = useFrames.index(gofid)
            else:
                print("Invalid frame")
        else:
            index = index + 1
        if command == 'q':
            break
        if 'print time' in command:
            print(timemark)
        if 'm - ' in command:  #add a message
            wrongFrames[fid] = timemark
            if len(command) > 1:
                wrongFrames[fid] = [timemark, command[1:]]
            print("fid set to:",fid)        
    print(wrongFrames)
    return wrongFrames

def visualize_frame_and_signals(frame, signalDict, fid):
    timeStr = v.vehicle(frame[list(frame.keys())[0]]).getGlobalT()
    frame_date_time = dru.convertNGSIMtimestampToDateTime(timeStr)
    #signals['green'].x/y, signals['red'].x/y ---- NOTE assumes no yellows...
    signalCordsColors = {'green':{'x':[], 'y':[] } , 'red':{'x':[],'y':[]} }
    for intersection in signalDict.keys():
        if not frame_date_time in signalDict[intersection].keys():
            mindis = 999999
            closest = None
            for date_time in signalDict[intersection]:
                dist = abs(frame_date_time.hour - date_time.hour) * 60 * 60
                dist = dist + abs(frame_date_time.minute - date_time.minute) * 60
                dist = dist + abs(frame_date_time.second - date_time.second)
                dist = dist + abs(frame_date_time.microsecond - date_time.microsecond) * 0.001
                if dist < mindis:
                    closest = date_time
                    mindis = dist
            greens = signalDict[intersection][closest]
        else:
            greens = signalDict[intersection][frame_date_time]
        cords = getCords(intersection)
        for ident in greens:
            ident = int(ident)
            if ident not in cords.keys():
                print("Not a light")
                continue #not a light, likely ped signal
            signalCordsColors['green']['x'].append(cords[ident][0])
            signalCordsColors['green']['y'].append(cords[ident][1])
        for key in cords.keys():
            if str(key) not in greens and key not in greens:
                signalCordsColors['red']['x'].append(cords[key][0])
                signalCordsColors['red']['y'].append(cords[key][1])    
    fut.plotFrame(frame, fid, signals=signalCordsColors)
    plt.clf()
    return frame_date_time

def getCords(intersectionNumber):
    if intersectionNumber == 1:
        return {2: (-10,105), 4: (40,85), 6: (15,70)}
    elif intersectionNumber == 2:
        return {1:(-10,510), 2: (20,400), 3:(40,475), 4: (-45,440), 5:(10,400), 6: (-20,500)}
    elif intersectionNumber == 3:
        return {2: (-25,1100), 3:(40,1090), 4: (-45,1060), 5:(-15,1100), 6: (10,1045)}
    elif intersectionNumber == 4:
        return {2: (-15,1600), 3:(40,1590), 4: (-45,1560), 5:(-5,1600), 6: (10,1545)}
