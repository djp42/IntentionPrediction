##
# File: /lib/util.py
# ------------------
# Commonly used functions.
##

import collections, itertools, copy
import numpy, scipy, math, random
import os, sys, time, importlib
import tokenize, re, string
import json, unicodedata
#import lib.constants as c
from utils import constants as c

##
# Function: loadJSONDict
# ----------------------
# Loads a JSON file into a python dictionary and returns that dictionary.
#
##
def loadJSONDict(jsonFilePath):

    # Read in the JSON file containing data
    fullJsonString = None
    with open(jsonFilePath, 'r') as f:
        fullJsonString = f.read().encode('ascii', errors='ignore')

    # Read the JSON file in as a dictionary
    print(f)
    d = json.JSONDecoder()
    returnDict = d.decode(fullJsonString)

    return returnDict

##
# Function: loadJSONDicts
# -----------------------
# Loads multiple JSON files into python dictionaries and returns 
# a list of those dictionaries.
##
def loadJSONDicts(jsonFilePaths):
    return [loadJSONDict(f) for f in jsonFilePaths]

##
# Function: dumpDictToJSONFile
# ----------------------------
# Writes a dictionary to a JSON file with the full path 'fullFilePath'
##
def dumpJSONDict(fullFilePath, dict2dump):
    jsonDatabase = json.dumps(dict2dump, sort_keys=True, indent=4)
    with open(fullFilePath, "w") as f:
        f.write(jsonDatabase)

##
# Function: naivelyMergeDicts
# ---------------------------
# Merges a list of dictionaries into one dictionary, not caring which
# value it keeps for keys that have multiple values.
##
def naivelyMergeDicts(listOfDicts):
    masterList = []
    for d in listOfDicts:
        masterList += d.items()
    return dict(masterList)

##
# Function: greedyMergeDicts
# ---------------------------
# Merges a list of dictionaries into one dictionary, not caring which
# value it keeps for keys that have multiple values.
##
def greedyMergeDicts(listOfDicts):
    masterDict = collections.defaultdict(list)
    for d in listOfDicts:
        for k, v in d.items():
            masterDict[k].append(v)
    return dict(masterDict)
##
# Function: listFilesWtihSuffix
# -----------------------------
# Returns a list of full file paths for all files whose names end in suffixStr 
# and are in the specified directory.
#
# Example suffixStr's are ".json", ".txt", ".py"
##
def listFilesWithSuffix(directoryPath, suffixStr):
    allFiles = os.listdir(directoryPath)
    return [os.path.join(directoryPath, f) for f in allFiles if f.endswith(suffixStr)]

##
# Function: deleteDuplicates
# --------------------------
# Does not change the argument given. Returns a version of the list given
# with only the first appearance of all duplicate elements saved.
##
def deleteDuplicates(li):
    new_li = []
    for elem in li:
        if elem not in new_li:
            new_li.append(elem)
    return new_li

##
# Function: deleteDuplicatesBy
# ----------------------------
# Does not change the argument given. Returns a version of the list given
# with only the first appearance of all duplicate elements saved.
#
# Duplicates are decided by the lambda function given.
##
def deleteDuplicatesBy(li, duplicatesQ):
    new_li = []
    for e1 in li:
        shouldAppend = True
        for e2 in new_li:
            if duplicatesQ(e1, e2):
                shouldAppend = False
                break
        if shouldAppend:
            new_li.append(e1)
    return new_li

  
##
# Function: string_appendDateAndTime
# ----------------------------------
# Return a string with the current date and time appended.
##
def string_appendDateAndTime(s):
    return "_".join([s, time.strftime("%m-%d-%Y"), time.strftime("%Hh-%Mm-%Ss")])

##
# Function: hasDeepKey
# --------------------
# Dive into a dictionary, recursively asking for values corresponding to a
# list of keys until you reach a non-dictionary-type value or you reach a
# sub-dictionary that doesn't have the current key in the key list.
#
# Example:
#   myDict = {"a": {"b": {"c": 1}, "d": 9}, "e": {"f": 3}}
#   hasDeepKey(myDict, ["a", "b", "c"]) returns True
#   hasDeepKey(myDict, ["a", "d"]) returns True
#   hasDeepKey(myDict, ["a", "b", "h"]) returns False
##
def hasDeepKey(myDict, keyList):
    curDict = myDict
    for key in keyList:
        if (type(curDict) is not dict) or (key not in curDict):
            return False
        curDict = curDict[key]
    return True

