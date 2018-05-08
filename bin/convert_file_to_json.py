##
# File: /bin/convert_file_to_json.py
# -------------------------------------
#
##
import collections, itertools, copy, Queue
import numpy, scipy, math, random
import os, sys, time, importlib
import tokenize, re, string
import json, unicodedata
import thread

from lib import util
from lib import constants as c

def main(argv):
	fileName = filter(lambda s: "file=" in s, argv)[0]
	fileName = fileName.split("=")[-1]
	filePath = os.path.join(c.PATH_TO_ROOT, fileName.lstrip("/"))

	lines = None
	with open(filePath, 'r') as f:
		lines = f.read().split("\n")
		lines = filter(lambda s: s.strip() != "", lines)

	lineDict = {"list": lines}

	jsonFileName = filePath.split("/")[-1]
	jsonFileName = ".".join(jsonFileName.split(".")[:-1]) + ".json"
	jsonFilePath = os.path.join(os.path.dirname(filePath), jsonFileName)

	util.dumpJSONDict(jsonFilePath, lineDict)