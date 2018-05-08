# Authors: Austin Ray, Bruno De Martino, Alex Lin
# File: sandbox.py
# ----------------
# This file exists so we can play around with different packages.
# As we develop useful functions, we will port groups of similar functions
# from sandbox.py into other package files.
#
# Note: Commenting encouraged so other people can learn from what you've done!

import collections, itertools, copy
from collections import Counter
import sys, os
import numpy, scipy, math, random
import re
import inspect
import json
from multiprocessing import Pool
from contextlib import closing
import thread
import threading
import time

print __file__
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lib import util
from lib import database
from lib import constants as c
from lib import nutrientdatabase as ndb
c.init(os.path.dirname(os.path.dirname(__file__)))
#import constraint

def printFunctionName():
    print ""
    print ""
    print "Function: " + inspect.stack()[1][3]
    print "---------"



def test_exception1():
    printFunctionName()
    var1 = "yooooo"
    myDict = {"hi": 1}
    try:
        var1 = myDict["booooo"]
    except KeyError:
        pass
    print "var1: ", var1

def test_exception2():
    printFunctionName()
    exc = Exception("spam", "eggs")
    try:
        raise exc
    except Exception as inst:
        print "inst.args: ", inst.args

def test_exception3():
    printFunctionName()
    class MyException(Exception):
        pass

    exc = Exception("I am a normal exception")
    myExc = MyException("I'm special!!!")
    try:
        raise myExc
        raise exc
    except Exception as inst:
        print "inst.args: ", inst.args
    except MyException as inst:
        print "inst.args: ", inst.args

def test_exception4():
    printFunctionName()
    class MyException(Exception):
        pass

    exc = Exception("I am a normal exception")
    myExc = MyException("I'm special!!!")
    try:
        try:
            raise exc
        except Exception as inst:
            print "inst.args1: ", inst.args
            raise myExc
    except MyException as inst:
        print "inst.args2: ", inst.args


def test_threading1():
    printFunctionName()

    def writeThreadNameToFile(threadName, message, curFile):
        filedir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "res", "threadtest")
        filename = os.path.join(filedir, threadName) + ".txt"
        with open(filename, 'w+') as f:
            f.write(message)


    # Create new threads
    threads = []
    for i in xrange(4):
        threadName = "Thread-" + str(i)
        message = "I, the " + str(i) + "th thread, am super duper awesome."

        # Create a new thread that, when its start() method is called, will
        # execute target(args)
        newThread = threading.Thread(\
            target=writeThreadNameToFile, 
            args=(threadName, message, __file__)\
        )

        # Append the new thread to the list of all threads
        # (you must keep track of all active threads)
        threads.append(newThread)


    # Start new Threads
    for t in threads:
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    print "Exiting Main Thread"

    print


def main(argv):
    print("Hello world!")
    continue

if __name__ == "__main__":
    main(sys.argv)
