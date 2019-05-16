# -*- coding: utf-8 -*-
"""
Created on Mon May 23 18:11:36 2016

@author: Derek
"""

from utils import frame_util as futil
import numpy as np
import os
from scipy import sparse
from utils import constants
from utils import vehicleclass as v
import time
import random


numUsing = 0 # 0 to use all

'''Returns the startX and startY for all merge vehicles'''
def getStartVals(filename):
    filepath = makeFullPath(filename, '-mergerStartTrajectories.txt')
    A = np.loadtxt(filepath)
    return A[:,[constants.LocalX,constants.LocalY]]

'''Removes the entry corresponding to this vid from the grid'''
def removeIDfromGrid(Frame, VID, Grid, compressed):
    #vehicleTraj = Frame[VID]
    if VID not in Frame:
        return Grid
    vehicleData = Frame[VID]
    veh = v.vehicle(vehicleData, compressed)
    xpos = veh.x
    ypos = veh.y
    #[xpos,ypos]=vehicleTraj[[constants.LocalX,constants.LocalY]]
    indexX, indexY = futil.GetGridIndices(xpos,ypos)
    if futil.InGridBounds(veh.getX(), veh.getY()):
        if Grid[indexX][indexY][0] > 1:
            Grid[indexX][indexY][0] = Grid[indexX][indexY][0]-1
            #recalculate velocities?
        else:
            Grid[indexX][indexY] = [0]*len(veh.GridInfo)
    return Grid

'''Called for each merging vehicle, gets all the input data.'''
def getXInner(row,dictOfGrids, initPos, dictOfFrames, compressed):
    VID = row[0]
    start = row[1]
    X_for_id = np.array([]) #This will have numFrames rows and sizeGrid+1 columns
    for frame in range(row[1],row[2]):
        t_elapsed = frame-start        
        #grid = dictOfGrids[frame]
        #grid = removeIDfromGrid(dictOfFrames[frame],VID,grid, compressed)
        start_grid = dictOfGrids[start]
        start_grid = removeIDfromGrid(dictOfFrames[start],VID,start_grid, compressed)
        
        #grid2 = dictOfGrids[frame-10]
        #grid2 = removeIDfromGrid(dictOfFrames[frame-10],VID,grid2)
        
        init_grid_avg = futil.getGridMeans(dictOfGrids[frame])
        
        #additional = np.append(t_elapsed, grid2.flatten())
        additional = np.append(initPos,t_elapsed)
        #additional = t_elapsed        
        Xrow = np.append(additional,start_grid.flatten())
        #Xrow = np.append(additional,init_grid_avg)
        #Xrow = additional
        Xrow.shape = (1,len(Xrow))
        if X_for_id.shape == (0,):
            X_for_id = Xrow
        else:
            X_for_id = np.append(X_for_id,Xrow,axis=0)
    # Xi = np.append(Xi, initPos)
    # Xi.shape = (1,len(Xi))
    return (X_for_id)

'''Gets ground truths for each merge vehicle'''
def getYInner(row, dictOfFrames, predict, compressed):
    y_for_id = np.array([]) #this will have numFrames rows and 1 column
    for frame in range(row[1],row[2]):
        veh = v.vehicle(dictOfFrames[frame], compressed)
        if predict == 'Y':
            yrow = veh.y
        elif predict == 'X':
            yrow = veh.x
        else:
            print("ERROR: invalid prediction request:", predict)
            return None
        y_for_id = np.append(y_for_id,yrow)
    return y_for_id

'''AVOID---This probably uses a significant amount of memory'''
def append(orig, add, axisNum=0):
    if orig.shape == (0,):
        orig = add
    else:
        orig = np.append(orig,add,axis=axisNum)
    return orig

#get the training examples
def getX(filename, trainIDs, testIDs, mean_centered):
    #filename="res/101_trajectories/aug_trajectories-0750am-0805am.txt"
    path = os.getcwd()+'/'
    compressed = 'compressed' in filename
    frameDict = futil.LoadDictFromTxt(path+filename, 'frame')
    print("Gotten frameDict",time.ctime())
    dictOfGrids = futil.GetGridsFromFrameDict(frameDict, mean_centered, compressed)
    print("Gotten dictOfGrids",time.ctime())
    #filepath = makePathMR(filename, '-mergerMinRanges')
    filepath = makeFullPath(filename, '-mergerRanges.txt')
    MR = np.loadtxt(filepath, dtype='int')
    '''MR=MergeRanges. MR[:,0]=merge ids, MR[:,1]=start frame, MR[:,2] = end'''
    print ("Done loading in getX", time.ctime())
    start = getStartVals(filename)    
    Xtrain = np.array([])   #will have numTrain*numFrames rows and size(grid)+1 columns
    Xtest = np.array([])
    it = 0
    trainEmpty = True
    testEmpty = True
    if not numUsing == 0:
        MR = MR[:numUsing]
    for row in MR:
        thisStart = start[it]
        XVID = sparse.csr_matrix(np.ascontiguousarray(getXInner(row, dictOfGrids,thisStart,frameDict, compressed)))
        if row[0] in trainIDs:
            if  trainEmpty == True:
                Xtrain = XVID
                trainEmpty = False
            else:
                Xtrain = sparse.vstack((Xtrain,XVID))#,axis=0)
            print("Finished getting X data for Merger with VID:",row[0]," and it is a training example", time.ctime())
        else:
            if testEmpty == True:
                Xtest = XVID
                testEmpty = False
            else:
                Xtest = sparse.vstack((Xtest,XVID))#np.append(Xtest,XVID,axis=0)
            print("Finished getting X data for Merger with VID:",row[0]," and it is a test example")
        it += 1
        print(Xtrain.shape)
    return Xtrain, Xtest

def getXClusters(filename, trainIDs, testIDs, mean_centered, clusterIDs0, clusterIDs1, clusterIDs2):
    #filename="res/101_trajectories/aug_trajectories-0750am-0805am.txt"
    path = os.getcwd()+'/'
    compressed = 'compressed' in filename
    frameDict = futil.LoadDictFromTxt(path+filename, 'frame')
    print("Gotten frameDict",time.ctime())
    dictOfGrids = futil.GetGridsFromFrameDict(frameDict, mean_centered, compressed)
    print("Gotten dictOfGrids",time.ctime())
    #filepath = makePathMR(filename, '-mergerMinRanges')
    filepath = makeFullPath(filename, '-mergerRanges.txt')
    MR = np.loadtxt(filepath, dtype='int')
    '''MR=MergeRanges. MR[:,0]=merge ids, MR[:,1]=start frame, MR[:,2] = end'''
    print ("Done loading in getX", time.ctime())
    start = getStartVals(filename)    
    Xtrain1 = np.array([])
    Xtrain2 = np.array([])
    Xtrain0 = np.array([])   #will have numTrain*numFrames rows and size(grid)+1 columns
    Xtest1 = np.array([])
    Xtest2 = np.array([])
    Xtest0 = np.array([])
    it = 0
    trainEmpty = [True]*3
    testEmpty = [True]*3
    if not numUsing == 0:
        MR = MR[:numUsing]
    for row in MR:
        thisStart = start[it]
        XVID = sparse.csr_matrix(np.ascontiguousarray(getXInner(row, dictOfGrids,thisStart,frameDict, compressed)))
        if row[0] in trainIDs:
            if row[0] in clusterIDs0:
                if trainEmpty[0] == True:
                    Xtrain0 = XVID
                    trainEmpty[0] = False
                else:
                    Xtrain0 = sparse.vstack((Xtrain0,XVID))#,axis=0)
            elif row[0] in clusterIDs1:
                if trainEmpty[1] == True:
                    Xtrain1 = XVID
                    trainEmpty[1] = False
                else:
                    Xtrain1 = sparse.vstack((Xtrain1,XVID))#,axis=0)
            elif row[0] in clusterIDs2:
                if trainEmpty[2] == True:
                    Xtrain2 = XVID
                    trainEmpty[2] = False
                else:
                    Xtrain2 = sparse.vstack((Xtrain2,XVID))#,axis=0)
            print("Finished getting X data for Merger with VID:",row[0]," and it is a training example", time.ctime())
        else:
            if row[0] in clusterIDs0:
                if testEmpty[0] == True:
                    Xtest0 = XVID
                    testEmpty[0] = False
                else:
                    Xtest0 = sparse.vstack((Xtest0,XVID))#,axis=0)
            elif row[0] in clusterIDs1:
                if testEmpty[1] == True:
                    Xtest1 = XVID
                    testEmpty[1] = False
                else:
                    Xtest1 = sparse.vstack((Xtest1,XVID))#,axis=0)
            elif row[0] in clusterIDs2:
                if testEmpty[2] == True:
                    Xtest2 = XVID
                    testEmpty[2] = False
                else:
                    Xtest2 = sparse.vstack((Xtest2,XVID))#,axis=0)
                #Xtest = sparse.vstack((Xtest,XVID))#np.append(Xtest,XVID,axis=0)
            print("Finished getting X data for Merger with VID:",row[0]," and it is a test example")
        it += 1
    return Xtrain0, Xtrain1, Xtrain2, Xtest0, Xtest1, Xtest2
    
def getYClusters(filename, trainIDs, testIDs, predict, clusterIDs0, clusterIDs1, clusterIDs2):
    path = os.getcwd()+'/'
    IDDict = futil.LoadDictFromTxt(path+filename, 'vid')
    compressed = 'compressed' in filename
    #filepath = makePathMR(filename, '-mergerMinRanges')
    filepath = makeFullPath(filename, '-mergerRanges.txt')
    MR = np.loadtxt(filepath, dtype='int')
    Ytrain0 = np.array([]) 
    Ytrain1 = np.array([])
    Ytrain2 = np.array([])    #will have numTrain*numFrames rows and 1 column
    Ytest0 = np.array([])
    Ytest1 = np.array([]) 
    Ytest2 = np.array([])
    if not numUsing == 0:
        MR = MR[:numUsing]
    for row in MR:
        YVID = np.ascontiguousarray(getYInner(row,IDDict[row[0]], predict, compressed))
        if row[0] in trainIDs:
            if row[0] in clusterIDs0:
                Ytrain0=append(Ytrain0,YVID)
            elif row[0] in clusterIDs1:
                Ytrain1=append(Ytrain1,YVID)
            elif row[0] in clusterIDs2:
                Ytrain2=append(Ytrain2,YVID)
             #uses append because Y is small in memory
            print("Finished getting Y data for Merger with VID:",row[0]," and it is a training example")
        else:
            if row[0] in clusterIDs0:
                Ytest0=append(Ytest0,YVID)
            elif row[0] in clusterIDs1:
                Ytest1=append(Ytest1,YVID)
            elif row[0] in clusterIDs2:
                Ytest2=append(Ytest2,YVID)
            #Ytest=append(Ytest,YVID)
            print("Finished getting Y data for Merger with VID:",row[0]," and it is a test example")
    return np.ascontiguousarray(Ytrain0),np.ascontiguousarray(Ytrain1), np.ascontiguousarray(Ytrain2), np.ascontiguousarray(Ytest0), np.ascontiguousarray(Ytest1), np.ascontiguousarray(Ytest2)

def getY(filename, trainIDs, testIDs, predict):
    path = os.getcwd()+'/'
    IDDict = futil.LoadDictFromTxt(path+filename, 'vid')
    compressed = 'compressed' in filename
    #filepath = makePathMR(filename, '-mergerMinRanges')
    filepath = makeFullPath(filename, '-mergerRanges.txt')
    MR = np.loadtxt(filepath, dtype='int')
    Ytrain = np.array([])    #will have numTrain*numFrames rows and 1 column
    Ytest = np.array([])
    if not numUsing == 0:
        MR = MR[:numUsing]
    for row in MR:
        YVID = np.ascontiguousarray(getYInner(row,IDDict[row[0]], predict, compressed))
        if row[0] in trainIDs:
            Ytrain=append(Ytrain,YVID) #uses append because Y is small in memory
            print("Finished getting Y data for Merger with VID:",row[0]," and it is a training example")
        else:
            Ytest=append(Ytest,YVID)
            print("Finished getting Y data for Merger with VID:",row[0]," and it is a test example")
    return np.ascontiguousarray(Ytrain), np.ascontiguousarray(Ytest)
    
def makePathMR(filename, end):
    path = os.getcwd()+'/'
    a = len('aug_trajectories-0750am-0805am.txt')
    return path+filename[:-a]+filename[(-a+4):-4]+end+'.txt'

def getSpan(filename):
    return filename[-17:][:-4]
    
def makePathToTrajectories(filename):
    outerFolder = filename[4:-35]
    path1 = os.getcwd() + '/res' + '/' + outerFolder + '/' 
    path = path1 + getSpan(filename) + '/'  
    if not os.path.exists(path):
        os.makedirs(path)  
    return path

def makeFullPath(filename, end=''):
    path = makePathToTrajectories(filename)
    return path + end

def makeTrainTestData(filename, portionTrain, seed=None):
    # example filename="res/101_trajectories/aug_trajectories-0750am-0805am.txt"
    #filepath = makePathMR(filename, '-mergerMinRanges')
    filepath = makeFullPath(filename, '-mergerRanges.txt')
    MR = np.loadtxt(filepath, dtype='int')
    traintest = [[],[]]
    random.seed(seed)
    if not numUsing == 0:
        MR = MR[:numUsing]
    for row in MR:
        traintest[random.random() > portionTrain].append(row[0])
    train = traintest[0]
    test = traintest[1]
    filepathTrain = makeFullPath(filename, 'trainIDs.txt')
    filepathTest = makeFullPath(filename, 'testIDs.txt')
    np.savetxt(filepathTrain, train)
    np.savetxt(filepathTest, test)
    return train, test

def loadTrainTestData(filename):
    filepathTrain = makeFullPath(filename, 'trainIDs.txt')
    filepathTest = makeFullPath(filename, 'testIDs.txt')
    trainIDs = np.loadtxt(filepathTrain)
    testIDs = np.loadtxt(filepathTest)
    return trainIDs, testIDs

def saveSparse(filepath, X):
    if X.shape == (0,):
        return
    data = X.data
    indices = X.indices
    indptr = X.indptr
    np.savetxt(filepath + '-data',data)
    np.savetxt(filepath + '-indices',indices)
    np.savetxt(filepath + '-indptr',indptr)

def loadSparse(filepath):
    data = np.loadtxt(filepath + '-data')
    indices = np.loadtxt(filepath + '-indices')
    indptr = np.loadtxt(filepath + '-indptr')
    return sparse.csr_matrix((data,indices,indptr))
    
def saveExampleData(filename,Xtrain,ytrain,Xtest,ytest, mean_centered, predict):
    filepath_Xtrain = makeFullPath(filename, '-Xtrain'+str(mean_centered))
    saveSparse(filepath_Xtrain, Xtrain)
    filepath_ytrain = makeFullPath(filename, '-ytrain'+str(mean_centered))
    np.savetxt(filepath_ytrain, ytrain)
    filepath_Xtest = makeFullPath(filename, '-Xtest'+str(mean_centered)+predict)
    saveSparse(filepath_Xtest, Xtest)
    filepath_ytest = makeFullPath(filename, '-ytest'+str(mean_centered)+predict)
    np.savetxt(filepath_ytest, ytest)

def readExampleData(filename, mean_centered, predict):
    filepath_Xtrain = makeFullPath(filename, '-Xtrain'+str(mean_centered))
    Xtrain = loadSparse(filepath_Xtrain)
    print("Xtrain loaded.",time.ctime())
    filepath_Xtest = makeFullPath(filename, '-Xtest'+str(mean_centered))
    Xtest = loadSparse(filepath_Xtest)
    print("Xtest loaded.",time.ctime())
    filepath_ytrain = makeFullPath(filename, '-ytrain'+str(mean_centered)+predict)
    ytrain = np.loadtxt(filepath_ytrain)
    print("ytrain loaded.",time.ctime())
    filepath_ytest = makeFullPath(filename, '-ytest'+str(mean_centered)+predict)
    ytest = np.loadtxt(filepath_ytest)
    print("ytest loaded.",time.ctime())
    return Xtrain, ytrain, Xtest, ytest
    
def saveExampleDataClusters(filename, Xtrain0, Xtrain1, Xtrain2, ytrain0, ytrain1, ytrain2,
                                       Xtest0, Xtest1, Xtest2, ytest0, ytest1, ytest2,
                                       mean_centered, predict):
    filepath_Xtrain = makeFullPath(filename, '-Xtrain0'+str(mean_centered))
    saveSparse(filepath_Xtrain, Xtrain0)
    filepath_Xtrain = makeFullPath(filename, '-Xtrain1'+str(mean_centered))
    saveSparse(filepath_Xtrain, Xtrain1)
    filepath_Xtrain = makeFullPath(filename, '-Xtrain2'+str(mean_centered))
    saveSparse(filepath_Xtrain, Xtrain2)
    
    filepath_ytrain = makeFullPath(filename, '-ytrain0'+str(mean_centered))
    np.savetxt(filepath_ytrain, ytrain0)
    filepath_ytrain = makeFullPath(filename, '-ytrain1'+str(mean_centered))
    np.savetxt(filepath_ytrain, ytrain1)
    filepath_ytrain = makeFullPath(filename, '-ytrain2'+str(mean_centered))
    np.savetxt(filepath_ytrain, ytrain2)
    
    filepath_Xtest = makeFullPath(filename, '-Xtest0'+str(mean_centered)+predict)
    saveSparse(filepath_Xtest, Xtest0)
    filepath_Xtest = makeFullPath(filename, '-Xtest1'+str(mean_centered)+predict)
    saveSparse(filepath_Xtest, Xtest1)
    filepath_Xtest = makeFullPath(filename, '-Xtest2'+str(mean_centered)+predict)
    saveSparse(filepath_Xtest, Xtest2)
    
    filepath_ytest = makeFullPath(filename, '-ytest0'+str(mean_centered)+predict)
    np.savetxt(filepath_ytest, ytest0)
    filepath_ytest = makeFullPath(filename, '-ytest1'+str(mean_centered)+predict)
    np.savetxt(filepath_ytest, ytest1)
    filepath_ytest = makeFullPath(filename, '-ytest2'+str(mean_centered)+predict)
    np.savetxt(filepath_ytest, ytest2)
    
    
def readExampleDataClusters(filename, mean_centered, predict):
    filepath_Xtrain0 = makeFullPath(filename, '-Xtrain0'+str(mean_centered))
    Xtrain0 = loadSparse(filepath_Xtrain0)
    filepath_Xtrain1 = makeFullPath(filename, '-Xtrain1'+str(mean_centered))
    Xtrain1 = loadSparse(filepath_Xtrain1)
    filepath_Xtrain2 = makeFullPath(filename, '-Xtrain2'+str(mean_centered))
    Xtrain2 = loadSparse(filepath_Xtrain2)
    print("Xtrain loaded.",time.ctime())
    
    filepath_Xtest0 = makeFullPath(filename, '-Xtest0'+str(mean_centered))
    Xtest0 = loadSparse(filepath_Xtest0)
    filepath_Xtest1 = makeFullPath(filename, '-Xtest1'+str(mean_centered))
    Xtest1 = loadSparse(filepath_Xtest1)
    filepath_Xtest2 = makeFullPath(filename, '-Xtest2'+str(mean_centered))
    Xtest2 = loadSparse(filepath_Xtest2)
    print("Xtest loaded.",time.ctime())
    
    filepath_ytrain0 = makeFullPath(filename, '-ytrain0'+str(mean_centered)+predict)
    ytrain0 = np.loadtxt(filepath_ytrain0)
    filepath_ytrain1 = makeFullPath(filename, '-ytrain1'+str(mean_centered)+predict)
    ytrain1 = np.loadtxt(filepath_ytrain1)
    filepath_ytrain2 = makeFullPath(filename, '-ytrain2'+str(mean_centered)+predict)
    ytrain2 = np.loadtxt(filepath_ytrain2)
    print("ytrain loaded.",time.ctime())
    
    filepath_ytest0 = makeFullPath(filename, '-ytest0'+str(mean_centered)+predict)
    ytest0 = np.loadtxt(filepath_ytest0)
    filepath_ytest1 = makeFullPath(filename, '-ytest1'+str(mean_centered)+predict)
    ytest1 = np.loadtxt(filepath_ytest1)
    filepath_ytest2 = makeFullPath(filename, '-ytest2'+str(mean_centered)+predict)
    ytest2 = np.loadtxt(filepath_ytest2)
    print("ytest loaded.",time.ctime())
    return Xtrain0, Xtrain1, Xtrain2, ytrain0, ytrain1, ytrain2, Xtest0, Xtest1, Xtest2, ytest0, ytest1, ytest2
    
    
    
    
    