# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:59:36 2016

Validation script
@author: djp42
"""


import numpy as np
from utils import constants as c
from utils import data2_class as dd2
from utils import data_util as du
from utils import vehicleclass as v
from utils import frame_util as futil
from utils import driver_util as dru
from utils import eval_util as eutil
from utils import LSTM
from sklearn.externals import joblib
from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing
import tensorflow as tf
import random
import os
import time
import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

tf.logging.set_verbosity(tf.logging.INFO)

def getUniqueCounts(data):
    typesOfY = {}
    for val in data.flat:
        if not val in typesOfY.keys():
            typesOfY[val] = 1
        else:
            typesOfY[val] = typesOfY[val] + 1
    return typesOfY

def validateDNN(test_folder, Xtrain, Ytrain, Xtest, Ytest):
    scaler = preprocessing.StandardScaler(copy=False).fit(Xtrain)
    scaler.transform(Xtrain)
    scaler.transform(Xtest)
    classifier = skflow.DNNClassifier(
        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(Xtrain),
        hidden_units=[128, 128], n_classes=3)

    
    Ytrain = [i-1 for i in Ytrain]
    Ytest = [i-1 for i in Ytest]
    num_batches = 10
    batch_size = int(len(Ytrain) / num_batches)
    XValid = Xtest[0:batch_size]
    YValid = Ytest[0:batch_size]
    '''for i in range(num_batches): # was testing out, works faster, not necessarily better
    #    classifier.partial_fit(Xtrain[i*batch_size:(i+1)*batch_size], 
    #                                  Ytrain[i*batch_size:(i+1)*batch_size])
    #    validation_score = metrics.accuracy_score(YValid, classifier.predict(XValid))
    #    print("Validation accuracy: %f" % validation_score)
        classifier.partial_fit(Xtrain, Ytrain, batch_size = batch_size)
        #classifier.fit(Xtrain, Ytrain)
        predictions = classifier.predict(Xtest)
        score = metrics.accuracy_score(Ytest, predictions)
        print("Validation accuracy: %f" % score)
    '''
    start = time.clock()
    classifier.fit(Xtrain, Ytrain)
    end = time.clock()
    print(end - start)
    predictions = classifier.predict(Xtest)
    probs = classifier.predict_proba(Xtest)
    np.savetxt(test_folder + "DNN_validation_p_dist", probs)
    score = metrics.accuracy_score(Ytest, predictions)
    print("Accuracy: %f" % score)

def doLSTM(filepath, testID, test_folder, numEpochs = 2):
    test_folder = test_folder + "/LSTM_formatted/"
    Xtrain, Ytrain, __, ___ = loadLSTMValidationSet(filepath, test_folder + '0/')
    for i in [1,2,3]:
      X, Y, _, __ = loadLSTMValidationSet(filepath, test_folder + str(i) + '/')
      Xtrain = np.concatenate((Xtrain, X))
      Ytrain = np.concatenate((Ytrain, Y))
    _1, _2, Xtest, Ytest = loadLSTMValidationSet(filepath, test_folder + '4/')
    
    typesOfY = getUniqueCounts(Ytrain)
    print(typesOfY)
    typesOfY = getUniqueCounts(Ytest)
    print(typesOfY)
    
        
    #scaler = preprocessing.StandardScaler(copy=False).fit(Xtrain)
    #for i in range(1, len(Xtrain[0,0,:])):
    # scaler = preprocessing.StandardScaler(copy=False).fit(Xtrain[:,:,i])
    # scaler.transform(Xtrain[:,:,i])
    # scaler.transform(Xtest[:,:,i])
    Xtrain, Xtest = du.normalize_wrapper(Xtrain, Xtest)
    validateLSTM(test_folder, Xtrain, Ytrain, Xtest, Ytest, numEpochs)

    
def validateLSTM(test_folder, Xtrain, Ytrain, Xtest, Ytest, numEpochs=2):
    print(Xtrain.shape)
    print(Ytrain.shape)
    print(Xtest.shape)
    print(Ytest.shape)
    for model in ["LSTM_128x2"]:#, "test", "test1", "test2", "test3", "test4"]:
        print("\n")
        print("===========================")
        print("Starting:", model)
        p_dists, timeTrain, timePred, all_tests_x, all_tests_y = LSTM.run_LSTM((Xtrain,Ytrain), (Xtest, Ytest), model=model,
                             save_path=None, numEpochs=numEpochs)
        np.savetxt(test_folder + "Ypred_LSTM", np.array(p_dists))
        score = 0
        numWrong = 0
        #all_tests_x = np.reshape(Xtest, (Xtest.shape[0] * Xtest.shape[1], Xtest.shape[2]))
        #all_tests_y = np.reshape(Ytest, (Ytest.shape[0] * Ytest.shape[1], Ytest.shape[2]))
        print(all_tests_x.shape)
        print(all_tests_y.shape)
        typesOfY = {}
        for val in all_tests_y.flat:
          if not val in typesOfY.keys():
            typesOfY[val] = 1
          else:
            typesOfY[val] = typesOfY[val] + 1
        print(typesOfY)
        
        print("Total predictions:", len(all_tests_y))
        print(p_dists.shape)
        for i in range(0, len(all_tests_y)-1):
            if i > len(p_dists):
                break
            actual = all_tests_y[i]
            p_right = p_dists[i][int(actual)]
            score += 1 - p_right
            if not p_right == max(p_dists[i]):
                numWrong += 1
        print("Score:", score)
        print("Num wrong:", numWrong)
        sumP0 = sum(p_dists[:,0])
        print("Sum of P for 0:", sumP0)
    return
    
def validateSVM(Xtrain, Ytrain, Xtest, Ytest):
    return
    
def validateBN(Xtrain, Ytrain, Xtest, Ytest):
    return
    
def loadLSTMValidationSet(filepath, test_folder):
    features = np.load(test_folder + "featureSetv2.npy")
    targets = np.load(test_folder + "targetSetv2.npy", )
    #fids = np.load(fullpath + "Fids")
    num = int(len(targets)/2)
    #return features[:num], targets[:num], features[num:], targets[num:]
    return features, targets, features, targets
        
def loadValidationSet(filepath, test_folder):
    features = np.loadtxt(test_folder + "featureSet")
    targets = np.loadtxt(test_folder + "targetSet", )
    #fids = np.load(fullpath + "Fids")
    num = int(len(targets)/2)
    return features[:num], targets[:num], features[num:], targets[num:]
    
def doOther(filepath, testID, test_folder, model_type="DNN"):
    test_folder = test_folder + "/0/"
    Xtrain, Ytrain, Xtest, Ytest = loadValidationSet(filepath, test_folder)
    typesOfY = getUniqueCounts(Ytrain)
    print(typesOfY)
    typesOfY = getUniqueCounts(Ytest)
    print(typesOfY)
    if model_type == "DNN":
        validateDNN(test_folder, Xtrain, Ytrain.astype(int), Xtest, Ytest.astype(int))    

def doBaselines():
    
    return

def run():
    filepath = dru.findPathForFile("AUGv2_trajectories-0830am-0900am.txt")
    testID = "Test1"
    test_folder = c.PATH_TO_RESULTS + str(testID)
    doLSTM(filepath, testID, test_folder, numEpochs=2)
    #doOther(filepath, testID, test_folder, model_type="DNN")

def testResults():
  scores = {}  #model: (score, numWrong)
  allButBN = ["SVM", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"]
  LSTMs = ["LSTM_128x2", "LSTM_128x3", "LSTM_256x2"]
  for testnum in [1,1.1,2,3,4]:
    print("Doing test:", testnum)
    testID = "Test" + str(testnum)
    test_folder = c.PATH_TO_RESULTS + str(testID) + os.sep
    for model in LSTMs: 
        if "LSTM" in model: 
            scores[model] = eutil.score(test_folder+"LSTM_formatted/", model)
        else:
            scores[model] = eutil.score(test_folder, model)
    print(scores)
 
def plotResults():
  #scores = {}  #model: (score, numWrong)
  #allModels = ["Marginal", "Conditional", "SVM", "DNN", "BN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"] 
  allModels = ["Marginal", "Conditional", "SVM", "DNN", "BN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"] 
  allButBN = ["SVM", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"]
  LSTMs = ["LSTM_128x2", "LSTM_128x3", "LSTM_256x2"]
  testset = ["SVM"]
  baselines = ["Marginal", "Conditional"] 
  models = allModels
  for testnum in [3,4]:#1,2,3,4,1.1]:
      print("Doing test:", testnum)
      testID = "Test" + str(testnum)
      test_folder = c.PATH_TO_RESULTS + str(testID) + os.sep
      #eutil.doAllPlotsForTest(test_folder, models, testID, limit_by=None, limits=None)
      eutil.doAllPlotsForTest(test_folder, models, testID, limit_by="dist", limits=">0")
    
plotResults()
#testResults()   
#run()
