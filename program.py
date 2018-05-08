# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:45:53 2016

Similar to program.py except made specifically for use with Julia and Bayes Nets,
as well as the machine learning

run with:
$ nohup python3 -u program.py | ts | tee ./<logfile>
Change the model in run() function

@author: LordPhillips
"""

import numpy as np
from collections import defaultdict
#import matplotlib.pyplot as plt
#import seaborn as sns
#from IPython import display
from lib import constants as c
from lib import data2_class as dd2
from lib import data_util as du
from lib import vehicleclass as v
from lib import frame_util as futil
from lib import driver_util as dru
from lib import score_util as sutil
#from lib import eval_util as eutil
#from lib import goal_processing as goals
#from lib import signals_util as su
from lib import LSTM
from sklearn.externals import joblib
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from collections import Counter
import tensorflow as tf
import tensorflow.contrib.learn as skflow

import random
import os
import time
import sys
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

'''Depricated
ending = 'trajectories-0830am-0900am.txt' #lankershim
filename = 'AUGv2_'+ending 
features = 'FEATURES'+ending
folder = 'Lankershim'
compressed = False'''


def reduceFids(fids):
    fids = sorted(list(fids))
    start_at = 500
    end_at = len(fids)-1000
    less_fids = fids[start_at:end_at]
    least_fids = [fid for fid in less_fids if fid % 5 == 0] #get every fifth
    #least_fids = [fid for fid in less_fids if fid % 10 in [0,3,7]]
    print("Reduced num frames from", len(fids), "to", len(least_fids))
    return least_fids

#return 2-dimensional array of numSplits rows and int(total / numSplits) columns
#remainder is discarded
def splitFids(numSplits, fids):
    fids = np.array(sorted(list(fids)))
    return np.array_split(fids, numSplits)

#def one_hot_encode_move(move):
#    move_enc = [0,0,0]
#    move_enc[move-1] = 1
#    return move_enc
    
def countWrong(Ytest, Ypred):
    wrong = 0
    total = len(Ytest)
    for i in range(0, total):
        if not Ytest[i] == Ypred[i]:
            wrong = wrong + 1
    print("Number wrong predictions:", wrong)
    print("Total predictions:", total)
    return wrong

def myScore(Ytest, Ypred, prob):
    score = 0
    total = len(Ytest)
    if not prob:
        return countWrong(Ytest, Ypred)
    else:
        for i in range(0, total):
            score += (1 - Ypred[i,Ytest[i]])
    print("Score:", score)
    return score

# returns dict with:
# key = i in range(0, numSplits), value = array(nsamples/5, nfeatures)
def load_fids(load_folder):
    allFids = []
    secFids = {}
    for subdir, dirs, files in os.walk(load_folder):
        for sub_folder in dirs:
            if len(sub_folder) > 1:
                continue
            i = int(sub_folder)
            fullpath = load_folder + sub_folder + '/'
            secFids[i] = np.loadtxt(fullpath + "Fids")
            allFids.extend(secFids[i])
    return secFids, allFids

def deleteZeroMoves(targets, features):
    print("initial length:", len(targets))
    numfound = 0
    toDelete = []
    for j in range(0, len(targets)):
        t = int(targets[j])
        if t == 0:
            toDelete.append(j)
            numfound = numfound + 1
    targets = np.delete(targets, (toDelete), axis=0)
    features = np.delete(features, (toDelete), axis=0)
    print("final length:", len(targets))
    print("found:", numfound)
    return targets, features

def generateFeaturesAndTargets(filepath, test_folder, portionTrain, load=None,
                               lanetype=True, history=False, histFs = [5,10,20,30], 
                               traffic = False, numFramesToFind=20):
    LSTMFolder = test_folder + "LSTM_formatted/"
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(LSTMFolder):
        os.makedirs(LSTMFolder)
    bool_peach=False
    if "peach" in filepath: bool_peach=True
    driver_data = dd2.data2(filepath)
    numSplits = int(1/(1-portionTrain))
    fids = reduceFids(driver_data.getFrames())
    sectionedFids = splitFids(numSplits, fids)
    s = os.sep
    for i in range(0, numSplits):
        if not os.path.exists(test_folder + str(i) + s):
            os.makedirs(test_folder + str(i) + s)
        if not os.path.exists(LSTMFolder + str(i) + s):
            os.makedirs(LSTMFolder + str(i) + s)
        np.savetxt(test_folder + str(i) + s + "Fids", np.array(sectionedFids[i]))
    allFeatures = {}  
    allTargets = {}
    for i in range(0, numSplits):
        print("Starting getting features and targets for split num:", i)
        allFeatures[i] = np.ascontiguousarray(driver_data.getFeatureVectors(sectionedFids[i], 
                                              lanetype, history, histFs, traffic, bool_peach))
        allTargets[i] = np.ascontiguousarray(driver_data.getTargets(sectionedFids[i]))
        print("Done getting features and targets split num:", i)
        # remove features where the move is a 0 (at the edge of map)
        allTargets[i], allFeatures[i] = deleteZeroMoves(allTargets[i], allFeatures[i])
        np.savetxt(test_folder + str(i) + s + "featureSet", allFeatures[i])
        np.savetxt(test_folder + str(i) + s + "targetSet", allTargets[i])

        print("Starting getting LSTM features and targets for split num:", i)
        allFeatures[i], allTargets[i] = driver_data.generateTrajFeaturesTargetsForLSTM(sectionedFids[i], lanetype,
                        history, histFs, traffic, numFramesToFind)
                        #shape will be (numInputs, numFramesInTrajectory, numFeatures)
        print(allFeatures[i].shape)
        print(allTargets[i].shape)
        np.save(LSTMFolder + str(i) + s + "featureSetv2", allFeatures[i])
        np.save(LSTMFolder + str(i) + s + "targetSetv2", allTargets[i])
        print("Done saving LSTM features and targets split num:", i)
        nframes = len(fids)
    return allFeatures, allTargets, nframes
    
#returns dict1, dict2
#dict1: key intersection, value: array of dimensions [num_training samples, num_features + 1], +1 because prediction at end of feature, also has fid,vid at end
#dict2: key intersection, value: array of dimensions [numInputs, numFramesInTrajectory, numFeatures + 1]
def getAllPossibleFeatures(lankershimfilepath, peachtreefilepath,
                               lanetype=False, history=False, histFs = [5,10,20,30], 
                               traffic = False, numFramesToFind=20):
    allFeatures_normal = {}
    allFeatures_LSTM = {}
    driver_data = dd2.data2(lankershimfilepath)
    fids = reduceFids(driver_data.getFrames())
    intersectionIDs = [1,2,3,4]
    shift = len(intersectionIDs)
    #allFeatures_normal = \
    #        driver_data.getAllFeatureVectors_intersection(fids, lanetype,
    #                                history, histFs, traffic, bool_peach=False, ids=intersectionIDs)
    allFeatures_LSTM = \
                driver_data.generateTrajFeaturesTargetsForLSTM(fids, lanetype,
                        history, histFs, traffic, numFramesToFind, intersectionIDs, bool_peach=False)
            #this will be a dictionary of intersectionid => features
            #what features are is (numInputs, numFramesInTrajectory, numFeatures)
    print("Done getting features for lankershim (", int(lanetype), int(history), int(traffic), ")")
    for i in intersectionIDs:
        allFeatures_normal[i] = convertFromLSTM(allFeatures_LSTM[i])
    print("Done converting for normal")
    driver_data = dd2.data2(peachtreefilepath)
    fids = reduceFids(driver_data.getFrames())
    intersectionIDs = [1,2,3,4,5]
    
    #allFeatures_norm_peach = \
    #            driver_data.getFeatureVectors_intersection(fids, lanetype,
    #                                history, histFs, traffic, bool_peach=True, ids=intersectionIDs)
    allFeatures_lstm_peach = \
                driver_data.generateTrajFeaturesTargetsForLSTM(fids, lanetype,
                        history, histFs, traffic, numFramesToFind, intersectionIDs, bool_peach=True)
    print("Done getting features for peachtree (", int(lanetype), int(history), int(traffic), ")")
    for i in intersectionIDs:
        allFeatures_normal[i+shift] = convertFromLSTM(allFeatures_lstm_peach[i])
        allFeatures_LSTM[i+shift] = allFeatures_lstm_peach[i]
    print("Done converting for normal")
    return allFeatures_normal, allFeatures_LSTM

#input is # features (numInputs, numFramesInTrajectory, numFeatures) 
def convertFromLSTM(features):
    print(features.shape)
    return features.reshape(features.shape[0]*features.shape[1], features.shape[2])

def testFeatureSelection(model, Xtrain, Ytrain, Xtest, Ytest, testnum, save_folder=None, testinters=[], skip=False):
    n = Xtrain.shape[-1]
    usedFeaturesIndexes = set()
    orderedIndexes = []
    prev_performance = (0.0,-1000.0)
    while len(orderedIndexes) <5:# added len to get DNN to finish quick#True:
        if skip or model == "Marginal" or model == "Conditional": 
            usedFeaturesIndexes = list(range(n))
            orderedIndexes = usedFeaturesIndexes
            break
        this_best_performance = prev_performance
        this_best_feature = -1
        for i in range(n):
            if i in usedFeaturesIndexes: continue
            these_features_indices = list(usedFeaturesIndexes)
            these_features_indices.append(i)
            #these_features_indices.sort()
            if "LSTM" in model:
                Xtrain_this = Xtrain[:,:,these_features_indices]
                Ytrain_this = Ytrain
                Xtest_this = Xtrain_this#Xtest[:,:,these_features_indices]
                Ytest_this = Ytrain
            else:
                Xtrain_this = Xtrain[:,these_features_indices]
                Ytrain_this = Ytrain
                Xtest_this = Xtrain_this#Xtest[:,these_features_indices]
                Ytest_this = Ytrain
            Ypred, timeFit, timePred, all_tests_x, all_tests_y = \
                    test(Xtrain_this, Ytrain_this, Xtest_this, Ytest_this, model, testnum, save_folder, exper=True)
            if "LSTM" in model:
                print("ASD", np.array(Ytest_this).shape, np.array(Ypred).shape)
                valid_set = Ytest_this[:,:-1,:]
                print("AS", np.array(valid_set).shape)
                valid_set = valid_set.flatten()
                print("ASD2", np.array(valid_set).shape, np.array(Ypred).shape)
                if max(set(valid_set)) > 2:
                    valid_set = [i-1 for i in valid_set]
                if len(Ypred[0]) == 4:
                    Ypred = Ypred[:,1:] #will be overwritten when saved
                print (np.array(Ypred).shape, np.array(valid_set).shape)
                acc = sutil.findAccuracy(Ypred, valid_set, True)
                score = sutil.findCrossEntropyScore(Ypred, valid_set, True, False)
            else:
                acc = sutil.findAccuracy(Ypred, Ytest_this, True)
                score = sutil.findCrossEntropyScore(Ypred, Ytest_this, True, False)
            this_performance = (acc, score)
            print("Features: ", these_features_indices, "Performance: ", this_performance)
            if acc >= this_best_performance[0] and score >= this_best_performance[1]: #increase log likelihood
                if acc > this_best_performance[0] or score > this_best_performance[1]: #at least one must outright improve
                    this_best_performance = this_performance
                    this_best_feature = i
        if this_best_performance != prev_performance:
            usedFeaturesIndexes.add(this_best_feature)
            orderedIndexes.append(this_best_feature)
            prev_performance = this_best_performance
            print("Features: ", sorted(list(usedFeaturesIndexes)), "Performance: ", prev_performance)
        else:
            break
    usedFeaturesIndexes = list(usedFeaturesIndexes)
    print(usedFeaturesIndexes)
    print("FINAL\nOrdered features: ", orderedIndexes, "Performance (validation set): ", prev_performance)
    print("Test Intersection:", testinters, "TestNum:", testnum, "Model:", model)
    if "LSTM" in model:
        Ypred, timeFit, timePred, all_tests_x, all_tests_y = \
                test(Xtrain[:,:,usedFeaturesIndexes], Ytrain, Xtest[:,:,usedFeaturesIndexes], Ytest, model, testnum, save_folder, exper=False)
    else:
        Ypred, timeFit, timePred, all_tests_x, all_tests_y = \
                test(Xtrain[:,usedFeaturesIndexes], Ytrain, Xtest[:,usedFeaturesIndexes], Ytest, model, testnum, save_folder, exper=False)
    return Ypred, timeFit, timePred, all_tests_x, all_tests_y

def trainTestDNN(Xtrain, Ytrain, Xtest, Ytest, testnum, save_path=None, max_epochs=None, nsteps=1000):
    nclasses = 3#len(set(Ytrain))
    print(Counter(Ytrain))
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    Ytrain = Ytrain.reshape((Ytrain.shape[0], 1))
    rand_perm = np.random.permutation(Xtrain.shape[0])
    print(Ytrain.shape, type(rand_perm), rand_perm.shape)
    Xtrain = Xtrain[rand_perm]
    Ytrain = np.array(Ytrain)[rand_perm]
    tenth = int(Xtrain.shape[0] / 10)
    Xval, Yval = Xtrain[:tenth],Ytrain[:tenth]
    Yval = Yval.flatten()
    Xtrain, Ytrain = Xtrain[tenth:],Ytrain[tenth:]
    tf.logging.set_verbosity(tf.logging.INFO)
    if save_path:
        modeldir = save_path + "DNN_model" + os.sep
        check_make_paths([modeldir])
        shutil.rmtree(modeldir)
        check_make_paths([modeldir])
        classifier = skflow.DNNClassifier(
            feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(Xtrain),
            hidden_units=[128, 128], n_classes=nclasses, model_dir=modeldir)
    else:
        modeldir = "tmp_DNN_"+ testnum + os.sep + str(len(Xtest))
        check_make_paths([modeldir])
        classifier = skflow.DNNClassifier(
            feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(Xtrain),
            hidden_units=[128, 128], n_classes=nclasses, model_dir = modeldir)
        shutil.rmtree(modeldir)
    start = time.clock()
    #log_path = modeldir + "logs" + os.sep
    #check_make_paths([log_path])
    if max_epochs:
        start2 = time.clock()
        prev_performance = (0.0,-1000.0)
        tolerance = 1e-15
        for epoch in range(max_epochs):
            rand_perm = np.random.permutation(Xtrain.shape[0])
            Xtrain_this = Xtrain[rand_perm]
            Ytrain_this = np.array(Ytrain)[rand_perm]
            classifier.fit(Xtrain_this, Ytrain_this, batch_size=1024, steps=nsteps)
            end2 = time.clock()
            this_pred = classifier.predict_proba(Xval)
            acc = sutil.findAccuracy(this_pred, Yval, True)
            score = sutil.findCrossEntropyScore(this_pred, Yval, True, False)
            print("Epoch",epoch,"Done. Took:", end2-start2)
            print("Acc:", acc, "LogLoss:", score)
            start2 = end2
            if score < prev_performance[1] + tolerance: 
                break
    else:
        classifier.fit(Xtrain, Ytrain)#, logdir=log_path)
    end = time.clock()
    timeFit = end - start
    print("Done fitting, time spent:", timeFit)
    start = time.clock()
    probs = classifier.predict_proba(Xtest)
    end = time.clock()
    timePred = end - start
    print("Done predicting, time spent:", timePred)
    #np.savetxt(save_path + "Ypred_DNN", np.array(probs))
    #print("Done saving predictions within traintest function for DNN")
    #try:
    #    classifier.save(modeldir)
    #except:
    #    print("dnn model saving failed")
    #print("Done saving the model")
    return probs, timeFit, timePred
    
#data is already normalized, fit intercept is false
def trainTestSVM(Xtrain, Ytrain, Xtest, Ytest, testnum, save_path=None, probs=False, num_iter=10000):
    #clf = svm.LinearSVC(fit_intercept=False, dual=False, loss='squared_hinge', class_weight='balanced',
    clf = svm.SVC(kernel='rbf', probability=probs, #class_weight='balanced',
            random_state=42, max_iter=num_iter, verbose=False, C=100)
    #maxFeatures = 10
    #print("Starting SVM feature selection with", maxFeatures, "max features")
    #Xtrain = SelectKBest(k=min(len(Xtrain[1,:]), maxFeatures)).fit_transform(Xtrain, Ytrain)
    print("Starting to fit SVM")# with probs=",probs)
    start = time.clock()
    clf.fit(Xtrain, Ytrain)
    end = time.clock()
    timeFit = end - start
    print("Done fitting, time spent:", timeFit)
    start = time.clock()
    if probs==False:
        Ypred = clf.predict(Xtest)
    else:
        Ypred = clf.predict_proba(Xtest)
    end = time.clock()
    timePred = end - start
    print("Done predicting, time spent:", timePred)
    #saver = tf.train.Saver()
    if save_path:
        joblib.dump(clf, save_path + "SVM")
        print("Done saving the model")
    return Ypred, timeFit, timePred

def trainTestNaiveBayes(Xtrain, Ytrain, Xtest, Ytest, testnum, probs=False):
    clf = GaussianNB()
    print("Starting to fit Naive Bayes")
    start = time.clock()
    clf.fit(Xtrain, Ytrain)
    print(clf.class_prior_)
    #clf.class_prior_ = [0.333,0.333,0.333]
    print(clf.class_prior_)
    end = time.clock()
    timeFit = end - start
    print("Done fitting, time spent:", timeFit)
    start = time.clock()
    if probs==False:
        Ypred = clf.predict(Xtest)
    else:
        Ypred = clf.predict_proba(Xtest)
    end = time.clock()
    timePred = end - start
    print("Done predicting, time spent:", timePred)
    return Ypred, timeFit, timePred

def test(Xtrain, Ytrain, Xtest, Ytest, model, testnum, save_path=None, exper=False):
    all_tests_x = np.array([])
    all_tests_y = np.array([])
    print("Starting model", model, "test", testnum)
    if exper==True: 
        Xtrain = Xtrain[::8]
        Ytrain = Ytrain[::8]
        save_path=None
    if model == "SVM":
        random.seed(42)
        combined = list(zip(Xtrain, Ytrain))
        random.shuffle(combined)
        Xtrain[:], Ytrain[:] = zip(*combined)
        iters = 50000
        if exper:
            iters = 100000
            Xtrain = Xtrain[::3]  #24th  --- a further every 5th to make it every 25th
            Ytrain = Ytrain[::3]
        Ypred, timeFit, timePred = trainTestSVM(Xtrain, Ytrain, Xtest, Ytest, testnum, save_path, probs=True, num_iter=iters)
    elif model == "nb":
        random.seed(42)
        combined = list(zip(Xtrain, Ytrain))
        random.shuffle(combined)
        Xtrain[:], Ytrain[:] = zip(*combined)
        Ypred, timeFit, timePred = trainTestNaiveBayes(Xtrain, Ytrain, Xtest, Ytest, testnum, probs=True)
    elif model == "DNN":
        max_epochs = 40
        nsteps = 1000
        if exper == True:
            max_epochs = 2
            nsteps = 10000
        Ypred, timeFit, timePred = trainTestDNN(Xtrain, Ytrain.astype(int), 
                                                Xtest, Ytest.astype(int), testnum, save_path, 
                                                max_epochs=max_epochs, nsteps=nsteps)
    elif "LSTM" in model:
        if exper == True:
            numEpochs=5
        else:
            numEpochs = 30
        Ypred, timeFit, timePred, all_tests_x, all_tests_y = LSTM.run_LSTM((Xtrain,Ytrain), (Xtest, Ytest), model=model, save_path=save_path, numEpochs = numEpochs)
        #np.save(save_path + "usedY", all_tests_y)
        #np.save(save_path + "usedX", all_tests_x)
    elif "Marginal" in model:
        Ypred, timeFit, timePred = sutil.trainTestMarginal(Xtrain, Ytrain.astype(int), 
                                                Xtest, Ytest.astype(int), testnum, save_path)                                            
    elif "Conditional" in model:
        Ypred, timeFit, timePred = sutil.trainTestConditional(Xtrain, Ytrain.astype(int), 
                                                Xtest, Ytest.astype(int), testnum, save_path)
    else:
        print("Invalid model type:", model, "Running test number:", testnum)
        return None
    return Ypred, timeFit, timePred, all_tests_x, all_tests_y
    
    


def createFeaturesOnly(filepath, testnum, numFrames=20):
    test_folder = c.PATH_TO_RESULTS + "Test" + str(testnum) + "/"
    use_lanetype = True
    use_history = False
    use_traffic = False
    if type(testnum) == float:
        use_lanetype = False
    if int(testnum) % 2 == 0:
        use_history = True
        use_traffic = True
    generateFeaturesAndTargets(filepath, test_folder, portionTrain=0.8, load=None,
                                lanetype=use_lanetype, history=use_history, 
                                histFs = [3,10,20,30], traffic = use_traffic, 
                                numFramesToFind=numFrames)

def createAllFeaturesAndTargets():
    filepath1 = dru.findPathForFile("AUGv2_trajectories-0830am-0900am.txt")   
    filepath2 = dru.findPathForFile("trajectories-peachtree.txt")   
    for testnum in [1, 1.1, 2]:
        print("Creating features and targets for testnum:", testnum, "and filepath:")
        print(filepath1)
        createFeaturesOnly(filepath1, testnum)
        print("Done with testnum:", testnum)
    for testnum in [3, 4]:
        print("Creating features and targets for testnum:", testnum, "and filepath:")
        print(filepath2)
        #manually move features for lankershim (not hard)
        createFeaturesOnly(filepath2, testnum)
        print("Done with testnum:", testnum)

#move to utils
def check_make_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    
def newCreateAllFeaturesAndTargets(testtypes, filename_lank="trajectories-lankershim.txt", 
                                   filename_peach="trajectories-peachtree.txt", 
                                   save=True, byIntersection = True):
    filepath1 = dru.findPathForFile(filename_lank)
    filepath2 = dru.findPathForFile(filename_peach)
    if byIntersection:
        save_path = c.PATH_TO_RESULTS + "ByIntersection" + os.sep
    else:
        save_path = c.PATH_TO_RESULTS + "General" + os.sep
    check_make_paths([save_path])
    for testnum in testtypes:
        check_make_paths([save_path+testnum+os.sep])
        allFeatures_normal, allFeatures_LSTM = getAllPossibleFeatures(filepath1, filepath2,
                               lanetype=bool(int(testnum[0])), history=bool(int(testnum[1])), histFs = [5,10,20,30], 
                               traffic = bool(int(testnum[2])), numFramesToFind=20)
        if not save: continue
        if not byIntersection:
            #AppendToEnd(intersectionID, features)
            #save new
            continue
        for intersectionID in allFeatures_LSTM.keys():
            #save features
            check_make_paths([save_path+testnum+os.sep+str(intersectionID) + os.sep])
            print(allFeatures_LSTM[intersectionID].shape)
            np.savetxt(save_path + testnum + os.sep + str(intersectionID) + os.sep + "featuresAndTargets", allFeatures_normal[intersectionID])
            np.save(save_path + testnum + os.sep + str(intersectionID) + os.sep + "LSTM_Formatted_featuresAndTargets", allFeatures_LSTM[intersectionID])
    return allFeatures_normal, allFeatures_LSTM
    #make features with vehicle id and frame id, which can be queried later
    #no longer using lane type feature, except must include for conditional baseline
    

'''
nonLSTMs = ["SVM", "DNN"]
LSTMs = ["LSTM_128x2", "LSTM_128x3", "LSTM_256x2"]
baselines = ["Marginal", "Conditional"] #categorical distributions
models_to_run = ["DNN"]
tests_to_run = [4]
def run(filepath1, load, test_nums=tests_to_run, models=LSTMs, filepath2=None):
    load_f = None#c.PATH_TO_RESULTS + "test2_830-900am/"
    for testnum in test_nums:
        for model in models:
            if testnum >= 3 and filepath2:
                test_split(filepath1, filepath2, portionTrain=0.8,
                           testnum=testnum, model=model)
            else:
                if testnum >= 3: 
                    print("Error before test started, no filepath given")
                    return
                test_non_split(filepath1, portionTrain=0.8,
                           testnum=testnum, model=model, load_folder=load_f)
'''
    
def augAllOrigFiles():
    stringToStart = "tra"
    stringToStart = "AUG"
    for subdir, dirs, files in os.walk(c.PATH_TO_RESOURCES):
        for file in files:
            filepath = subdir + os.sep + file
            if not filepath.endswith(".txt") or not file[:3] == stringToStart: 
                continue    
            print(filepath)
            du.augOrigData(filepath)

def doStuffForPeachtree():
    folder = c.PATH_TO_RESOURCES + "Peachtree" + os.sep
    filename1 = "trajectories-1245pm-0100pm.txt"
    filename2 = "trajectories-0400pm-0415pm.txt"
    #for filepath in [folder + filename1, folder+filename2]:
    #    du.augOrigData(filepath)
    #combine these two that are not following one another
    newFile = futil.combineTrajFilesNoOverlap(folder + filename1, 
                              folder + filename2, 
                              skipFront=500, skipEnd=1000)
    print(newFile)
    return(newFile)

def combineThose2files(overlapStartFrame=10201, maxVid1=1438):
    file1 = 'trajectories-0830am-0845am.txt'
    file2  ='trajectories-0845am-0900am.txt'
    return futil.combineTrajFiles(dru.findPathForFile(file1), dru.findPathForFile(file2))

def hasAnLSTM(models):
    for model in models:
        if ("LSTM") in model: return True
    return False

def new_train(models, testtypes, intersections, saving):
    pass

def new_test(models, testtypes, intersections, saving, graphs):
    pass


def new_train_and_test(models, testtypes, split_inters, saving, graphs, exper=False):
    path_to_load = c.PATH_TO_RESULTS + "ByIntersection" + os.sep 
    hasLSTM = hasAnLSTM(models)
    print("in new_train_and_test, params:", models, testtypes, split_inters, saving, graphs, exper)
    train_inters = split_inters[1] 
    test_inters = split_inters[0]    
    print(train_inters)
    print(test_inters)
    for testnum in testtypes:
        load_folder = path_to_load + testnum + os.sep
        save_folder = load_folder + "TestOn" + ",".join([str(i) for i in test_inters]) + os.sep
        check_make_paths([save_folder])
        if hasLSTM:
            featuresLSTM, targetsLSTM = du.getFeaturesLSTM(load_folder, testnum, train_inters)
            testfeaturesLSTM, testtargetsLSTM = du.getFeaturesLSTM(load_folder, testnum, test_inters)
            print(featuresLSTM.shape)
            print(targetsLSTM.shape)
            print(testfeaturesLSTM.shape)
            print(testtargetsLSTM.shape)
        features, targets = du.getFeaturesnonLSTM(load_folder, testnum, train_inters)
        testfeatures, testtargets = du.getFeaturesnonLSTM(load_folder, testnum, test_inters)
        print(features.shape)
        print(targets.shape)
        print(testfeatures.shape)
        print(testtargets.shape)
        for model in models:
            print(model)
            if "LSTM" in model:
                Xtrain = featuresLSTM
                Ytrain = targetsLSTM
                Xtest = testfeaturesLSTM
                Ytest = testtargetsLSTM
            else:
                Xtrain = features
                Ytrain = targets
                Xtest = testfeatures
                Ytest = testtargets
            #adfsd = list(Ytrain)
            #print("YYYYYYYY", adfsd.index(1), adfsd.index(2), adfsd.index(3))
            Xtrain, Xtest = du.normalize_wrapper(Xtrain, Xtest)
            if exper:
                testFeatureSelection(model, Xtrain, Ytrain, Xtest, Ytest, testnum)
                continue
            Ypred, timeFit, timePred, all_tests_x, all_tests_y = testFeatureSelection(model, Xtrain, Ytrain, Xtest, Ytest, testnum, save_folder, test_inters, skip=True)
            np.savetxt(save_folder + "Ypred_" + model, np.array(Ypred))
            print(model, "predictions saved, test", testnum)


def evaluate(models, testnums, testOn=["1,2,3,4"], dist_hist=False, quiet=False, load=False):
    scores = defaultdict(dict)  #model: (score, numWrong)
    if load:
        scores = loadScores(score_folder=os.getcwd()+os.sep+"scores"+os.sep+"".join([i for i in testOn.split(",")])+os.sep, models=models, testnums=testnums, dist_hist=dist_hist)
        return scores
    for testnum in testnums:
        print("Doing test:", testnum, "testing on:", testOn)
        #test_folder = c.PATH_TO_RESULTS[:-1] + "_backup/" + "ByIntersection" + os.sep + testnum + os.sep
        #print(test_folder)
        test_folder = c.PATH_TO_RESULTS + "ByIntersection" + os.sep + testnum + os.sep
        for model in models:
            print("Scoring for model:", model)
            score = sutil.score(test_folder, model, testInters=testOn, dist_hist=dist_hist, quiet=quiet) 
                #score, acc, prec
                #acc, counts, variances
            scores[testnum][model] = score
    return scores

#save_folder should swap "general" for testOnX if not the general(averaged) results
def saveScores(scores, save_folder=os.getcwd()+os.sep+"scores"+os.sep+"general", dist_hist=False, loaded=False):
    for testnum in sorted(list(scores.keys())):
        for model in sorted(list(scores[testnum].keys())):
            filepath = save_folder + os.sep + str(testnum) + os.sep + model + os.sep
            check_make_paths([filepath])
            if not dist_hist:
                accuracy, precision, score = scores[testnum][model]
                np.savetxt(filepath+"accuracy.txt", np.array(accuracy).reshape(1,1))
                np.savetxt(filepath+"score.txt", np.array(score).reshape(1,1))
            else:
                x = scores[testnum][model]
                accuracies = x[0]
                counts = x[1]
                if len(x) == 3:
                    variances = x[2] 
                    varss = [(i, variances[i]) for i in sorted(list(variances.keys()))]
                    np.savetxt(filepath+"variances.txt",varss)
                accuracies = [(i, accuracies[i]) for i in sorted(list(accuracies.keys()))]
                counts = [(i, counts[i]) for i in sorted(list(counts.keys()))]
                np.savetxt(filepath+"accuracies.txt",accuracies)
                np.savetxt(filepath+"counts.txt",counts)

def saveAllScoresExcel(scores, save_folder=os.getcwd()+os.sep+"scores"+os.sep+"general", dist_hist=False):
    valstosave = []
    for testinter in scores: #should only be one
        saveFile = save_folder + os.sep + "scores_dist_" + str(dist_hist)
        with open(saveFile, 'w') as f:
            for testnum in sorted(scores[testinter]):
                f.write(str(testnum)+"\n")
                first = True  #for with dist histogram, only need counts once
                for model in sorted(scores[testinter][testnum]):
                    a = scores[testinter][testnum][model]
                    key = str(model)
                    if not dist_hist:
                        a = (float(a[0]), float(a[1]))
                        f.write(" ".join([key, str(a[0]), str(a[1])]) + "\n")
                    else:
                        f.write(str(model)+"\n")
                        accuracies = a[0]
                        counts = a[1]
                        for pos, val in accuracies:
                            f.write(" ".join([str(pos), str(val)]) + "\n")
                        if first:
                            f.write("Counts"+"\n")
                            for pos, val in counts:
                                f.write(" ".join([str(pos), str(val)]) + "\n")
                            first=False

def loadScores(score_folder = os.getcwd()+os.sep+"scores"+os.sep, models=["LSTM_128x2"], testnums=["000"], dist_hist=False):
    scores = {}
    for testnum in testnums:
        scores[testnum] = {}
        for model in models:
            filepath = score_folder + str(testnum) + os.sep + model + os.sep
            if not dist_hist:
                accuracy = np.loadtxt(filepath+"accuracy.txt")
                score = np.loadtxt(filepath+"score.txt")
                scores[testnum][model] = (accuracy, score)
            else:
                accuracies = np.loadtxt(filepath+"accuracies")
                counts = np.loadtxt(filepath+"counts")
                scores[testnum][model] = (accuracies, counts)
    return scores

def printScores(scores):
    for testnum in sorted(list(scores.keys())):
        print("\nTest:", testnum)
        for model in sorted(list(scores[testnum].keys())):
            print(model, scores[testnum][model])
            
def doEvalThings(models, testtypes, teston, opts):
    all_scores = {}
    for testinter in teston.split(","):
        this_inters = ",".join(testinter)
        all_scores[testinter] = evaluate(models, testtypes, this_inters, "d" in opts, "q" in opts, "l" in opts)
    print("".join(["-"]*80))
    print("".join(["-"]*80))
    for testinter in sorted(list(all_scores.keys())):
        print("".join(["="]*40))
        if not "d" in opts:
            print("score for testinter:", testinter)
            printScores(all_scores[testinter])
        if "s" in opts and "l" not in opts:
            saveScores(all_scores[testinter],save_folder=os.getcwd()+os.sep+"scores"+os.sep+str(testinter), dist_hist = "d" in opts)
        if "l" in opts and "s" in opts:
            saveAllScoresExcel(all_scores, save_folder=os.getcwd()+os.sep+"scores"+os.sep+str(testinter), dist_hist = "d" in opts)

        
#THIS IS ALL THAT IS NEEDED STARTING WITH A PLAIN TRAJECTORY FILE
#filepath = combineThose2files()  #<--- only needed when combining
#du.augOrigData(dru.findPathForFile(filename))
#newfilename = 'AUGv2_' + filename

#signalfilename='Signals0845-0900handInferred.txt'
#su.visualizeForValidation(dru.findPathForFile(filename), formattype=0)
#su.visualizeForValidation(dru.findPathForFile(filename), dru.findPathForFile(signalfilename), formattype=2)
#signalDict = su.readSignalTimeSplit(dru.findPathForFile(filename))
#testLSTM()


#createFeaturesOnly(dru.findPathForFile("AUGv2_trajectories-0830am-0900am.txt"), 2)
#createFeaturesOnly(dru.findPathForFile("AUGv2_trajectories-0830am-0900am.txt"), 2.1)

#doStuffForPeachtree()

#createAllFeaturesAndTargets()

'''
premadeFeatures = True
run(dru.findPathForFile("AUGv2_trajectories-0830am-0900am.txt"), 
    load=premadeFeatures, test_nums=tests_to_run, models=models_to_run, 
    #filepath2 = None)
    filepath2 = dru.findPathForFile("trajectories-peachtree.txt"))
'''
#doStuffForPeachtree()
#evalu.loadPrintScore("", p_dist=True, offset=False, model_type="BN")
#evalu.loadPrintScore("", p_dist=True, offset=True, model_type="LSTMsmall")

nonLSTMs = ["SVM", "BN", "nb", "DNN"]
LSTMs = ["LSTM_128x2", "LSTM_128x3", "LSTM_256x2"]
baselines = ["Marginal", "nb"] 
#nb is naive bayes
model_choices = {
    "all": ["Marginal", "SVM", "BN", "nb", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "bases": baselines,
    "nb": ["nb"],
    "lstms": LSTMs,
    "svm": ["SVM"],
    "bn": ["BN"],
    "svmbn": ["SVM", "BN"],
    "dnn": ["DNN"],
    "nlstms": nonLSTMs,
    "nns": ["DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "nn": ["DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "lstm1": ["LSTM_128x2"],
    "lstm2": ["LSTM_128x3"],
    "lstm3": ["LSTM_256x2"],
    "lstmtest": ["LSTM_test1"],
    "nsvmbn": ["Marginal", "nb", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "nsvm": ["Marginal", "BN", "nb", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "nbasesbn": ["SVM", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "nbasessvm": ["BN", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "nbases": ["SVM", "BN", "DNN", "LSTM_128x2", "LSTM_128x3", "LSTM_256x2"],
    "dnnlstm1": ["DNN", "LSTM_128x2"],
    "notnns": ["Marginal", "SVM", "BN", "nb"],
}

'''To incorporate command line arguments
    arg1 = augment, featurize, train, test, or train and test, or evaluate saved predictions
    arg2, ... are options specific to arg1
'''
def main():
    if len(sys.argv) == 1:
        print("No arguments received, defaulting to...")
    elif sys.argv[1] == "a":
        print("Augmenting raw trajectoris")
        for filename in sys.argv[2:]:
            print("Augmenting file:", filename)
            du.augOrigData(dru.findPathForFile(filename))
    elif sys.argv[1] == "c": 
        print("Combining trajectory files")
        doStuffForPeachtree()
        combineThose2files()
    elif sys.argv[1] == "f":
        print("Featurizing augmented data")
        if len(sys.argv) > 3:
            filename_lank = sys.argv[3]
            filename_peach = sys.argv[4]
        else:
            filename_lank = "AUGv2_trajectories-lankershim.txt"
            filename_peach = "AUGv2_trajectories-peachtree.txt"
        print("Enter ',' separated testtypes, blank for all")
        testtypes = input()
        if testtypes == "": testtypes = ["000","100","001","010","011"]
        else: testtypes = testtypes.split(",")
        print("Test types to featurize for:", testtypes)
        print("Lank as:", filename_lank, "Peach as:", filename_peach)
        if sys.argv[2] == "s":
            print("saving featurzied general data")
            newCreateAllFeaturesAndTargets(testtypes,filename_lank, filename_peach, save=True, byIntersection=False)
        elif sys.argv[2] == "i":
            print("saving feautrized data by intersection")
            newCreateAllFeaturesAndTargets(testtypes,filename_lank, filename_peach, save=True)
        elif sys.argv[2] == "n":
            print("not saving")
            newCreateAllFeaturesAndTargets(testtypes,filename_lank, filename_peach, save=False)
        else:
            print("invalid featurization option,", sys.argv[2], "options are s (save general), i (save by intersection), n (dont save - not recommended)")
        print("Done featurizing")
    elif "t" in sys.argv[1]:
        #argv[2] == models (svm or dnn or lstms or nns (dnn and lstms) or all, etc.)
        #argv[3] == testtypes - "," separated 000,001,010,011,100
        #argv[4] == "," separated test intersections, assume train on all except test
	#argv[5] == optional - subset, number of training examples - UNUSED
        #argv[6] == s (save), 0 (dont save) - UNUSED, save all
        model_arg = sys.argv[2]
        if not model_arg in model_choices:
            print("invalid model choice of:", model_arg)
            print("valid choices are:", list(model_choices.keys()))
            return
        models = model_choices[sys.argv[2]]
        testtypes = sys.argv[3].split(",")
        saving = True #sys.argv[6]
        str_inters = sys.argv[4].split(",")
        for test_inters in str_inters:
          list_test = [int(i) for i in test_inters]
          train_inters = sorted(list( set([1,2,3,4,5,6,7,8,9]) - set(list_test)))
          print("test inters:", list_test) 
          print("train inters:", train_inters)
          intersections = ([int(i) for i in test_inters],[int(i) for i in train_inters])
          if sys.argv[1] == "tr":
            if not saving:
                print("indicated to train without saving, which is useless, will save")
            print("training and saving")
            saving = True
            new_train(models, testtypes, intersections, saving)
          elif sys.argv[1] == "te":
            print("testing only")
            graphs = input("Save graphs?")
            new_test(models, testtypes, intersections, saving, graphs)
          elif sys.argv[1] == "t":
            print("training and testing")
            graphs = False#input("Save graphs?")
            new_train_and_test(models, testtypes, intersections, saving, graphs)
    elif sys.argv[1] == "e": 
        print("evaluating.")
        #argv[2] == models (svm or dnn or lstms or nns (dnn and lstms) or all)
        #argv[3] == testtypes - "," separated 000,001,010,011,100
        #argv[4] == testOnIntersection - "," separated for multiple
        #argv[5] is optional 
        #           if contains "d" then produce distance histograms
        #           if contains "s" then save scores to files 
        #           if contains "q" then do only print final outputs
        #           if contains "l" then load the files. If s and l, will save to excel
        #           if contains "g" then make graphs and plot (currently unsupported, becoming supported)
        #       ex: "ds" would make distance histogram values and save
        model_arg = sys.argv[2]
        if not model_arg in model_choices:
            print("invalid model choice of:", model_arg)
            print("valid choices are:", list(model_choices.keys()))
            return
        models = model_choices[sys.argv[2]]
        testtypes = sys.argv[3].split(",")
        teston = sys.argv[4]
        opts = ""
        if len(sys.argv)>5:
            opts = sys.argv[5]
        doEvalThings(models, testtypes, teston, opts)
    elif sys.argv[1] == "x": #experimental stuffs
        print("doing crazy things probably")
        #argv[2] == models (svm or dnn or lstms or nns (dnn and lstms) or all)
        #argv[3] == testtypes - "," separated 000,001,010,011,100
        #argv[4] == "," separated test intersections, assume train on all except test
	#argv[5] == optional - subset, number of training examples - UNUSED
        #argv[6] == s (save), 0 (dont save) - UNUSED, save all
        model_arg = sys.argv[2]
        if not model_arg in model_choices:
            print("invalid model choice of:", model_arg)
            print("valid choices are:", list(model_choices.keys()))
            return
        models = model_choices[sys.argv[2]]
        testtypes = sys.argv[3].split(",")
        saving = True #sys.argv[6]
        str_inters = sys.argv[4].split(",")
        for test_inters in str_inters:
          list_test = [int(i) for i in test_inters]
          train_inters = sorted(list( set([1,2,3,4,5,6,7,8,9]) - set(list_test)))
          print("test inters:", list_test) 
          print("train inters:", train_inters)
          intersections = ([int(i) for i in test_inters],[int(i) for i in train_inters])
          new_train_and_test(models, testtypes, intersections, saving, False, exper=True)
    elif sys.argv[1] == "h":
        print("options are a - augment, f - featurize, tr - train, te - test, t - train and test")
        print("secondary options are a=>[], f=>[s or i or n (save general or intersection or dont save)], tr/te/t=>[models, [intersections], y/n (save), optional size of subset]")
    else:
        print ("invalid argument:", sys.argv[1], "options are a - augment, f - featurize, tr - train, te - test, t - train and test")

main()
