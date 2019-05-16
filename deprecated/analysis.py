# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:51:10 2016

@author: Derek
"""

'''This is the final that will perform analyisis on the models
    Specifically, it will be performing "black box" testing.
    The main components of the file:
        Loading the models
                In the case of the BayesNet, we defer to analysis.jl
            For the DNN, we can load the model
            For the LSTMs, it would be preferred to just load the model, but I have
                been having major issues with TensorFlow, because it like cannot save the 
                variables correctly for some reason / it doesnt load them correctly.
                I tried simply renaming the variables, and resetting their values, 
                but apparently there is more to it than that
              Thus, we have to retrain the LSTMs for this purpose. 
              This is exactly the same as the real training, and the only downside 
              is (a pretty major one) that this analysis takes much longer.
              Regardless, the same outcome should be present, just I must select 
              only a few testing situations to analyze.
        Selecting the testing situations
            This is done by hand, based on a visual inspection of the results, seeing which
            make the least sense or otherwise are interesting. 
            At the moment, I am leaning towards analyzing only a few intersections but all the feature sets
        Doing the stuff and the things
            Basically, because all of the non-BN models normalized the inputs, 
            its very easy to compare the effectuve weights. 
            All that is done, is from a baseline input [0]* num_inputs
                I iterate over each feature and vary it in range(-1,1,0.05)
                Recording the probability distribution that is output.
        From the outputs of the above, I will plot the sensitivity of each of the inputs,
            and  I hypothesize velocity will be the most sensitive, with headway mostly ignored.
'''

import os
import sys
sys.path.append(os.environ["INTENTPRED_PATH"])

from utils import LSTM
from sklearn.externals import joblib
from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import tensorflow as tf
import tensorflow.contrib.learn as skflow

from utils import constants as c
from utils import data_util as du
import time

import numpy as np

#creates the set of data to test sensitivity of inputs
def createAnalysisTestData(numFeatures, traj_len=1):
    base = [0.0]*numFeatures
    Xtest = np.array([base]*traj_len)
    Xtest = Xtest.reshape(1,traj_len,numFeatures)
    Y = 0
    Ytest = np.array([Y]*traj_len)
    Ytest = Ytest.reshape(1,traj_len,1)
    for i in range(numFeatures):
        for val in [round(-1.0 + 0.05*x,2) for x in range(int(205/5))]:
            this_features = [0.0]*numFeatures
            this_features[i] = val
            this_entry = np.array([this_features]*traj_len)
            this_entry = this_entry.reshape(1,traj_len,numFeatures)
            Xtest = np.vstack((Xtest, this_entry))
            this_y = np.array([0]*traj_len)
            this_y = this_y.reshape(1,traj_len,1)
            Ytest = np.vstack((Ytest, this_y))
    print(Xtest.shape)
    print(Ytest.shape)
    return Xtest, Ytest
            
#this function is given the model, well due to the load issues, just the intersection and feature sets
#test_inters is a list, like [1] or [1,2]
#testtype is a string like "001"
def analyze_model(test_inters, testtype, model):
    path_to_load = c.PATH_TO_RESULTS + "ByIntersection" + os.sep 
    load_folder = path_to_load + testtype + os.sep
    save_folder = load_folder + "TestOn" + ",".join([str(i) for i in test_inters]) + os.sep
    Ypred = None
    if "LSTM" in model:
        Xtrain, Ytrain = du.getFeaturesLSTM(load_folder, testtype, list({1,2,3,4,5,6,7,8,9}-set(test_inters)))
        #Xtest, Ytest = du.getFeaturesLSTM(load_folder, testtype, test_inters)
        means, stddevs = du.normalize_get_params(Xtrain)
        Xtrain = du.normalize(Xtrain, means, stddevs)
        numFeatures = Xtrain.shape[2]
        Xtest, Ytest = createAnalysisTestData(numFeatures, traj_len=Xtrain.shape[1])
        #train the LSTM again
        Ypred, timeFit, timePred, all_tests_x, all_tests_y = LSTM.run_LSTM((Xtrain,Ytrain), (Xtest, Ytest), model=model, save_path="ignore.out")
    else:
        Xtrain, Ytrain = du.getFeaturesnonLSTM(load_folder, testtype, list({1,2,3,4,5,6,7,8,9}-set(test_inters)))
        #Xtest, Ytest = du.getFeaturesnonLSTM(load_folder, testtype, test_inters)
        means, stddevs = du.normalize_get_params(Xtrain)
        Xtrain = du.normalize(Xtrain, means, stddevs)
        numFeatures = Xtrain.shape[1]
        Xtest, _ = createAnalysisTestData(numFeatures, traj_len=1)
        classifier = skflow.DNNClassifier(
            feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(Xtrain),
            hidden_units = [128,128], n_classes=3)#, model_dir=save_folder)
        #try:
        #    Ypred = classifier.predict_proba(Xtest)
        #except:
        print("Could not load saved model, re-training :(.")
        Ytrain = [int(i-1) for i in Ytrain]
        start = time.clock()
        max_epochs = 10
        if max_epochs:
            start2 = time.clock()
            for epoch in range(max_epochs):
                classifier.fit(Xtrain, Ytrain, steps=1000)
                end2 = time.clock()
                print("Epoch",epoch,"Done. Took:", end2-start2)
                start2 = end2
        else:
            classifier.fit(Xtrain, Ytrain)#, logdir=log_path)
        Ypred = classifier.predict_proba(Xtest)
        end = time.clock()
        timeFit = end - start
    print("Done fitting, time spent:", timeFit)

    np.savetxt(save_folder + "analysis_Ypred_" + model, np.array(Ypred))
    print(model, "analysis predictions saved, test", testtype, save_folder,"analysis_Ypred_", model)
    return Ypred
        

def doTheThings(models=["LSTM_128x2","LSTM_128x3","LSTM_256x2"]):
    for intersection in [3,7]:
        for testtype in ["000","001","010","011","100"]:
            for model in models:
                analyze_model([intersection],testtype,model)

features_test = {
        "000":9,"001":65,"010":45,"011":101,"100":13,"111":103
        }

def doAnalysisThings(models, testtypes, testinters, opts):
    score_folder = os.getcwd()+os.sep+"results"+os.sep+"ByIntersection"+os.sep
    for intersect in testinters:
        print("=".join(["="]*40))
        print("Intersection",intersect)
        for testnum in testtypes:
            print("-".join(["-"]*40))
            print("Testnum ", testnum)
            numfeatures = features_test[testnum] 
            for model in models:
                filepath = score_folder + str(testnum) + os.sep + "TestOn" + str(intersect) + os.sep + "analysis_Ypred_" + model
                analysis_stuff = np.loadtxt(filepath)
                #X, Y = createAnalysisTestData(numfeatures)
                impact_per_feature = [0] * numfeatures
                this_feature = 0
                if model != "BN":
                    if "LSTM" in model:
                        traj_len = 20
                        analysis_stuff = analysis_stuff[:,1:]
                        analysis_stuff = analysis_stuff[0::(traj_len-1),:]
                    else:
                        analysis_stuff = analysis_stuff[0::numfeatures,:]
                    for row in range(1,len(analysis_stuff-1)):
                        if row % 41 == 0: #len(list(range(int(205/5))))
                            impact_per_feature[this_feature] /= 41
                            impact_per_feature[this_feature] *=numfeatures
                            this_feature += 1
                            continue
                        impact = abs(analysis_stuff[row,1] - analysis_stuff[row+1,1]) + abs(analysis_stuff[row,2] - analysis_stuff[row+1,2])
                        impact_per_feature[this_feature] += impact
                    print(model, " & ", " & ".join([str(i)[:6] for i in impact_per_feature]))



models = ["DNN","LSTM_128x2","LSTM_128x3","LSTM_256x2"]
testtypes = ["000","100"]
testintersections = [3,7]
options = None


models = ["LSTM_128x2"]
testtypes = ["111"]
testintersections = [1]
doAnalysisThings(models, testtypes, testintersections, options)
