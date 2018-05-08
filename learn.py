# -*- coding: utf-8 -*-
"""
Created on Mon May 23 18:14:54 2016

@author: Derek
"""

import numpy as np
from lib import learn_util
from lib import util
from lib import constants as c
import sys
import time
from sklearn.externals import joblib
from sklearn import svm
from sklearn import linear_model
import os



#This will probably have to be made better at some point
filename="res/101_trajectories/aug_trajectories-0750am-0805am.txt"
#filename="res/101_trajectories/101_full_trajectories_compressed.txt"

repickTrainTest = 0 #1 to recalulate, 0 to load, -1 to use memory,
seed = 1
remakeData = 2 #1 to recalulate, 0 to load, -1 to use memory
               #2 for clustered recalc, 3 for clustered read, 4 clustered mem
mean_centered = 0 #1 to mean center, 0 to not mean center
predict = 'X' #'Y' or 'X'

if repickTrainTest == 1:
    trainIDs, testIDs = learn_util.makeTrainTestData(filename, .75, seed)
elif repickTrainTest == 0:
    trainIDs, testIDs = learn_util.loadTrainTestData(filename)

cluster0 = np.loadtxt(learn_util.makeFullPath(filename, 'vids_cluster_0.txt'))
cluster1 = np.loadtxt(learn_util.makeFullPath(filename, 'vids_cluster_1.txt'))
cluster2 = np.loadtxt(learn_util.makeFullPath(filename, 'vids_cluster_2.txt'))

if remakeData == 1:
    print("Recalculating all data, mean_centered =",mean_centered,"predicting",predict,"position.")
    print("started at",time.ctime())
    Xtrain, Xtest = learn_util.getX(filename, trainIDs, testIDs, mean_centered)
    print("Finished gathering and formatting X data",time.ctime())
    ytrain, ytest = learn_util.getY(filename, trainIDs, testIDs, predict)
    print("Finished gathering and formatting Y data",time.ctime())
    learn_util.saveExampleData(filename, Xtrain, ytrain, Xtest, ytest, mean_centered, predict)
elif remakeData == 2:
    print("Recalculating all data WITH CLUSTERS, mean_centered =",mean_centered,"predicting",predict,"position.")
    print("started at",time.ctime())
    Xtrain0, Xtrain1, Xtrain2, Xtest0, Xtest1, Xtest2 = learn_util.getXClusters(filename, 
                        trainIDs, testIDs, mean_centered, cluster0, cluster1, cluster2)
    print("Finished gathering and formatting X data",time.ctime())
    ytrain0, ytrain1, ytrain2, ytest0, ytest1, ytest2 = learn_util.getYClusters(filename, 
                        trainIDs, testIDs, predict, cluster0, cluster1, cluster2)
    print("Finished gathering and formatting Y data",time.ctime())
    learn_util.saveExampleDataClusters(filename, Xtrain0, Xtrain1, Xtrain2, ytrain0, ytrain1, ytrain2,
                                       Xtest0, Xtest1, Xtest2, ytest0, ytest1, ytest2,
                                       mean_centered, predict)
elif remakeData == 0:
    print("Loading data from file...",time.ctime())
    Xtrain, ytrain, Xtest, ytest = learn_util.readExampleData(filename, mean_centered, predict)
elif remakeData == 3:
    Xtrain0, Xtrain1, Xtrain2, ytrain0, ytrain1, ytrain2, Xtest0, Xtest1, Xtest2, ytest0, ytest1, ytest2 = learn_util.readExampleDataClusters(filename, mean_centered, predict)


#otherwise, read from files


    
#run this after the model is fit
#if using a model with specific values (like penalties), include that in type
def saveModelStuff(model, modelType, Xtest, ytest, Xtrain, ytrain, filename, clustered=False): #modelType = 'SVM'
    if Xtest.shape == (0,):
        print("Xtest is empty")
        return
    print("Done fitting model, getting predictions...", time.ctime())
    predictions = model.predict(Xtest)
    print ("Done with predictions, scoring...", time.ctime())    
    score = model.score(Xtest,ytest)
    #print ("Getting predictions on train data...", time.ctime())    
    #predictionsTrain = model.predict(Xtrain)
    print ("Scoring check...", time.ctime())    
    check = model.score(Xtrain,ytrain)
    
    print("Done with all testing, saving outputs.", time.ctime())
    subfolder = util.string_appendDateAndTime(modelType) + '/'
    path = learn_util.makePathToTrajectories(filename) + subfolder
    if not os.path.exists(path):
        os.makedirs(path)  
    np.savetxt(path + 'ACTUALS-TEST.txt', ytest)
    np.savetxt(path + 'PREDICTIONS-TEST.txt', predictions)
    np.savetxt(path + 'SCORE-TEST.txt', np.array([score]))         
    #np.savetxt(path + 'ACTUALS-TRAIN.txt', ytrain)
    #np.savetxt(path + 'PREDICTIONS-TRAIN.txt', predictionsTrain)
    np.savetxt(path + 'SCORE-TRAIN.txt', np.array([check]))
    
    joblib.dump(model, path + 'MODEL')
    print('model ', modelType, ': score = ', score, 'train_score = ', check)
    


'''svmR = svm.SVR(cache_size=2500) #default,
print("Fitting default model...", time.ctime())
svmR.fit(Xtrain,ytrain)
modelType = 'SVM-1-0.1-PREDICT-'+predict
if mean_centered==1:
    modelType = modelType + '-mean_centered'
saveModelStuff(svmR, modelType, Xtest, ytest, Xtrain, ytrain, filename)'''
clustered = (remakeData >=2)

if not clustered:
    print(Xtrain.shape)
    print(Xtest.shape)
    print(ytrain.shape)
    print(ytest.shape)
    for penalties in [1]: # this now includes default
        for eps in [.1]:
            svmR = svm.SVR(C=penalties,epsilon=eps,cache_size=1500) #kernel='rbf',
            print("Fitting svm model...", time.ctime())
            svmR.fit(Xtrain,ytrain)
            model_type = 'SVM-'+str(penalties)+'-'+str(eps)+'-PREDICT-'+predict
            if mean_centered==1:
                model_type = model_type + '-mean_centered'
            saveModelStuff(svmR, model_type , Xtest, ytest, Xtrain, ytrain, filename, clustered)

    linmod1 = linear_model.LinearRegression() #aka least squares
    print("Fitting linreg model...", time.ctime())
    linmod1.fit(Xtrain, ytrain)
    modelType = 'linReg-PREDICT-'+predict
    if mean_centered==1:
        modelType = modelType + '-mean_centered'
    saveModelStuff(linmod1, modelType, Xtest, ytest, Xtrain, ytrain, filename, clustered)
else:
    print(Xtrain0.shape)
    print(Xtest0.shape)
    print(ytrain0.shape)
    print(ytest0.shape)
    
    print(Xtrain1.shape)
    print(Xtest1.shape)
    print(ytrain1.shape)
    print(ytest1.shape)
    
    print(Xtrain2.shape)
    print(Xtest2.shape)
    print(ytrain2.shape)
    print(ytest2.shape)
    penalties = 1
    eps = .1
    svmR = svm.SVR(C=penalties,epsilon=eps,cache_size=1500)
    linmod = linear_model.LinearRegression() #aka least squares
    print("Making svm for cluster 0")
    print("Fitting svm model...", time.ctime())
    svmR.fit(Xtrain0,ytrain0)
    model_type = 'SVM-'+str(penalties)+'-'+str(eps)+'-PREDICT-'+predict+'CLUSTER0'
    if mean_centered==1:
        model_type = model_type + '-mean_centered'
    saveModelStuff(svmR, model_type , Xtest0, ytest0, Xtrain0, ytrain0, filename, clustered)
    modelType = 'linReg-PREDICT-'+predict+'CLUSTER0'
    linmod.fit(Xtrain0,ytrain0)
    saveModelStuff(linmod, modelType , Xtest0, ytest0, Xtrain0, ytrain0, filename, clustered)
    print("Making svm for cluster 1")
    print("Fitting svm model...", time.ctime())
    svmR.fit(Xtrain1,ytrain1)
    model_type = 'SVM-'+str(penalties)+'-'+str(eps)+'-PREDICT-'+predict+'CLUSTER1'
    if mean_centered==1:
        model_type = model_type + '-mean_centered'
    saveModelStuff(svmR, model_type, Xtest1, ytest1, Xtrain1, ytrain1, filename, clustered)
    modelType = 'linReg-PREDICT-'+predict+'CLUSTER1'
    linmod.fit(Xtrain1,ytrain1)
    saveModelStuff(linmod, modelType, Xtest1, ytest1, Xtrain1, ytrain1, filename, clustered)

    print("Making svm for cluster 2")
    print("Fitting svm model...", time.ctime())
    svmR.fit(Xtrain2,ytrain2)
    model_type = 'SVM-'+str(penalties)+'-'+str(eps)+'-PREDICT-'+predict+'CLUSTER2'
    if mean_centered==1:
        model_type = model_type + '-mean_centered'
    saveModelStuff(svmR, model_type, Xtest2, ytest2, Xtrain2, ytrain2, filename, clustered)
    modelType = 'linReg-PREDICT-'+predict+'CLUSTER2'
    linmod.fit(Xtrain2,ytrain2)
    saveModelStuff(linmod, modelType, Xtest2, ytest2, Xtrain2, ytrain2, filename, clustered)




'''#followed advice from http://stackoverflow.com/questions/34475245/sklearn-svm-svr-and-svc-getting-the-same-prediction-for-every-input
for i in range(len(ytrain[0])):
    #train model on Xtrain, ytrain[:,i]
    svmR.fit(Xtrain,ytrain[:,i])
    ypredict = svmR.predict(Xtest)
    if i % 15 == 0:
        print('Currently on iteration:', i)
        print('Predictions:',ypredict)
        print('Actuals:',ytest[:,i])
    ytests.append(ytest[:,i])
    predictions.append(ypredict)
    scores.append(svmR.score(Xtest,ytest[:,i]))
diff = np.array(ytests)-np.array(predictions)
norm = np.linalg.norm(diff)
print(diff)
print(norm)
print (scores)
print (max(scores))
print (min(scores))'''