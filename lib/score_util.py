# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:53:05 2016

utilities to help with evaluating the SVM 
@author: djp42
"""

from sklearn import metrics
from lib import constants as c
from lib import data_util as du
import numpy as np
import os
import math
import time
from collections import Counter
import operator

class filenames:
    def __init__(self):
        self.targetString = "targetSet"
        self.featureString = "featureSet"
        #self.LSTMtargetString = "used_Y.npy"
        #self.LSTMfeatureString = "used_X.npy"
        self.SVMPreds = "Ypred_SVM"
        self.DNNPreds = "Ypred_DNN"
        self.MarginalPreds = "Ypred_Marginal"
        self.ConditionalPreds = "Ypred_Conditionals"
        self.BNPreds = "BN_p_dists.txt"
        self.LSTMPreds = "Ypred_"
        self.LSTMTypes = ["128x2", "256x2", "128x3"]
        self.LSTMActuals = "usedY.npy"
        self.LSTMFeatures = "usedX.npy"
        self.default_path = c.PATH_TO_RESULTS + "test1/0/"
        self.default_default_path = c.PATH_TO_RESULTS + "ByIntersection/000/"

def getValsForDistanceHistogram(actuals, predictions, features, nbins=25, testnum="000", max_d=None):
    count_dists_int = []
    wrong_dists_int = []
    distInd = 7
    if testnum[0] == "1":
        distInd = 11
    for i in range(len(predictions)):
        dist = int(features[i, distInd])
        if not int(actuals[i]) == int(predictions[i]):
            wrong_dists_int.append(dist)
        count_dists_int.append(dist)
    if not max_d:
        max_dist = max(count_dists_int)
    else:
        max_dist = max_d
    bin_width = max_dist / nbins
    bin_counts = {}
    bin_wrongs = {}
    bin_accuracies = {}
    count_ratios = {}
    for bin_n in range(nbins+1):
        bin_start = (bin_n-1)*bin_width
        bin_end = bin_start + bin_width
        bin_pos = (bin_start + bin_end) / 2
        if not bin_pos in bin_counts.keys():
            bin_counts[bin_pos] = 0
            bin_wrongs[bin_pos] = 0
        for dist in wrong_dists_int:
            if dist > bin_start and dist <= bin_end:
                bin_wrongs[bin_pos] += 1
        for dist in count_dists_int:
            if dist > bin_start and dist <= bin_end:
                bin_counts[bin_pos] += 1
    num_entries = len(count_dists_int)
    feet_keys = sorted(list(bin_counts.keys()))
    for feet_key in feet_keys:  #convert feet to meters
        meter_key = feet_key * 0.3048
        bin_counts[meter_key] = bin_counts.pop(feet_key)
        bin_wrongs[meter_key] = bin_wrongs.pop(feet_key)
    for key in bin_counts.keys():
        if bin_counts[key] > 0:
            bin_accuracies[key] = 1.0 - (bin_wrongs[key]/bin_counts[key])
        count_ratios[key] = bin_counts[key] / num_entries
    return bin_accuracies, count_ratios

#testfolder is results/ByIntersection/000/ by default
#testInters is 1 by default
def score(test_folder=None, model="SVM", doAvg=False, limit_by=None, limits=None, testInters="1", 
          dist_hist=False, quiet=False):
    #limit_by can be LaneType, movetype, or distance 
    # and limits can be (min, max) for distance, or type of move, or list of lanetypes to use
    fileclass = filenames()
    if not test_folder:
        test_folder = fileclass.default_default_path
    Ps = True
    if "LSTM" in model:
        offset = False
    else:
        offset = True
    if not dist_hist:
        score, accuracy, precision = loadPrintScoreWrapper(test_folder, testInters.split(","), Ps, offset,
                                                       model, doAvg, limit_by, limits, dist_hist, quiet)
        return accuracy, precision, score
    else:
        accuracies, counts = loadPrintScoreWrapper(test_folder, testInters.split(","), Ps, offset,
                                                       model, doAvg, limit_by, limits, dist_hist, quiet)
        if testInters == "1,2,3,4,5,6,7,8,9":
            bins = sorted(list(accuracies.keys()))
            other_accs = {}
            for testInter in testInters.split(","):
                a,c = loadPrintScoreWrapper(test_folder, [testInter], Ps, offset,
                                        model, doAvg, limit_by, limits, dist_hist, q=True)
                other_accs[testInter] = a
            variances = {}
            list_accs_by_d = {}
            for b in bins:
                list_accs_by_d[b] = []
                for t in testInters.split(","):
                    if b in other_accs[t]:
                        list_accs_by_d[b].append(other_accs[t][b])
                variances[b] = np.var(list_accs_by_d[b])
            return accuracies, counts, variances
        return accuracies, counts


''' function: loadPrintScoreWrapper
        #folderpath is results/ByIntersection/000/ by default
        args: 
            folderpath: absolute path to folder where predicitons and actuals are
            p_dist: bool of whether or not a p_dist is output
            offset: bool indicating if actuals need to be offset to be used as indexes
            model_type: string for one model type to use (SVM, BN, DNN, LSTM_128x2...)
        
        returns:
            tuple of lists of score (float) and counts (int), which is also printed
'''
def loadPrintScoreWrapper(folderpath, testInters = [1],
                          p_dist=False, offset=True, model_type="SVM",
                          avg_over_CVs = False, limit_by=None, limits=None, dist_hist=False, q=False):
    #limit_by can be LaneType, movetype, or distance 
    # and limits can be (min, max) for distance, or type of move, or list of lanetypes to use
    files = filenames()
    if folderpath == "":
        folderpath = files.default_default_path
    print(folderpath)
    scores = []
    precisions = []
    precision = 0
    accuracies = []
    accuracy = 0
    testnum = folderpath.split(os.sep)[-2]
    #for test_inter in testInters:
    if "LSTM" in model_type:
        features, actuals = du.getFeaturesLSTM(folderpath, testnum, testInters, q=q)
        print(actuals.shape)
        actuals = actuals[:,:-1,:] #LSTM doesnt predict for last in trajectory for whatever reasoni
        print(actuals.shape)
    elif "BN" in model_type:
        features, actuals = du.getFeaturesnonLSTM(folderpath, testnum, testInters, all=True, q=q)
    else:
        features, actuals = du.getFeaturesnonLSTM(folderpath, testnum, testInters, q=q)
    #if model_type == "BN":
    #    predictions = {}
    #    for i in testInters:
    #        predictions[i] = np.loadtxt(folderpath + str(i) + os.sep + "BN_p_dists.txt")
    #    predictions = np.ascontiguousarray(np.concatenate(([predictions[i] for i in sorted(testInters)])))
    #    #predictions = np.loadtxt(folderpath + ",".join([str(i) for i in testInters]) + os.sep + "BN_p_dists.txt")
    #else:
    print(features.shape)
    predictions = {}
    for i in testInters:
        predictions[i] = np.loadtxt(
            os.path.join(folderpath, "TestOn" + str(i), "Ypred_" + model_type))
        #if i == "1" and model_type == "SVM":  #need to swap 1 and 2 classes for only this one, due to order they appeared in the data I would guess, although not really sure why but clearly necessary.
        #    for j in range(len(predictions[i])):
        #        predictions[i][j,1], predictions[i][j,0] = predictions[i][j,0], predictions[i][j,1]
    predictions = np.ascontiguousarray(np.concatenate(([predictions[i] for i in sorted(testInters)])))
    #predictions = np.loadtxt(folderpath + "TestOn" + ",".join([str(i) for i in testInters]) + os.sep + "Ypred_" + model_type)
    actuals = actuals.flatten()
    if max(set(actuals)) > 2:
        actuals = np.array([i-1 for i in actuals])
    if p_dist and len(predictions[0]) == 4:
        predictions = predictions[:,1:]
    print(predictions.shape, actuals.shape)
    #p_dist = (type(predictions[0]) != int)
    if not p_dist:
        if max(set(predictions)) > 2:
            predictions = [i-1 for i in predictions]
    if dist_hist:
        if len(features.shape) > 2:
            features = features.reshape((features.shape[0]*features.shape[1], features.shape[2])) #essentially flattens
        if p_dist: 
            predictions = convertProbabilityToInt(predictions)
        return getValsForDistanceHistogram(actuals, predictions, features, nbins=25, testnum=testnum, max_d=550)
            #returns a dictionary of distance (middle of bin) to accuracy, 
                #and a dictionary for portion of total in that bin
    #score = findMeanSquaredError(predictions,actuals,p_dist)
    score = findCrossEntropyScore(predictions, actuals, p_dist, offset)
    accuracy = findAccuracy(predictions, actuals, p_dist=p_dist)#, offset=offset)
    precision = findPrecision(actuals, predictions, brokenDown=True,
                              p_dist=p_dist, feats=features)
    return score, accuracy, precision

def avgPrecision(unaveragedPrecision):
    #will be a list of dicts. unap[i] is dct for cv i
    #unap[i][0] is precision for predicting move 0 in cv i
    averagedPrecision = {}
    numCVs = len(unaveragedPrecision)
    allmoves = []
    for i in range(numCVs):
        allmoves.extend(list(unaveragedPrecision[i].keys()))
    moves = set(allmoves)
    for i in moves:
        averagedPrecision[i] = 0
    for cv in range(numCVs):
        prec_for_cv = unaveragedPrecision[cv]
        for i in moves:
            if i in prec_for_cv.keys():
                this_prec = prec_for_cv[i]
                averagedPrecision[i] += this_prec / numCVs
    return averagedPrecision

#wrapper for features and actuals preds, necessary when slicing
def getActsPredsFeats(folderpath, model_type, fwd_bkwd=None,
                          limit_by=None, limits=None, means=None, stds=None,
                          opt_load_predictions=None):
    #limit_by can be LaneType, movetype, or distance 
    # and limits can be (min, max) for distance, or type of move, or list of lanetypes to use
    # screwed up, for LSTMs, test 1 and 2 it goes: test - lstm_formatted - cv - model
    # for test 3 and 4 it goes: test - lstm_formatted - model -fwd/bkwd
    files = filenames()
    if folderpath == "":
        folderpath = files.default_path
    act_feat_path = folderpath
    strFwdBkwd = ""
    strFwdBkwd2 = ""
    if fwd_bkwd:
        strFwdBkwd = fwd_bkwd
        strFwdBkwd2 = fwd_bkwd
        if "bk" in fwd_bkwd:
            if "est3" in folderpath:
                act_feat_path = folderpath.replace("est3","est1")
            elif "est4" in folderpath:
                act_feat_path = folderpath.replace("est4","est2")
            strFwdBkwd2 = ""
    actfile = act_feat_path + strFwdBkwd2 + files.targetString
    if opt_load_predictions and "LSTM" in model_type:
        act_feat_path = opt_load_predictions
    if fwd_bkwd and not "LSTM" in model_type:
        actuals = np.concatenate([np.loadtxt(
                os.path.join(act_feat_path, str(i), files.targetString)
            ) for i in range(5)])
    if model_type in ["SVM", "DNN", "Marginal", "Conditional"]:
        predfile = os.path.join(folderpath, model_type, strFwdBkwd + "YPred_" + model_type)
        if opt_load_predictions:
            predfile = os.path.join(opt_load_predictions, model_type, strFwdBkwd + "YPred_" + model_type)
    elif model_type == "BN":
        predfile = folderpath + strFwdBkwd + files.BNPreds
        if opt_load_predictions:
            predfile = os.path.join(opt_load_predictions, strFwdBkwd + files.BNPreds)
    elif "LSTM" in model_type:
        predfile = os.path.join(folderpath, model_type, strFwdBkwd + "YPred_" + model_type)
        #results/Test4/LSTM_formatted/LSTM_128x2/bkwd/Ypred_LSTM_128x2
        if opt_load_predictions:
            predfile = os.path.join(opt_load_predictions, model_type, strFwdBkwd + "YPred_" + model_type)
            
        actfile = os.path.join(act_feat_path, model_type, strFwdBkwd + files.LSTMActuals)
        #results/Test4/LSTM_formatted/LSTM_128x2/bkwd/usedY.npy
    else:
        print("Invalid model type of:", model_type)
        return 0, 0, 0
    print("Predictions from:", predfile)
    predictions = np.loadtxt(predfile)
    if "LSTM" in model_type:
        actuals = np.load(actfile)
        print("Actuals from:", actfile)
        features = getFeatures(act_feat_path, model_type, strFwdBkwd)
        actuals, predictions, features = slice_and_dice(actuals, predictions,
                                                        features, limit_by,
                                                        limits, means=means, stds=stds)
        actuals, predictions, features = slice_and_dice(actuals, predictions,
                                                        features, limit_by="move",
                                                        limits="!0", means=means,
                                                        stds=stds)
    else:
        features = getFeatures(act_feat_path, model_type, strFwdBkwd)
        if not fwd_bkwd:
            actuals = np.loadtxt(actfile)
        actuals, predictions, features = slice_and_dice(actuals, predictions, 
                                                        features, limit_by, limits,
                                                        means, stds)
    return actuals, predictions, features

def getFeatures(folderpath, model_type, fwd_bkwd=None):
    files = filenames()
    if folderpath == "":
        folderpath = files.default_path
    strFwdBkwd = ""
    if fwd_bkwd:
        strFwdBkwd = fwd_bkwd
    if fwd_bkwd and not "LSTM" in model_type:
        features = np.concatenate([np.loadtxt(
                os.path.join(folderpath, str(i), files.featureString)
            ) for i in range(5)])
    elif "LSTM" in model_type:
        fpath = os.path.join(folderpath, model_type, strFwdBkwd + files.LSTMFeatures)
        features = np.load(fpath)
        print("features from:", fpath)
        #features = np.reshape(features, (features.shape[0]*features.shape[1], features.shape[2]))
    else: #not fwd bkwd, just getting one featureset?
        features = np.loadtxt(os.path.join(folderpath, files.featureString))
    return features

#deprecated
def getActualsPredictions(folderpath, model_type, fwd_bkwd=None,
                          limit_by=None, limits=None, means=None, stds=None,
                          opt_load_predictions=None):
    acts, preds, _ = getActsPredsFeats(folderpath, model_type, fwd_bkwd,
                          limit_by, limits, means, stds, opt_load_predictions)
    return acts, preds
    #limit_by can be LaneType, movetype, or distance 
    # and limits can be (min, max) for distance, or type of move, or list of lanetypes to use
    files = filenames()
    if folderpath == "":
        folderpath = files.default_path
    strFwdBkwd = ""
    if fwd_bkwd:
        strFwdBkwd = fwd_bkwd
    actfile = folderpath + strFwdBkwd + files.targetString
    if fwd_bkwd and not "LSTM" in model_type:
        actuals = np.concatenate([np.loadtxt(folderpath + str(i) + os.sep + files.targetString) for i in range(5)])
    if model_type in ["SVM", "DNN", "Marginal", "Conditional"]:
        predfile = folderpath + model_type + os.sep + strFwdBkwd + "YPred_" + model_type
    elif model_type == "BN":
        predfile = folderpath + strFwdBkwd + files.BNPreds
    elif "LSTM" in model_type:
        predfile = folderpath + model_type + os.sep + strFwdBkwd + files.LSTMPreds + model_type
        actfile = folderpath + model_type + os.sep + strFwdBkwd + files.LSTMActuals
    else:
        print("Invalid model type of:", model_type)
        return 0, 0, 0
    predictions = np.loadtxt(predfile)
    if "LSTM" in model_type:
        actuals = np.load(actfile)
        features = getFeatures(folderpath, model_type, fwd_bkwd)
        actuals, predictions, _ = slice_and_dice(actuals, predictions, features,
                                                 limit_by, limits, means, stds)
        actuals, predictions, _ = slice_and_dice(actuals, predictions, features=[],
                                                 limit_by="move", limits="!0",
                                                 means=means, stds=stds)
    elif not fwd_bkwd:
        actuals = np.loadtxt(actfile)
        features = getFeatures(folderpath, model_type, fwd_bkwd)
        actuals, predictions, _ = slice_and_dice(actuals, predictions, features,
                                              limit_by, limits, means, stds)
    return actuals, predictions

def slice_and_dice(actuals, predictions, features, limit_by, limits, means=None, stds=None):
    #limit_by can be LaneType, movetype, or distance 
    # and limits can be (min, max) for distance, or type of move, or list of lanetypes to use
    indexes = []
    if limit_by == None:
        return actuals, predictions, features
    if limit_by == "dist":
        if limits == ">0":
            if features == []:
                print("Error, passed in no features")
                return actuals, predictions, features
            mu, s = du.normalize_get_params(features)
            not_normed = False
            for m in mu:
                if abs(m) > 1:
                    not_normed = True
                    break
            if len(features[0,:]) < 13:
                distInd = 7
            else:
                distInd = 11
            if not not_normed: #thus normed
                features[:, distInd] = du.unnormalize(features[:,distInd], means[distInd], stds[distInd])
            min_dist = min(features[:,distInd])
            indexes = [i for i in range(len(predictions)) if features[i, distInd] > min_dist]
        else:
            print("Unsupported slice limits, returning unsliced")
    elif limit_by == "move":
        if limits == "!0":  #for use with LSTMs
            print(set(actuals.flatten()))
            if len(set(actuals.flatten())) >= 4:
                indexes = [i for i in range(len(actuals)) if actuals[i] > 0]
    else:
        print("Unsupported slice limit_by, returning unsliced")
    return getIndexesFrom(actuals, predictions, features, indexes) 

def getIndexesFrom(actuals, predictions, features, indexes):
    if len(indexes) == 0:
        return actuals, predictions, features
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    if features == []:
        return actuals[indexes], predictions[indexes], features
    features = np.array(features)
    return actuals[indexes], predictions[indexes], features[indexes]

def getMeanAndLoadfolder_etc(folderpath, model_type, fwd_bkwd):
    test_folder = folderpath
    model = model_type
    print("Scoring for model:", model, "folderpath:", test_folder)
    end = test_folder[-2:] #"i/"
    if "Test1" in test_folder or "Test2" in test_folder:
        if "LSTM" in model:
            other_folder = test_folder[:-len("LSTM_formatted/i/")]
            mean, std = du.get_norm_params_from_file("", other_folder)
        else:
            mean, std = du.get_norm_params_from_file("", test_folder[:-len(end)])
        load_folder = test_folder
    else:
        if "fwd" in fwd_bkwd:
            if "est3" in test_folder:
                normNum = 1
                loadNum = 3
            else:
                normNum = 2
                loadNum = 4
        else: #bk, test on lank
            if "est3" in test_folder:
                normNum = 3
                loadNum = 1
            else:
                normNum = 4
                loadNum = 2
            #actuals, pred, featurs from Test1/2
            #norm from Test 3/4
        if "LSTM" in model:
            norm_folder = test_folder[:-len("i/LSTM_formatted/")] + str(normNum) + os.sep + "LSTM_formatted" + os.sep
            load_folder = test_folder[:-len("i/LSTM_formatted/")] + str(loadNum) + os.sep + "LSTM_formatted" + os.sep
        else:
            norm_folder = test_folder[:-len("i/")] + str(normNum) + os.sep 
            load_folder = test_folder[:-len("i/")] + str(loadNum) + os.sep 
        other_folder = norm_folder[:-len("LSTM_formatted/i/")] + str(normNum) + os.sep

        if "LSTM" in model:
            mean, std = du.get_norm_params_from_file("", other_folder)
        else:
            mean, std = du.get_norm_params_from_file("", norm_folder)
    opt_load_predicitions = None #this is needed cause I did my whole format wrong... iDumb
    if fwd_bkwd and "b" in fwd_bkwd:
        opt_load_predicitions = norm_folder #this will be the test3/4 where the prediciton file is
    if "BN" in model:
        if fwd_bkwd:
            if "fwd" in fwd_bkwd:
                fwd_bkwd = "peach/"
            else: #bkwd, test on Lank, need get from other
                fwd_bkwd = "lank/"
    return mean, std, load_folder, opt_load_predicitions, fwd_bkwd

def do_score_count(folderpath, p_dist=False, offset=True, model_type="SVM",
                   fwd_bkwd=None, limit_by=None, limits=None):
    #limit_by can be LaneType, movetype, or distance 
    # and limits can be (min, max) for distance, or type of move, or list of lanetypes to use
    #fwd means train on lank test on peach
    mean, std, load_folder, opt_load_predicitions, fwd_bkwd = getMeanAndLoadfolder_etc(folderpath, model_type, fwd_bkwd)
    actuals, predictions = getActualsPredictions(load_folder, model_type, fwd_bkwd, limit_by, limits, mean, std, opt_load_predicitions)
    if max(set(actuals.flatten())) > 2:
        actuals = np.array([i-1 for i in actuals])
    if p_dist and len(predictions[0]) == 4:
        predictions = predictions[:,1:]
    if not p_dist:
        if max(set(predictions)) > 2:
            predictions = [i-1 for i in predictions]
    score = findCrossEntropyScore(predictions, actuals, p_dist, offset)
    accuracy = findAccuracy(predictions, actuals, p_dist=p_dist, offset=offset)
    precision = findPrecision(actuals, predictions, brokenDown=True,
                              p_dist=p_dist, offset=offset)
    return score, accuracy, precision
        
#new score is mean squared error
def findMeanSquaredError(predictions, actuals, p_dist=False):
    mse = 0
    n = len(predictions)
    n2 = float(n)
    print(n, len(actuals))
    if p_dist:
        for i in range(n):
            p_right = predictions[i, int(actuals[i])]
            p_right = p_right / sum(predictions[i])
            if np.isnan(p_right):
                p_right = 0
            p_wrong = 1.0 - p_right
            mse += (p_wrong ** 2) / n2
    else:
        for i in range(n):
            if predictions[i] != int(actuals[i]):
                mse += 1.0 / n2
    return mse

''' function: findCrossEntropyScore
      args: 
        predictions: numpy array of shape (num_predictions, x)
            if probability distribution x = number of possible classifications
                otherwise x = 1
        actuals: numpy array of shape (num_predictions,)
        p_dist: true if predictions are a probability distribution
        offset: true if actuals need to be offset to use as indexes
        
      returns: cross entropy score
'''
def findCrossEntropyScore(predictions, actuals, p_dist=False, offset=True):
    if max(set(actuals)) > 2:
        actuals = [i-1 for i in actuals]
    if p_dist:
        return crossEntropyWithProbs(predictions, actuals)
    return crossEntropyWithNoProbs(predictions, actuals)
    
def loglikelihood(predictions, actuals):
    loglikelihood = 0
    for i in range(len(predictions)):
        p_right = predictions[i, int(actuals[i])] / sum(predictions[i,:])
        if np.isnan(p_right) or p_right == 0:
          p_right = 1e-20
        loglikelihood += math.log(p_right)
    return loglikelihood/len(predictions)

def crossEntropyWithProbs(predictions, actuals):
    if len(set(actuals)) == 2:
        actuals[-1] = 2
    #to just get likelihood
    return loglikelihood(predictions, actuals)
    #return metrics.log_loss(actuals,predictions)
    loss = 0
    for i in range(len(predictions)):
        p_right = predictions[i, int(actuals[i])] 
        p_right = p_right / sum(predictions[i])  #in case do not sum to 1
        if np.isnan(p_right) or p_right == 0:
          p_right = 1e-20
        #p_wrong = 1.0 - p_right
        loss += math.log(p_right)
    return -loss/len(predictions)
    
def crossEntropyWithNoProbs(predictions, actuals):
    p_dists = np.zeros((len(actuals),int(max(set(predictions)))+1), float)
    for i in range(0, len(actuals)):
        p_dists[i] = [1e-20] * len(p_dists[i])
        p_dists[i, int(predictions[i])] = 1.0 - (2*1e-20)
        
    return crossEntropyWithProbs(p_dists, actuals)


def convertProbabilityToInt(p_dists):
    newPreds = []
    for i in range(len(p_dists)):
        index = 0
        maxp = 0
        index, maxp = max(enumerate(p_dists[i,:]), key=operator.itemgetter(1))
        newPreds.append(index)
    return newPreds
    
def convertIntToProbability(predictions):
    numMoves = max(predictions)-1
    p_dists = np.zeros((len(predictions),numMoves))
    for i in range(len(predictions)):
        p_dists[i,predictions[i]-1] = 1
    return p_dists

''' Usage: 
    accuracy = findAccuracy(predictions, actuals, p_dist, offset)
    Accuracy and precision are mathematically equivalent the way I am doing it
'''
def findAccuracy(predictions, actuals, p_dist=False, offset=False,
                 verbose=True):
    return findPrecision(actuals, predictions, brokenDown=False, p_dist=p_dist, offset=offset)

def print_confusion_matrix(actuals, predictions, features):
    cm = np.zeros(shape=(3,3), dtype=int)
    if features != None:
        if len(features.shape) == 3:
            features = features.reshape((features.shape[0]*features.shape[1], features.shape[2]))
        distInd = 11
        if features.shape[1] <= 11:
            distInd = 7
        for i in range(len(predictions)):
            dist = int(features[i, distInd])
            if dist > 20:continue
            move = actuals[i] #row index
            pred = predictions[i]
            cm[move][pred]+=1
    else:
        for i in range(len(predictions)):
            move = actuals[i] #row index
            pred = predictions[i]
            cm[move][pred]+=1
    print("Printing confusion matrix:")
    print(cm)

#without broken down, returns precision per class, otherwise is just overall accuracy
# offset is deprecated
def findPrecision(actuals, predictions, brokenDown=False, p_dist=True, offset=False, feats=None):
    if p_dist:
        predictions = convertProbabilityToInt(predictions)
    minAct = min(actuals)
    if minAct > 0:
        actuals = [int(i)-minAct for i in actuals]
    minPred = min(predictions)
    if minPred > 0:
        predictions = [int(i)-minPred for i in predictions]
    precisionCountsDict = {} #type: (correct predictions of type, total predictions of type)
    try:
        movetypes = set(actuals)
    except:
        movetypes = set(np.array(actuals).flatten())
    for move in movetypes:
        precisionCountsDict[move] = [0,1]  #prior such that division works
    for i in range(len(predictions)):
        move = actuals[i]
        pred = predictions[i]
        if not pred in precisionCountsDict.keys():
            precisionCountsDict[pred] = [0,0]
        if move == pred:
            precisionCountsDict[move][0] += 1
        precisionCountsDict[move][1] += 1
    precisionDict = {}
    precisionCounts = [0,0]
    for key in precisionCountsDict.keys():
        precisionCounts[0] += precisionCountsDict[key][0]
        precisionCounts[1] += precisionCountsDict[key][1]
        try:
            precisionDict[key] = precisionCountsDict[key][0]/precisionCountsDict[key][1]
        except:
            print(key)
            precisionDict[key] = precisionCountsDict[key][0]/(precisionCountsDict[key][1]+1)
    if brokenDown:
        #print(precisionCountsDict)
        print_confusion_matrix(actuals,predictions,feats)
        return precisionDict
    accuracy = precisionCounts[0] / precisionCounts[1]
    return accuracy

def trainTestConditional(Xtrain, Ytrain, Xtest, Ytest, testnum, save_path=None):
    start = time.clock()
    count_moves = {}
    minmove = min( min(Ytest), min(Ytrain))
    if minmove > 0: #offset because 0 indexed, but moves 1 indexed
        Ytest = [i-minmove for i in Ytest]
        Ytrain = [i-minmove for i in Ytrain]
    numMoves = max(max(Ytrain), max(Ytest)) + 1
    for i in range(len(Ytrain)):
        _, laneTypeEncoding = getLane(Xtrain[i])
        move = Ytrain[i]
        laneType = ""
        for j in laneTypeEncoding:
            laneType += str(int(j))
        if not laneType in count_moves.keys():
            count_moves[laneType] = [0] * (numMoves)
        count_moves[laneType][move] += 1
    p_moves = {}
    for lanetype in count_moves.keys():
        if not lanetype in p_moves.keys():
            p_moves[lanetype] = [0] * (numMoves)
        num_this = sum(count_moves[lanetype])
        p_moves[lanetype] = [count_moves[lanetype][i] / num_this for i in range(numMoves)]
    end = time.clock()
    timeFit = end - start
    start = time.clock()
    numTest = len(Ytest)
    Ypred = [[0] * numMoves] * numTest
    for i in range(numTest):
        _, laneTypeEncoding = getLane(Xtest[i])
        laneType = ""
        for j in laneTypeEncoding:
            laneType += str(int(j))
        if laneType in p_moves.keys():
            Ypred[i] = p_moves[laneType]
        else:
            Ypred[i] = [1/numMoves] * (numMoves)  
    end = time.clock()
    timePred = end-start
    return Ypred, timeFit, timePred
    
def trainTestMarginal(Xtrain, Ytrain, Xtest, Ytest, testnum, save_path=None):
    start = time.clock()
    count_moves = {}
    minmove = min( min(Ytest), min(Ytrain))
    if minmove > 0:
        Ytest = [i-minmove for i in Ytest]
        Ytrain = [i-minmove for i in Ytrain]
    numMoves = max(max(Ytrain), max(Ytest)) + 1
    count_moves = [0] * numMoves
    for i in Ytrain:
        count_moves[i] += 1
    n_samples = len(Ytrain)
    p_moves = [count_moves[i]/n_samples for i in range(len(count_moves))]
    end = time.clock()
    timeFit = end - start
    start = time.clock()
    Ypred = [p_moves] * len(Ytest)
    end = time.clock()
    timePred = end-start
    return Ypred, timeFit, timePred

def getLane(featureVector):
    lanesToSides = featureVector[:2]
    if len(featureVector >= 10):
        laneTypeEncoding = featureVector[2:6]
        for i in range(len(laneTypeEncoding)):
            if abs(round(laneTypeEncoding[i]) - 1) < 1e-10 or round(laneTypeEncoding[i]) > 1:
                laneTypeEncoding[i] = str(int(1))
            else:
                laneTypeEncoding[i] = str(int(0))
        #laneTypeEncoding = [int(round(i)) for i in laneTypeEncoding] # this was not working randomly
    else: #test 1.1
        laneTypeEncoding = [1,1,1,0]
    return lanesToSides, laneTypeEncoding
