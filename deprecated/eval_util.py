# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:53:05 2016

utilities to help with evaluating the SVM 
@author: LordPhillips
"""

import os
import sys
sys.path.append(os.environ["INTENTPRED_PATH"])

from sklearn import metrics
from utils import constants as c
from utils import data_util as du
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython import display
import time

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

def doAllPlotsForTest(test_folder, models, testID, limit_by=None, limits=None,
                      plot_savepath=None):
    #limit_by can be LaneType, movetype, or distance 
    # and limits can be (min, max) for distance, or type of move, or list of lanetypes to use
    accuracies = {}
    precisions = {}
    scores = {}
    plot_types = ["Confusion", "DistanceH", "Lanetype"]
    #plot_types = ["Lanetype"]
    if plot_savepath == None:
        plot_savepath = test_folder + "plots" + os.sep
    if limit_by:
        plot_savepath = plot_savepath + str(limit_by) + "_" + str(limits) + os.sep
    title = testID
    doAvg = True
    if "Test3" == testID or "Test4" == testID:
        doAvg = True
    for model in models:
        this_folder = test_folder
        this_title = title + "_" + model
        if "LSTM" in model:
            this_folder += "LSTM_formatted/"
        #plotAll
        plot(this_folder, model, plot_types, limit_by, limits, this_title, plot_savepath)
        #add the accuracy and precision to the overall
  
'''        accuracy, precision, loss = score(this_folder, model, doAvg, limit_by, limits)
        accuracies[model] = accuracy
        precisions[model] = precision
        scores[model] = loss
    print(accuracies)
    print(precisions)
    print(scores)
    #plotAccuracies(accuracies, models, title, plot_savepath)
    accuracies = [(key, accuracies[key]) for key in accuracies.keys()]
    precisions = [(key, precisions[key]) for key in precisions.keys()]
    scores = [(key, scores[key]) for key in scores.keys()]
    np.savetxt(plot_savepath+title+"Accuracy", accuracies, fmt="%s")
    np.savetxt(plot_savepath+title+"Precision", precisions, fmt="%s")
    np.savetxt(plot_savepath+title+"Scores", scores, fmt="%s")'''

def plotAccuracies(accuracies, models, title, plot_savepath, doAvg=True):
    colors = ['r', 'b', 'g', 'y', 'k', 'c', 'm', '0.75']
    width=0.8/len(models)
    i = 0
    x_poses = []
    for key in models:#accuracies.keys():
        x = [0]
        if not doAvg:
            x = list(range(len(accuracies[key])))
        x = [j + (width*i) for j in x]
        plt.bar(x, accuracies[key], width=width, color = colors[i])
        i += 1
        x_poses.extend(x)
    plt.xticks([i + width/2 for i in x_poses], models)
    plt.ylim((.40,1.00))
    this_title = title + "_overall_accuracies" 
    this_title = this_title.replace(".","-")
    plt.title(this_title)
    plt.savefig(plot_savepath+this_title)
    plt.show()

def plot(test_folder=None, model="SVM", plot_type="Confusion", limit_by=None,
         limits=None, title="", plot_savepath=None):
    #limit_by can be LaneType, movetype, or distance 
    # and limits can be (min, max) for distance, or type of move, or list of lanetypes to use
    this_title = title 
    fileclass = filenames()
    if not test_folder:
        test_folder = fileclass.default_default_path
    Ps = True
    if model == "SVM": Ps = False
    offset = True
    actuals = np.array([])
    predictions = np.array([])
    features = np.array([])
    if not plot_savepath:
        plot_savepath = test_folder + "plots/"
    if not os.path.exists(plot_savepath):
        os.makedirs(plot_savepath)
    if "LSTM" in model: offset = True # No longer necessary when removing 0's
    if "Test1" in test_folder or "Test2" in test_folder:
        if "LSTM" in model:
            other_folder = test_folder[:-len("LSTM_formatted/")]
            mean, std = du.get_norm_params_from_file("", other_folder, cvs=True)
        else:
            mean, std = du.get_norm_params_from_file("", test_folder, cvs=True)

        for i in range(5):
            folderpath2 = test_folder + str(i) + os.sep
            actuals2, predictions2, features2 = getActsPredsFeats(folderpath2, model, 
                                                                  None, limit_by, limits,
                                                                  mean, std)
            if actuals.size > 0:
                actuals = np.concatenate((actuals, actuals2))
                predictions = np.concatenate((predictions, predictions2))
                features = np.concatenate((features, features2))
            else:
                actuals = actuals2
                predictions = predictions2
                features = features2
        do_plot(actuals, predictions, plot_type, Ps, offset, this_title, features,
                mean, std, savepath=plot_savepath)
    else:
        for i in ["fwd/"]:#["fwd/", "bkwd/"]:  #fwd is train Lank test Peach.
            print(test_folder, model)
            mean, std, load_folder, opt_load_predicitions, i = getMeanAndLoadfolder_etc(test_folder, model, i)
            actuals, predictions, features = getActsPredsFeats(load_folder, model, i, 
                                limit_by, limits, means=mean, stds=std, opt_load_predictions=opt_load_predicitions)
            do_plot(actuals, predictions, plot_type, Ps, offset, 
                    this_title+"_"+i[:-1], features, mean=mean, std=std, savepath=plot_savepath)
    return actuals, predictions, Ps, offset

def do_plot(actuals, predictions, plot_type, Ps, offset, title,
            features=None, mean=None, std=None, savepath=None):
    print(len(predictions))
    print(len(actuals))
    print(len(features))
    if offset == True:
        actuals = [int(i)-1 for i in actuals]
    if Ps:
        predictions = convertProbabilityToInt(predictions)
    if min(predictions) == 1:
        predictions = [int(i)-1 for i in predictions]
    else:
        predictions = [int(i) for i in predictions]
    if len(set(predictions)) > 3 or max(set(predictions)) >= 3:
        predictions = [int(i)-1 for i in predictions]
    if type(plot_type) == list:
        plot_types = plot_type
        for plot_type in plot_types:
            wrap_plot(plot_type, actuals, predictions, features, title, savepath, mean, std)
    else:
        wrap_plot(plot_type, actuals, predictions, features, title, savepath, mean, std)
    return

def wrap_plot(plot_type, actuals, predictions, features, title, savepath, mean, stdev):
    this_title = title + "_" + plot_type
    if plot_type == "Confusion":
        heights = makeConfusionMatrix(actuals, predictions, title=this_title, savepath=savepath)
        np.savetxt(savepath+title+"_Confusion-heights", heights)
    elif plot_type == "DistanceH":
        accuracy_cords, count_cords = makeDistanceHistogram(actuals, predictions, features,
                                          mean, stdev, title=this_title, savepath=savepath)
        np.savetxt(savepath+title+"_DistanceH-accuracies", accuracy_cords)
        np.savetxt(savepath+title+"_DistanceH-counts", count_cords)
    elif plot_type == "Lanetype":
        accuracy_cords, count_cords, entries = makeLaneBar(actuals, predictions, features,
                                          mean, stdev, title=this_title, savepath=savepath)
        np.savetxt(savepath+title+"_Lanetype-accuracies", accuracy_cords)
        np.savetxt(savepath+title+"_Lanetype-counts", count_cords)
        np.savetxt(savepath+title+"_Lanetype-entries", entries, fmt="%s")
    else:
        print("Plot type", plot_type, "is unsupported.")
    return

'''creates a confusion matrix for the model. 
    X axis is the actual move type
    Y axis is the predicted move type
'''
def makeConfusionMatrix(actuals, predictions, percentage = True, swap=False,
                      title=None, savepath=None):
    counts = {}
    for i in range(len(actuals)):
        y = int(actuals[i])
        x = int(predictions[i])
        if not y in counts.keys():
            counts[y] = [0] * len(set(predictions))
        counts[y][x-1] += 1
    print(counts)
    width =  0.5
    colors = ['r', 'b', 'g', 'y', 'k', 'c', 'm', '0.75']
    numKeys = len(list(counts.keys()))
    numVals = len(counts[list(counts.keys())[0]])
    heights = [[0] * numKeys] * numVals  #yes, its flipped on purpose
    for key in sorted(list(counts.keys())):
        cumsum = 0
        tot = sum(counts[key])
        for i in range(numVals):
            num = counts[key][i]
            height = (tot - cumsum) 
            if percentage: height /= tot
            plt.bar(key, height, width, color=colors[i])
            heights[i][int(key)] = height
            cumsum += num
    plt.title(title)
    if title:
        plt.savefig(savepath+title, format="png")
    plt.show()

    return heights
    
'''features is of shape num_predictions x num_features
'''
def makeDistanceHistogram(actuals, predictions, features, means=None, stds=None,
                      nbins=25, title=None, savepath=None):
    #x = distance(rounded to an int), y = numwrong
    counts_dists_int = []
    wrong_dists_int = []
    mu, s = du.normalize_get_params(features)
    not_normed = False
    for m in mu:
        if abs(m) > 2:
            not_normed = True
            break
    if len(features[0,:]) < 13:
        distInd = 7
    else:
         distInd = 11
    if not not_normed: #thus normed
        features[:, distInd] = du.unnormalize(features[:,distInd], means[distInd], stds[distInd])
    for i in range(len(predictions)):
        dist = int(features[i,distInd])
        if not int(actuals[i]) == int(predictions[i]):
            wrong_dists_int.append(dist)
        counts_dists_int.append(dist)    
    max_dist = max(counts_dists_int)
    bin_width = max_dist / nbins
    bin_counts = {}
    bin_wrongs = {}
    bin_accuracies = {}
    count_ratios = {}
    for bin_n in range(nbins+1):
        bin_start = (bin_n - 1) * bin_width
        bin_end = bin_start + bin_width
        bin_pos = (bin_start + bin_end) / 2
        if not bin_pos in bin_counts.keys():
            bin_counts[bin_pos] = 0
            bin_wrongs[bin_pos] = 0
        for dist in wrong_dists_int:
            if dist > bin_start and dist <= bin_end:
                bin_wrongs[bin_pos] += 1
        for dist in counts_dists_int:
            if dist > bin_start and dist <= bin_end:
                bin_counts[bin_pos] += 1
    num_entries = len(counts_dists_int)
    for key in bin_counts.keys():
        if bin_counts[key] > 0:
            bin_accuracies[key] = 1.0 - (bin_wrongs[key]/bin_counts[key])
        count_ratios[key] = bin_counts[key] / num_entries
    width = bin_width 
    x = sorted([i for i in bin_accuracies.keys()])
    plt.bar(left=x, height=[bin_accuracies[key] for key in x],
            width=width, color='r')
    plt.bar(left=x, height=[count_ratios[key] for key in x],
            width=width, color='b')
    plt.ylim((0,1.00))
    plt.xlim((x[0],x[-1]+bin_width))
    plt.title(title)
    if title:
        plt.savefig(savepath+title, format="png")
    plt.show()
    return [(key, bin_accuracies[key]) for key in x], [(key, count_ratios[key]) for key in x]
    
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
    

def makeLaneBar(actuals, predictions, features, means=None, stds=None,
                      title=None, savepath=None):
    counts_laneT = {}
    wrong_laneT = {}
    mu, s = du.normalize_get_params(features)
    not_normed = False
    for m in mu:
        if abs(m) > 1:
            not_normed = True
            break
    if not not_normed: #thus normed
        for i in range(2,6):
            features[:, i] = du.unnormalize(features[:,i], means[i], stds[i])
    for i in range(len(predictions)):
        lanesToSide, laneTypeEncoding = getLane(features[i])
        laneType = ""
        for j in laneTypeEncoding:
            laneType += str(int(j))
        if not laneType in counts_laneT.keys():
            counts_laneT[laneType] = 0
            wrong_laneT[laneType] = 0
        counts_laneT[laneType] += 1
        if not int(actuals[i]) == int(predictions[i]):
            wrong_laneT[laneType] += 1
    entries = sorted(list(counts_laneT.keys())) #sorting only for consistency
    print("Entries, will be labels:", entries)
    num_entries = len(entries)
    accuracies = {}
    count_ratios = {}
    for key in entries:
        accuracies[key] = 1 - wrong_laneT[key] / counts_laneT[key]
        count_ratios[key] = counts_laneT[key] / sum(counts_laneT.values())
    width = 1
    x = list(range(num_entries))
    x = [i * 2 for i in x]
    plt.bar(left=x, height=[accuracies[key] for key in entries],
            width=width, color='r')
    plt.bar(left=x, height=[count_ratios[key] for key in entries],
            width=width, color='b')
    plt.ylim((0,1.00))
    plt.xticks([i + width/2 for i in x], entries)
    plt.title(title)
    if title:
        plt.savefig(savepath+title, format="png")
    plt.show()
    return [(accuracies[key]) for key in entries], [(count_ratios[key]) for key in entries], entries

def plotAccuracy(actuals, predictions, percentage=True, p_dist=True, offset=True):
    accuracy = findAccuracy(predictions, actuals, p_dist = p_dist, offset = offset)
    return accuracy
    #accuracy must be plot for all mdoels, yeah?

#testfolder is results/ByIntersection/000/ by default
#testInters is 1 by default
def score(test_folder=None, model="SVM", doAvg=False, limit_by=None, limits=None, testInters=[1]):
    #limit_by can be LaneType, movetype, or distance 
    # and limits can be (min, max) for distance, or type of move, or list of lanetypes to use
    fileclass = filenames()
    if not test_folder:
        test_folder = fileclass.default_default_path
    if model == "SVM":
        Ps = False
    else:
        Ps = True
    if "LSTM" in model:
        offset = False
    else:
        offset = True
    score, accuracy, precision = loadPrintScoreWrapper(test_folder, testInters, Ps, offset,
                                                       model, doAvg, limit_by, limits)
    return accuracy, precision, score


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
                          avg_over_CVs = False, limit_by=None, limits=None):
    #limit_by can be LaneType, movetype, or distance 
    # and limits can be (min, max) for distance, or type of move, or list of lanetypes to use
    files = filenames()
    if folderpath == "":
        folderpath = files.default_default_path
    scores = []
    precisions = []
    precision = 0
    accuracies = []
    accuracy = 0
    testnum = folderpath.split(os.sep)[-2]
    #for test_inter in testInters:
    if "LSTM" in model_type:
        features, actuals = du.getFeaturesLSTM(folderpath, testnum, testInters)
    else:
        features, actuals = du.getFeaturesnonLSTM(folderpath, testnum, testInters)
    predictions = np.loadtxt(folderpath + "TestOn" + ",".join([str(i) for i in testInters]) + os.sep + "Ypred_" + model_type)
    actuals = actuals.flatten()
    if max(set(actuals)) > 2:
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
    '''
    if "Test1" in folderpath or "Test2" in folderpath:
        for i in range(0, num_CV_folds):
            folderpath2 = folderpath + str(i) + os.sep 
            score, accuracy, precision = do_score_count(folderpath2, p_dist, offset, 
                                                        model_type, None, limit_by, limits)
            scores.append(score)
            accuracies.append(accuracy)
            precisions.append(precision)
            print("Model:", model_type, "CV:", i, "Score:", score, "Accuracy:", accuracy)
            print("Precision:", precision)
    elif "Test3" in folderpath or "Test4" in folderpath:
        accuracies = []
        for i in ["fwd/", "bkwd/"]: 
            score, accuracy, precision = do_score_count(folderpath, p_dist, offset, 
                                                        model_type, i, limit_by, limits)
            scores.append(score)
            accuracies.append(accuracy)
            precisions.append(precision)
            print("Model:", model_type, "CV:", i, "Score:", score, "Accuracy:", accuracy)
            print("Precision:", precision)
    if avg_over_CVs:
        accuracy = 0
        precision = 0
        score = 0
        for i in accuracies:
            accuracy += i / len(accuracies)
        precision = avgPrecision(precisions)
        for i in scores:
            score += i / len(scores)
        print("Averaged Accuracy:", accuracy, "Precision:", precision, "Score:", score)
    else:
        accuracy = accuracies
        precision = precisions
        score = scores
    '''
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
        actuals = np.concatenate([np.loadtxt(act_feat_path + str(i) + os.sep + files.targetString) for i in range(5)])
    if model_type in ["SVM", "DNN", "Marginal", "Conditional"]:
        predfile = folderpath + model_type + os.sep + strFwdBkwd + "YPred_" + model_type
        if opt_load_predictions:
            predfile = opt_load_predictions + model_type + os.sep + strFwdBkwd + "YPred_" + model_type
    elif model_type == "BN":
        predfile = folderpath + strFwdBkwd + files.BNPreds
        if opt_load_predictions:
            predfile = opt_load_predictions + strFwdBkwd + files.BNPreds
    elif "LSTM" in model_type:
        predfile = folderpath + model_type + os.sep + strFwdBkwd + "YPred_" + model_type
        #results/Test4/LSTM_formatted/LSTM_128x2/bkwd/Ypred_LSTM_128x2
        if opt_load_predictions:
            predfile = opt_load_predictions + model_type + os.sep + strFwdBkwd + "YPred_" + model_type
            
        actfile = act_feat_path + model_type + os.sep + strFwdBkwd + files.LSTMActuals
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
        features = np.concatenate([np.loadtxt(folderpath + str(i) + os.sep + files.featureString) for i in range(5)])
    elif "LSTM" in model_type:
        features = np.load(folderpath + model_type + os.sep + strFwdBkwd + files.LSTMFeatures)
        print("features from:", str(folderpath + model_type + os.sep + strFwdBkwd + files.LSTMFeatures))
        #features = np.reshape(features, (features.shape[0]*features.shape[1], features.shape[2]))
    else: #not fwd bkwd, just getting one featureset?
        features = np.loadtxt(folderpath + files.featureString)
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
    if p_dist:
        return crossEntropyWithProbs(predictions, actuals)
    return crossEntropyWithNoProbs(predictions, actuals)
    
def crossEntropyWithProbs(predictions, actuals):
    #currently binary log loss
    loss = 0
    for i in range(len(predictions)):
        p_right = predictions[i, int(actuals[i])] 
        p_right = p_right / sum(predictions[i])  #in case do not sum to 1
        if np.isnan(p_right) or p_right == 0:
          p_right = 1e-20
        #p_wrong = 1.0 - p_right
        loss += math.log(p_right)
    return -loss/len(predictions)
    #return metrics.log_loss(actuals,predictions)        
    
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
        for j in range(len(p_dists[i])):
            if p_dists[i,j] > maxp:
                index = j
                maxp = p_dists[i,j]
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

#without broken down, returns precision per class, otherwise is just overall accuracy
# offset is deprecated
def findPrecision(actuals, predictions, brokenDown=False, p_dist=True, offset=False):
    if p_dist:
        predictions = convertProbabilityToInt(predictions)
    #minAct = min(actuals)
    #if minAct > 0:
    #    actuals = [int(i)-minAct for i in actuals]
    #minPred = min(predictions)
    #if minPred > 0:
    #    predictions = [int(i)-minPred for i in predictions]
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
            precisionCountsDict[pred][0] += 1
        precisionCountsDict[pred][1] += 1
    precisionDict = {}
    precisionCounts = [0,0]
    for key in precisionCountsDict.keys():
        precisionCounts[0] += precisionCountsDict[key][0]
        precisionCounts[1] += precisionCountsDict[key][1]
        precisionDict[key] = precisionCountsDict[key][0]/precisionCountsDict[key][1]
    if brokenDown:
        return precisionDict
    accuracy = precisionCounts[0] / precisionCounts[1]
    return accuracy

def trainTestConditional(Xtrain, Ytrain, Xtest, Ytest, testnum, save_path=None):
    start = time.clock()
    count_moves = {}
    minmove = min( min(Ytest), min(Ytrain))
    if minmove > 0:
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
