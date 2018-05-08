# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:47:56 2016

@author: Derek
"""

#This fail is to be used with SUMO data
#It is designed to be used in conjunction with DriverDataProcessing, by calling
#various functions from it.

'''Main points:
    1:
    Re-format csv data to required form
    After doing so, train LSTM on it
    Save LSTM model
    
    2:
    Load LSTM model (specified or default path)
        #If this takes too long, may have to change plan

    2.5:
    Given LSTM model and test trajectory:
        output/print probability distribution over next turns
'''


from lib import LSTM
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.contrib.learn as skflow
import time


#converters

def convertVID(vid):
    if "." in str(vid):
        return (int(float(vid)*(10*len(str(vid))-2)))
    return int(float(vid))

def convertTimeToFID(time):
    return int(float(time) * 10)

def convertYaw(yaw):
    return float(yaw)
    
def convertLane(lanes):
    try:
        return int(lanes)
    except:
        return -1 

def convertHdwy(obj):
    try:
        return float(obj)
    except:
        return 1000.0    

def convertMove(move):
    if move == b'"left"':
        return 1
    elif move == b'"right"':
        return 2
    elif move == b'"straight"':
        return 0
    else:
        #print("Error move", move)
        return 0

#given the csv filepath, load the file and reformat, returning Xtrain, Ytrain
def loadReformatCSV(csv_file="intersection3/refined_turning_data.csv"):
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1,\
            converters = {0:convertVID, 1:convertTimeToFID, 6:convertYaw, 7:convertLane, 
                          8:convertLane, 9:convertHdwy, 11:convertMove})
    #new_ordering = [7, 8, 3, 5, 2, 4, 6, 9, 10, 1, 0, 11]
    new_ordering = [7, 8, 3, 5, 2, 4, 6, 9, 10, 1, 0, 11]
    data = data[:,new_ordering]
    #data should be: (doesnt actually matter too much, but might as well)
        #[lanesToMedian, lanesToCurb,Vy, Ay, Vx, Ax, yaw, hdwy, dist, fid, vid, move] 
    #features will be shape (numInputs, trajectory_len, numFeatures)
    np.random.shuffle(data) #to make IID
    traj_len = 10 #at .2s timesteps = 2 seconds trajectory info
    num_features = 9
    vidToFramesToFeatures = {}
    for i in range(len(data)):
        fid = data[i,9]
        vid = data[i,10]
        if not vid in vidToFramesToFeatures:
            vidToFramesToFeatures[vid] = {fid: data[i]}
        else:
            vidToFramesToFeatures[vid][fid] = data[i]
    Xtrain = np.array([])
    Ytrain = np.array([])
    for vid in sorted(list(vidToFramesToFeatures.keys())):
        fids = sorted(list(vidToFramesToFeatures[vid].keys()))
        i = 0
        fid = fids[i]
        feats = vidToFramesToFeatures[vid][fid]
        startYaw = feats[6]
        admissablesFeatsThisVID = []
        while abs(startYaw - feats[6]) <= 1.0: #only add trajectory on approach to intersection
            admissablesFeatsThisVID.append(feats)
            i+= 1
            if i >= len(fids): break
            fid = fids[i]
            feats = vidToFramesToFeatures[vid][fid]
        j = 0
        n = len(admissablesFeatsThisVID)
        while j + traj_len < n:
            this_traj = np.ascontiguousarray(admissablesFeatsThisVID[j:j+traj_len])
            this_traj = this_traj[:,:-3]
            turn = np.array(admissablesFeatsThisVID[j:j+traj_len])
            turn = turn[:,-1]
            this_traj = this_traj.reshape(1, traj_len, num_features)
            turn = turn.reshape(1, traj_len, 1)
            if len(Xtrain) == 0:
                Xtrain = this_traj
                Ytrain = turn
            else:
                Xtrain = np.vstack((Xtrain, this_traj))
                Ytrain = np.vstack((Ytrain, turn))
            j += 1
    print(Xtrain.shape, Ytrain.shape)
    return Xtrain, Ytrain
                

def trainSaveLSTM(Xtrain, Ytrain, model="LSTM_128x2"):
    save_path = os.getcwd() + os.sep + model + os.sep
    check_make_paths([save_path])
    LSTM.run_LSTM((Xtrain, Ytrain), [], model, save_path, train_only=True)
    print(save_path)
    return

def loadDataTrainSaveLSTM(csv_file="intersection3/refined_turning_data.csv"):
    print("Loading Xtrain and Ytrain from:", csv_file)
    Xtrain, Ytrain = loadReformatCSV(csv_file)
    means, stddevs = normalize_get_params(Xtrain)
    Xtrain = normalize(Xtrain, means, stddevs)
    np.savetxt(csv_file[:-4]+"_norm_params.txt", np.array([means, stddevs]))
    print("Done loading Xtrain and Ytrain")
    trainSaveLSTM(Xtrain,Ytrain)
    print("Done training and saving LSTM")
    return

#give numpy arrays (X_1, X_2), (Y_1, 1)
def input_fn(X,Y=None): # returns x, y (where y represents label's class index).
    X_t = {"": tf.constant(X)}
  #                for i in range(X.shape[1])}
    if Y != None:
        return X_t, tf.constant(Y)
    return tf.constant(X)


def trainDNN(Xtrain, Ytrain, model="DNN"):
    tf.logging.set_verbosity(tf.logging.ERROR)
    modeldir = os.getcwd() + os.sep + model + os.sep
    check_make_paths([modeldir])
    classifier = skflow.DNNClassifier(
        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(Xtrain),
        hidden_units=[128, 128], n_classes=3, model_dir=modeldir)
    #print(tf.contrib.learn.infer_real_valued_columns_from_input(Xtrain))
    #return
    #classifier.evaluate(input_fn=input_fn_eval)
    #classifier.predict(x=x) # returns predicted labels (i.e. label's class index).
    Ytrain = [int(i) for i in Ytrain]
    start = time.clock()
    #classifier.fit(input_fn=lambda: input_fn(Xtrain, Ytrain))
    max_epochs = 10
    start2 = time.clock()
    for epoch in range(max_epochs):
        classifier.fit(input_fn=lambda: input_fn(Xtrain, Ytrain),steps=1000)
        loss = testDNN(Xtrain, classifier=classifier, Y=Ytrain)
        end2 = time.clock()
        print("Epoch",epoch,"Done. Took:", end2-start2, "loss of:", loss)
        start2 = end2
    end = time.clock()
    timeFit = end - start
    print("Done fitting, time spent:", timeFit)
    print("Done saving the model")
    testDNN(Xtrain, classifier=classifier, Y=Ytrain)



def testDNN(X, model="DNN", classifier=None, Y=None):
    modeldir = os.getcwd() + os.sep + model + os.sep
    if classifier == None:
      classifier = skflow.DNNClassifier(
        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X),
        hidden_units=[128, 128], n_classes=3, model_dir=modeldir)
    print("classifier created")
    if Y != None:
        ev = classifier.evaluate(input_fn=lambda: input_fn(X, Y), steps=1)
        loss_score = ev["loss"]
        print("Loss: {0:f}".format(loss_score))
        return loss_score
    probs = (classifier.predict_proba(X))#input_fn=lambda: input_fn(X)))
    print("PROBS:", probs)
    for i in probs:
        print(i)
    return probs

def getXYDNN(csv_file="intersection3/refined_turning_data.csv"):
    Xtrain, Ytrain = loadReformatCSV(csv_file)
    print(Xtrain.shape)
    print(Ytrain.shape)
    Xtrain = Xtrain[:,-1,:]
    Ytrain = Ytrain[:,-1,:]
    print(Xtrain.shape)
    print(Ytrain.shape)    
    means, stddevs = normalize_get_params(Xtrain)
    Xtrain = normalize(Xtrain, means, stddevs)
    return Xtrain, Ytrain

def loadDataTrainSaveDNN(csv_file ="intersection3/refined_turning_data.csv"):
    Xtrain, Ytrain = loadReformatCSV(csv_file)
    print(Xtrain.shape)
    print(Ytrain.shape)
    Xtrain = Xtrain[:,-1,:]
    Ytrain = Ytrain[:,-1,:]
    print(Xtrain.shape)
    print(Ytrain.shape)    
    means, stddevs = normalize_get_params(Xtrain)
    Xtrain = normalize(Xtrain, means, stddevs)
    np.savetxt(csv_file[:-4]+"_norm_params.txt", np.array([means, stddevs]))
    print("Done loading Xtrain and Ytrain")
    trainDNN(Xtrain,Ytrain)


#model_load_path should be: "abc/def/128x2.meta"
#X of shape (numInputs=1(probably), traj_len, numFeatures)    
def getBelief(X, filepath=None, model="LSTM_128x2"):
    if "LSTM" in model:
        if filepath == None:
            filepath = os.getcwd() + os.sep + model + os.sep + model[len("LSTM_"):]+".meta"
        p_dists = LSTM.run_LSTM_testonnly(X, filepath, model)
    else:
        if filepath == None:
            filepath = os.getcwd() + os.sep + model + os.sep
        p_dists = getBeliefDNN(X)
    return p_dists #only care about the last one, but outputs a prob. distr
   
def check_make_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
   
def normalize_get_params(X):
    shape = X.shape
    numFeatures = shape[-1]
    means = np.zeros((numFeatures,))
    stddevs = np.zeros((numFeatures,))
    for i in range(numFeatures):
        if len(shape) == 3:
            a = X[:,:,i]
        elif len(shape) == 2:
            a = X[:,i]
        else:
            print("Unhandled shape:", shape)
            return [0], [1]
        means[i] = np.mean(a)
        stddevs[i] = np.std(a)
    return means, stddevs
            
def unnormalize(x, mean, stddev):
    return (x * stddev) + mean
    
def normalize(X, means, stddevs):
    try:
        return (X-means)/stddevs   
    except:
        for col in range(X.shape[1]):
            for row in range(X.shape[0]):
                X[row,col] = float(X[row,col])-means[col]
        return X / stddevs
        

def load_model_params(X, model_load_path=None, model="LSTM_128x2"): #Xtrain for sizing
    traj_len = X.shape[1]
    in_size = X.shape[2]
    out_size = 1
    if model_load_path == None:
        model_load_path = os.getcwd() + os.sep + model + os.sep + model[len("LSTM_"):]+".meta"
    session, mtest = LSTM.get_vars_to_test(model_load_path, in_size, out_size, traj_len,
                                           model="128x2", num_classes=3)
    return session, mtest

def getBeliefModel(X, session, mtest, mean, stdev, Xtrain=None):
    if mean == None:
        mean, stdev = normalize_get_params(Xtrain)
    X = normalize(X, mean, stdev)
    p_dists = LSTM.predict(X, session, mtest)
    return p_dists[-1] #only care about the last one, but outputs a prob. distr

def countWrong(p_dists, Y):
    numWrong = 0
    n = 0
    for traj_i in range(Y.shape[0]):
        last_move = int(Y[traj_i][0][0])
        p_dist = p_dists[traj_i][0]
        if max(p_dist) != p_dist[last_move]:
            numWrong += 1
        n += 1
    return numWrong, n

def countWrongLinear(p_dists, Y):
    numWrong = 0
    n = 0
    Y = Y.flatten()
    for traj_i in range(len(Y)):
        if max(p_dists[traj_i]) != p_dists[traj_i][int(Y[traj_i])]:
            numWrong += 1
        n += 1
    return numWrong, n

def loadDNNonly(modeldir=os.getcwd()+os.sep+"DNN"+os.sep):
    tf.logging.set_verbosity(tf.logging.ERROR)
    X, Ytrain = getXYDNN("refined_turning_data.csv")
    classifier = skflow.DNNClassifier(
        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X),
        hidden_units=[128, 128], n_classes=3, model_dir=modeldir)
    return classifier

def getBeliefDNN(X, classifier=None):
    if not classifier:
        modeldir = os.getcwd() + os.sep + "DNN" + os.sep
        classifier = skflow.DNNClassifier(
            feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X),
            hidden_units=[128, 128], n_classes=3, model_dir=modeldir)
    probs = classifier.predict_proba(X)
    probs_list = [i for i in probs]
    return probs_list

def testJohnsDNNBelief():
    tf.logging.set_verbosity(tf.logging.ERROR)
    Xtrain, Ytrain = getXYDNN("refined_turning_data.csv")
    numWrong = 0
    n = 0
    start = time.time()
    for i in range(100):#len(Ytrain)):
        X = Xtrain[i]
        X = X.reshape((1, 9))
        probs = getBeliefDNN(X)[0]
        if i == 0:
            print(probs)
        if max(probs) != probs[int(Ytrain[i][0])]:
            numWrong += 1
        n += 1
    end = time.time()
    print(numWrong, "wrong /", n, "== accuracy of:", 1-(float(numWrong) / n))
    print("scoring took:", end-start, "a time of :", (end-start)/n, "per example")

def testAccuracyDNN():
    tf.logging.set_verbosity(tf.logging.ERROR)
    Xtrain, Ytrain = getXYDNN("refined_turning_data.csv")
    DNNgetAccuracy(Xtrain, Ytrain)

def DNNgetAccuracy(X, Y_for_score):
    modeldir = os.getcwd() + os.sep + "DNN" + os.sep
    check_make_paths([modeldir])
    classifier = skflow.DNNClassifier(
        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X),
        hidden_units=[128, 128], n_classes=3, model_dir=modeldir)
    probs = classifier.predict_proba(X)
    probs_list = [i for i in probs]
    print(probs_list[0])
    numWrong, n = countWrongLinear(probs_list, Y_for_score)
    print(numWrong, "wrong /", n, "== accuracy of:", 1-(float(numWrong) / n))
    return (1-float(numWrong) / n)



def run(model="LSTM_128x2"):
    random.seed(42)
    #loadDataTrainSaveLSTM("refined_turning_data.csv")
    Xtrain, Ytrain = loadReformatCSV("refined_turning_data.csv")
    Y = Ytrain[:,:-1,:]
    print(set(Y.flatten()))
    mean, stdev = normalize_get_params(Xtrain)
    numWrong = 0
    n = 0
    Xtrain = normalize(Xtrain, mean, stdev)
    p_dists = getBelief(Xtrain, model=model)
    print(p_dists)
    #p_dists = p_dists.reshape((Y.shape[0], Y.shape[1], p_dists.shape[-1]))
    '''for ordering in [[0,1,2], [0,2,1], [1,0,2],[1,2,0],[2,1,0],[2,0,1]]:
        print(ordering)
        p = p_dists[:,ordering]
        print(p[0])
        numWrong, n = countWrongLinear(p, Y)
        print(numWrong, "/", n, "== ", float(numWrong) / n)
    '''
    for input_i in range(Xtrain.shape[0]):
        X = Xtrain[input_i]
        X = X.reshape(1, X.shape[0], X.shape[1])
        X = normalize(X, mean, stdev)
        pdist = getBelief(X, model=model)
        if pdist[int(Ytrain[input_i][-1][0])-1] != max(pdist):
            numWrong += 1
        n += 1
        if n % 100 == 0:
            print(numWrong, n)
    return

#when you have a trajectory X, in shape (1, trajectory_len, num_features)
#will output probability distribution over the moves
def johnsfunction(X):
    return getBelief(X)[-1]

# X shape (x, 11)
def convertInput(X):
    converters = {0:convertVID, 1:convertTimeToFID, 6:convertYaw, 7:convertLane, 
                          8:convertLane, 9:convertHdwy, 11:convertMove}
    for row in range(X.shape[0]):
        for col in converters.keys():
            X[row,col] = converters[col](X[row,col])
    return X
    
#pass in the current features only to getBelief: shape (1, numfeatures=9)
def johngetDNNbelief(X, classifier=None):
    tf.logging.set_verbosity(tf.logging.ERROR)
    #vid, fid, vx, vy, ax, ay, yaw, lanesMed, lanesCurb, hdwy, dist, 0 
    #to lanesMed, lanesCurb, Vy, Ay, Vx, Ax, yaw, hdwy, dist, fid, vid
    new_ordering = [7, 8, 3, 5, 2, 4, 6, 9, 10, 1, 0, 11]
    X = np.array(X)
    if len(X.shape) == 1:
        X = X.reshape((1, len(X)))
    X = convertInput(X)
    X = X[:,new_ordering]
    X = X[:,:9]
    X = np.array(X, dtype=float)
    m, s = np.loadtxt('refined_turning_data_norm_params.txt')
    X = normalize(X, m, s)
    return getBeliefDNN(X, classifier)

    

#loadDataTrainSaveDNN("refined_turning_data.csv")

#testAccuracyDNN()