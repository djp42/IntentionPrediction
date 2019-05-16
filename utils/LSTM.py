# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:19:42 2016

@author: LordPhillips
based on TensorFlow LSTM Tutorial
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#import tensorflow.nn.rnn_cell as rnn_cell
    

import time
import math
import numpy as np
import random
from collections import Counter
from utils import score_util as sutil

#from tensorflow.models.rnn.ptb import reader

#flags = tf.flags
#logging = tf.logging

#flags.DEFINE_string(
#    "model", "small",
#    "A type of model. Possible options are: small, medium, large.")
#flags.DEFINE_string("data_path", None, "data_path")
#flags.DEFINE_bool("use_fp16", False,
#                  "Train using 16-bit floats instead of 32bit floats")



def data_type():
  return tf.float64


class Model(object):
  """The model."""

  def __init__(self, is_training, config, input_size=10, output_size=1):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self._is_training = is_training
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.float64, [batch_size, num_steps, input_size])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps, output_size])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())
    inputs = self._input_data
    
    '''
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)
    '''
    
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output) 

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    softmax_out = tf.nn.softmax(logits)
    
    
    '''
    logits: List of 2D Tensors of shape [batch_size*num_steps x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    def sequence_loss_by_example(logits, targets, weights,...)
    '''
    loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
        [tf.reshape(self._targets, [-1])], [tf.ones([batch_size * num_steps], 
                    dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state
    
    if not is_training:
      self._cost = cost = (cost, softmax_out)
      return
    
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    
  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost
    
  @property
  def ret_outputs(self):
    return self._outputs
    
  @property
  def ret_logits(self):
    return self._logits

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def is_training(self):
    return self._is_training



class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 3 #10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 3 #10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 3 #10000


def get_config(model):
    if model == "small":
        return SmallConfig()
    elif model == "medium":
        return MediumConfig()
    elif model == "large":
        return LargeConfig()
    elif model == "test":
        return TestConfig()
    elif "test" in model:
        if model == "test1":
            return Test1Config()
        elif model == "test2":
            return Test2Config()
        elif model == "test3":
            return Test3Config()
        elif model == "test4":
            return Test4Config()
    elif model == "128x2":
        return onebytwoConfig()
    elif model == "128x3":
        return onebythreeConfig()
    elif model == "256x2":
        return twobytwoConfig()
    raise ValueError("Invalid model: %s", model)
  
def run_epoch(session, model, data, eval_op, verbose=False):
  """Runs the model on the given data. data[0] is features, data[1] is targets"""
  features = np.array(data[0])
  fshape = features.shape
  num_trajs = features.shape[0]
  data_len = features.shape[1]
  num_batches = num_trajs // model.batch_size
  num_runs = (data_len - 1) // model.num_steps
  num_epochs = num_batches * num_runs
  epoch_size = num_epochs
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
  #ITERATE OVER DATA
  tested_x = np.array([], dtype=float)
  tested_y = np.array([], dtype=int)
  p_dists = np.array([], dtype=float)
  for step, (x, y) in enumerate(data_iterator(data, model.batch_size,
                                                    model.num_steps)):
    fetches = [model.cost, model.final_state, eval_op]
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    cost, state, _ = session.run(fetches, feed_dict)
    if not model.is_training:
        (cost, softmax) = cost
        if p_dists.size == 0:
            p_dists = softmax
            tested_x = x
            tested_y = y
        else:
            p_dists = np.vstack((p_dists, softmax))
            tested_x = np.vstack((tested_x, x))
            tested_y = np.vstack((tested_y, y))
        
    costs += cost#
    iters += model.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))
  if model.is_training:
      return np.exp(costs/iters)
  tested_x = np.reshape(tested_x, (tested_x.shape[0] * tested_x.shape[1], tested_x.shape[2]))
  tested_y = np.reshape(tested_y, (tested_y.shape[0] * tested_y.shape[1], tested_y.shape[2]))
  return p_dists, tested_x, tested_y #np.exp(costs / iters)    

def get_acc_loss(p_dists, validation_set, print_b=True):
    if len(p_dists[0]) == 4:
        p_dists = p_dists[:,1:]
    acc = sutil.findAccuracy(p_dists, validation_set, True)
    score = sutil.findCrossEntropyScore(p_dists, validation_set, True, False)
    if print_b:
        print(np.array(p_dists).shape, np.array(validation_set).shape)
        print(p_dists[42], validation_set[42])
        print("Train... Acc:", acc, "loglikelihood:", score)
    return acc, score

def run_LSTM(train_data, test_data, model='test', save_path=None, numEpochs=None, train_only=False, valid_data=None):
    print("AAAGGGGHHHH:", Counter(train_data[1].flatten()))
    print("AAAGGGGHHHH:", Counter(test_data[1].flatten()))
    print(train_data[0].shape)
    print(train_data[1].shape)
    print(test_data[0].shape)
    print(test_data[1].shape)
    rand_perm = np.random.permutation(train_data[0].shape[0])
    train_data = train_data[0][rand_perm], train_data[1][rand_perm]
    if valid_data == None:
        tenth = int(train_data[0].shape[0] / 10)
        valid_data = (train_data[0][:tenth],train_data[1][:tenth])
        train_data = (train_data[0][tenth:],train_data[1][tenth:])
    print("HELLO\n",train_data[0].shape, train_data[1].shape)
    print(valid_data[0].shape, valid_data[1].shape)
    traj_len = train_data[0].shape[1]
    in_size = train_data[0].shape[2]
    out_size = train_data[1].shape[2]
    num_steps = traj_len - 1
    if "LSTM" in model:
        model = model[len("LSTM_"):]
    config = get_config(model)
    if numEpochs:
        config.max_max_epoch = numEpochs
    num_classes = len(set(train_data[1].flatten()))
    #the above causes nan perplexity
    num_classes = 3
    config.num_steps = num_steps
    config.vocab_size = num_classes 
    eval_config = get_config(model)
    eval_config.batch_size = 1
    eval_config.num_steps = num_steps #1
    eval_config.vocab_size = num_classes
    all_test_x = []
    all_test_y = []
    validation_set = valid_data[1]#[:,:-1,:] 
    validation_set = validation_set.flatten()
    #if max(set(validation_set)) > 2:
    #    validation_set = [i-1 for i in validation_set]

    with tf.Graph().as_default(), tf.Session(
                    config=tf.ConfigProto(inter_op_parallelism_threads=16,
                                          intra_op_parallelism_threads=16,
                                          use_per_session_threads=True)) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model(is_training=True, config=config, input_size=in_size, 
                      output_size = out_size)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = Model(is_training=False, config=eval_config, 
                          input_size=in_size, output_size = out_size)#eval_config

        print("HELLO\n",train_data[0].shape)
        print(valid_data[0].shape)
        tf.initialize_all_variables().run()
        start = time.clock()
        tolerance = 1e-15
        prev_best_loss = -1000
        min_num_epochs = 6
        #for i in range(2):
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** i#max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)
            rand_perm = np.random.permutation(train_data[0].shape[0])
            this_train_data = train_data[0][rand_perm], train_data[1][rand_perm]

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, this_train_data, m.train_op,
                                       verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mvalid = Model(is_training=False, config=eval_config, 
                          input_size=in_size, output_size = out_size)#eval_config
            
            valid_p_dists, _, __ = run_epoch(session, mvalid, valid_data, tf.no_op())
            acc, score = get_acc_loss(valid_p_dists, validation_set)
            if score < prev_best_loss + tolerance:# and i > min_num_epochs:
                break
            prev_best_loss = score
        end = time.clock()
        timeTrain = end - start
        print("Done training, time spent:", timeTrain)
        
        #saver = tf.train.Saver()
        #if save_path:
        #    saved_path = saver.save(session, save_path + model)
        #else:
        #    saved_path = saver.save(session, model)
            
        #print(model,"saved in:")
        #print(saved_path)
        #numVids = (train_data[0].shape)[1]
        if train_only:
            print("Not testing, returning.")
            return [], timeTrain, 0, [], []
        start = time.clock()
        p_dists, all_test_x, all_test_y = run_epoch(session, mtest, test_data, tf.no_op())
        #p_dists, all_test_x, all_test_y = runTestsPerVid2(test_data, session, mtest)
        
        end = time.clock()
        timePred = end - start
        print("Done testing, time spent:", timePred)
        print(p_dists.shape)
        #print("Test Perplexity: %.3f" % test_perplexity)
    return p_dists, timeTrain, timePred, all_test_x, all_test_y
    

def data_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: np array [[[feature1],[feature2],...[featureN]]],[target1,target2...targetN]
            ==> raw_data[0] = features, raw_data[1] = targets
            ==> each feature has length l
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data. x is of the shape [batch_size, num_steps, input_size]
    y are the nextMove targets, of the shape [batch_size, num_steps]

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  #new version -- features = num_fids x num_vids x num_features+1
  # actuals = num_fids x num_vids x 1
  # num_steps is the number of frames to give (so traj len??)
  # for model, input is of shape: [batch_size, num_steps, input_size]
  #                             aka (numVids, numFramesGiving, num_features+1)
  #             output is of [batch_size, num_steps, output_size]
  # Phase 2, passing in features of shape (numStuff, numFramesPerTrajectory, numFeatures)
  # default is (X, 50, 10)
  # goal is to pass as a batch, x of these trajectories of shape (1, 50, 10)
  # then, numSteps will take effect within the trajectory, if that makes sense...
  features = np.array(raw_data[0], dtype=np.float64)
  actuals = np.array(raw_data[1], dtype=np.int32)
  num_trajs = features.shape[0]
  data_len = features.shape[1] #this is traj_len
  input_size = features.shape[2]
  #a batch = batch_size trajectories, so features [i:i+batch_size,:,:]
  # given a batch of shape (batch_size, :, :) 
  # feed in [:, num_steps, :] at a time?
  # I should really just have num_steps be shape[1], but whatever
  if data_len < num_steps:
      print(data_len, num_steps)
      raise ValueError("num steps is greater than trajectory length")
  num_batches = num_trajs // batch_size
  num_runs = (data_len - 1) // num_steps
  for i in range(num_batches):
      batch_x = features[i*batch_size:(i+1)*batch_size, :, :]
      batch_y = actuals[i*batch_size:(i+1)*batch_size, :, :]
      for j in range(num_runs):
          x = batch_x[:, j*num_steps:(j+1)*num_steps, :]
          y = batch_y[:, j*num_steps:(j+1)*num_steps, :]
          yield (x, y)
  return
  if len(features.shape) == 2:  # means test
      input_size = features.shape[1]
      output_size = 1#len(actuals[0,:])
      data_len = actuals.shape[0]    
      batch_len = data_len // batch_size
      xdata = np.zeros([batch_size, batch_len, input_size], dtype=np.float64)
      ydata = np.zeros([batch_size, batch_len, output_size], dtype=np.int32)
      for i in range(batch_size):
          xdata[i] = features[batch_len * i:batch_len * (i + 1)]
          ydata[i] = actuals[batch_len * i:batch_len * (i + 1)]
  else:
      try:
          data_len = actuals.shape[0] * actuals.shape[1]
      except:
          print(actuals.shape)
      #actuals = np.reshape(actuals, (data_len, output_size))
      input_size = features.shape[2]
      output_size = 1#len(actuals[0,:])
      batch_len = data_len // batch_size
      xdata = np.zeros([batch_size, batch_len, input_size], dtype=np.float64)
      ydata = np.zeros([batch_size, batch_len, output_size], dtype=np.int32)
      print(xdata.shape)
      #xdata = np.swapaxes(features,0,1)#np.zeros([batch_size, batch_len, input_size], dtype=np.float64)
      #ydata = np.swapaxes(actuals,0,1) #np.zeros([batch_size, batch_len, output_size], dtype=np.int32)
      for i in range(batch_size):
        xdata[i] = features[batch_len * i:batch_len * (i + 1),:]
        ydata[i] = actuals[batch_len * i:batch_len * (i + 1),:]

  epoch_size = (batch_len - 1) // num_steps
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    if batch_size == 1:
        x = xdata[:, i*num_steps:(i+1)*num_steps]
        y = ydata[:, i*num_steps:(i+1)*num_steps]
    else:
        x = xdata[:, i*num_steps:(i+1)*num_steps,:]
        y = ydata[:, i*num_steps:(i+1)*num_steps,:]
    #y = [i-1 for i in y] # do not need because have 0's

    #x = data[:, i*num_steps:(i+1)*num_steps]
    #y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)
    

if __name__ == "__main__":
  tf.app.run()

class onebytwoConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35 #gets overwritten to traj len - 1
  hidden_size = 128
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 10
  vocab_size = 4 #10000
class onebythreeConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 3
  num_steps = 35 #gets overwritten to traj len - 1
  hidden_size = 128
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 10
  vocab_size = 4 #10000
class twobytwoConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35 #gets overwritten to traj len - 1
  hidden_size = 256
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 10
  vocab_size = 4 #10000

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 0.99
  max_grad_norm = 1
  num_layers = 1
  num_steps = 1
  hidden_size = 10
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 1
  vocab_size = 4 


class Test1Config(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 0.99
  max_grad_norm = 1
  num_layers = 1
  num_steps = 1
  hidden_size = 10
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 10   ##This is the diff
  vocab_size = 4
  
class Test2Config(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 0.99
  max_grad_norm = 1
  num_layers = 1
  num_steps = 10    ##This is the diff
  hidden_size = 10
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 10   ##This is the diff
  vocab_size = 4

class Test3Config(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 0.99
  max_grad_norm = 1
  num_layers = 2    ##This is the diff
  num_steps = 10    ##This is the diff
  hidden_size = 10
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 10   ##This is the diff
  vocab_size = 4

class Test4Config(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 0.99
  max_grad_norm = 1
  num_layers = 1
  num_steps = 10    ##This is the diff
  hidden_size = 128 ##This is the diff
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 10   ##This is the diff
  vocab_size = 4
