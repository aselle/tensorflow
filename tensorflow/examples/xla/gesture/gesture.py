"Gesture model for TensorFlow"
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from __future__ import division

import argparse
import math
import os
import sys
import random
import matplotlib.pylab as pylab
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import tf_logging as logging
# Number of data points
N=32
# Number of examples to generate
N_examples=2048
# Size of training batch
BATCH_SIZE=256
# Number of classes
N_CLASSES=3
# Number of training epochs throughout the data
EPOCHS=2048
# Names of classes
CLASS_TO_NAME = {0: "circle", 1: "line", 2: "other"}

def build_set():
  """Build a synthetic gesture database"""
  np.random.seed(33244)
  data_list = []
  # Circles
  for i in range(N_examples):
    r=np.random.rand()*100+.5
    theta_offset = np.random.rand(N)*2*math.pi / N
    theta_start = np.random.rand()*2*math.pi
    thetas = np.linspace(0, 2*math.pi,N) + theta_offset + theta_start
    radius_offset = 0.2*(np.random.rand(N)-0.5)*r
    radius_vector = r + radius_offset
    foo = np.transpose(np.stack([radius_vector*np.cos(thetas),radius_vector*np.sin(thetas)],axis=0))
    data_list.append((foo,np.array([1.,0.,0.]),0))
    #labels_list.append(1)
  # Lines
  for i in range(N_examples):
    r=np.random.rand()*100+.5
    theta_offset = np.random.rand(N)*2*math.pi / N
    delta = np.random.random((2,1))-.5
    t=np.linspace(0.,1.,N)*r
    line = np.stack([t,t])*(delta*np.ones((2,N)))
    perturb = (np.random.random((2,N))-.5)*.02*r
    line+=perturb
    foo = np.transpose(line)
    data_list.append((foo,np.array([0.,1.,0.]),1))
  # Random paths
  for i in range(N_examples):
    angle_offsets = 1.*(np.random.rand(N)-0.5)
    angles = np.cumsum(angle_offsets)
    deltas = np.transpose(np.stack([np.cos(angles),np.sin(angles)]))

    fin = np.cumsum(deltas, axis=0)
    data_list.append((fin,np.array([0.,0.,1.]),2))
  random.shuffle(data_list)
  return data_list

def viz(data, filename):
  """Build a set of data and vizualize it."""
  columns = 4
  max_rows_to_display = 8
  rows = min(max_rows_to_display,len(data)/columns)
  idx = 0
  pylab.figure(figsize=(20,10))
  #pylab.subplots(rows,columns)
  for i in range(rows):
    for j in range(columns):
      if idx >= len(data): break
      pylab.subplot(rows, columns, idx)
      bar = data[idx][0][:]
      xs = [x[0] for x in bar]
      ys = [x[1] for x in bar]
      pylab.scatter(xs,ys)
      pylab.title(CLASS_TO_NAME[data[idx][2]])
      idx+=1
  pylab.savefig(filename)


class GestureModel:
  def __init__(self, data_list):
    N_training=3*N_examples//4
    self.training_set = data_list[:N_training]
    self.eval_set = data_list[N_training:]

  def buildModel(self):
    self.features = tf.placeholder(shape=[BATCH_SIZE,N,2],dtype=tf.float32,name="features")
    self.labels = tf.placeholder(shape=[BATCH_SIZE,N_CLASSES],dtype=tf.float32,name="labels")
    center = tf.reduce_sum(self.features, axis=1) / N
    max = tf.reduce_max(self.features, axis=1) 
    min = tf.reduce_min(self.features, axis=1) 
    fixed = (self.features - tf.reshape(center,(BATCH_SIZE,1,2))) / tf.reshape(max-min,(BATCH_SIZE,1,2))
    curr = fixed
    curr = tf.nn.relu(tf.layers.conv1d(curr, 8, 1))
    curr = tf.layers.average_pooling1d(curr,2,2)
    curr = tf.nn.relu(tf.layers.conv1d(curr, 8, 1))
    curr = tf.layers.average_pooling1d(curr,2,2)
    curr = tf.contrib.layers.fully_connected(curr,3)
    curr = tf.nn.relu(curr)
    curr = tf.layers.average_pooling1d(curr,5,5)
    out = curr
    #self.out_class = tf.nn.top_k(out)[1]
    print (out.get_shape())
    self.out_class = tf.argmax(out[:,0,:], 1)
    out_class_named = tf.identity(self.out_class, name="out_class")
    self.loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=out))
    self.global_step = tf.get_variable("global_step",shape=(),initializer=tf.zeros_initializer())
    self.lr = tf.train.exponential_decay(1., self.global_step, 10000, 0.1)
    self.update_step = tf.assign_add(self.global_step, 1)

    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train = optimizer.minimize(self.loss)

  def trainModel(self, sess):
    """Train and validate a gesture model, also save it."""
    # Setup the model checkpoint
    saver = tf.train.Saver()
    loss_val = 0.
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
      random.shuffle(self.training_set)
      if epoch % 10==0:
        print("epoch %d/%d %f"%(epoch,EPOCHS,loss_val))
      idx = 0
      while 1:
        datas = self.training_set[idx:idx+BATCH_SIZE]
        if len(datas) < BATCH_SIZE:
         break
        #print datas[0][0].shape
        #print [x[0].shape for x in datas]
        feature = np.stack([x[0] for x in datas])
        label = np.stack([x[1] for x in datas])
        #print "feat,label ",feature.shape,label.shape
        #break
        train_val,loss_val, update_val, lr_val, global_step_val=sess.run(
          [self.train, self.loss, self.update_step, self.lr, self.global_step], 
          feed_dict={self.features: feature, self.labels:label})
        #if global_step_val % 128 == 0:
        #  print "step",global_step_val,"loss",loss_val,"lr",lr_val
        idx+=BATCH_SIZE
    #sys.exit(0)

    # Write checkpoint and the graph
    saver.save(sess, "gesture.ckpt")
    tf.train.write_graph(sess.graph_def, os.getcwd(), "gesture.pb", False)

  def loadCheckpoint(self, sess):
    saver = tf.train.Saver()
    saver.restore(sess, os.getcwd()+"/gesture.ckpt")

  def eval(self, sess, data_set):
    idx = 0
    right = 0
    count=0

    while 1:
      datas = data_set[idx:idx+BATCH_SIZE]
      if len(datas) < BATCH_SIZE:
         break
        
      feature = np.stack([x[0] for x in datas])
      label = np.stack([x[1] for x in datas])
      classes = np.stack([x[2] for x in datas])

      if len(datas) < BATCH_SIZE:
         break
      out_class_val, = sess.run(
        [self.out_class],
        feed_dict={self.features: feature})
      #print classes.shape, out_class_val.shape
      out_class_val = np.reshape(out_class_val,(BATCH_SIZE,))
      #print classes
      #print out_class_val
      num_right = np.count_nonzero(classes== out_class_val)
      num_in_batch = out_class_val.shape[0]
      #print num_right, num_in_batch
      right += num_right
      count += num_in_batch
      #print vals
      idx+=BATCH_SIZE
    tf.train.write_graph(sess.graph_def, os.getcwd(), "gesture.pb", False)
    return  (float(right) / count)



def main(unused_argv):
  parser = argparse.ArgumentParser(description="Train a naive gesture recognition model")
  parser.add_argument("cmd", type=str, help="the command, can be 'train' or 'eval'")
  args = parser.parse_args()

  data_list = build_set()
  gesture = GestureModel(data_list)
  with tf.Session() as sess:
    #viz(data_list, "examples.png")
    gesture.buildModel()
    if args.cmd == "train":
      gesture.trainModel(sess)
    elif args.cmd == "eval":      
      gesture.loadCheckpoint(sess)
    else:
      raise RuntimeError("Invalid cmd, must be train or eval")
    print("training accuracy: %f" % gesture.eval(sess, gesture.training_set))
    print("eval. accuracy:    %f" % gesture.eval(sess, gesture.eval_set))

if __name__=="__main__":
  app.run()
      
