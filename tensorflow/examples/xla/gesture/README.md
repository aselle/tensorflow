# Gesture recognition with XLA AOT compilation

## Gesture recognition with XLA AOT compilation

```
git clone https://github.com/aselle/tensorflow
git checkout gesture
```

## Why?

* TensorFlow libraries are large.
* Size is key on mobile devices
* XLA can replace standard nodes with compiled code

## What we will do

* Define a simple model
* Compile that model to a standalone library
* Test it
* Talk about using it on android 

## Prerequisites

Currently you need to build TensorFlow from source to use XLA. XLA must
be enabled in ./configure:
```
Do you wish to build TensorFlow with the XLA
just-in-time compiler (experimental)? [y/N] y
```
After this you can build a whl as usual
```sh
$ bazel build -c opt //tensorflow/tools/pip_package:build_pip_package 
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```
For this project, I am going to use bazel heavily, because it is much easier for XLA AOT compilation. 

## Basic steps

*Make a model*

1. Write the model and add build rule (`gesture.py`, `BUILD`)
2. Train the model
3. Dump checkpoint and graph (`gesture.ckpt`, `gesture.pb`)
4. Freeze the model (`gesture_frozen.pb`)

*Build compact AOTC code*

1. Define proto `gesture.config.pbtxt`
2. Define `tf_library` build rule in `BUILD`
3. Define `gesture_simple.cc` and `BUILD` rule
4. Build and run `gesture_simple`



## Making the model

The main idea is that we have a sequence of mouse or touch drags that form a path. We want to recognize if the path is a circle, line or something else. For simplicity, we train on synthetic data (noisy circles, lines and random paths). They look like this

![Images of example paths](examples.png)


### Write a model

The details of the model are specified in `gesture.py`. Each model is given a class number 0 is circle, 1 is line, and 2 is other. The model is a 1D convolutional Neural network, which is probably crazy overkill for this application.

### Preparing your model code for XLA

There are two things that you need to be careful of in building your model. First, you must explicitly name input placeholders and the inference op.
That is because later we'll need to refer to them by name in the xla compile rules to make a AOT compilation.
```python
    self.features = tf.placeholder(shape=[BATCH_SIZE,N,2],dtype=tf.float32,name="features")
    self.labels = tf.placeholder(shape=[BATCH_SIZE,N_CLASSES],dtype=tf.float32,name="labels")
```
To save the model checkpoint and graph you follow a pattern like this:
```python
    # Initialize your model variables
    ...
    # Train your model
    ...
    # Save your model
    saver = tf.train.Saver()
    saver.save(sess, "gesture.ckpt")
    tf.train.write_graph(sess.graph_def, os.getcwd(), "gesture.pb", False)
```

#### Full code

```python
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
  max_rows_to_display = 4
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
    viz(data_list, "examples.png")
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
     
```

### Training the model

Now we are going to train the model:
```
# train the model... make sure training accuracy is above 90%
cd tensorflow/examples/xla/gesture
bazel  build -c opt gesture
../../../../bazel-bin/tensorflow/examples/xla/gesture/gesture train
# make sure checkpoint works
../../../../bazel-bin/tensorflow/examples/xla/gesture/gesture eval
```

### Validating the model

Make sure you see a relatively good training accuracy i.e.
```
training accuracy: 0.961589
eval. accuracy:    0.960069
```
If not, just run it again. This is not a super robust model, it is intended for illustration.

### Freezing the model

Now, the only thing left to do to have a fully baked model that we can use is to freeze the checkpoint. This bakes all variable values from the checkpoint into constants that are used to replace the variables in a new graph. This graph is written to `gesture_frozen.pb`. This can be done with the
`freeze_graph.py` command:
```sh
bazel build -c opt //tensorflow/python/tools:freeze_graph
../../../../bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_checkpoint `pwd`/gesture.ckpt \
  --input_graph gesture.pb  \
  --output_node_names out_class --input_binary \
  --output_graph gesture_frozen.pb
```

(Don't forget --input_binary to indicate protos are binary)

## Build compact AOTC code

To compile a library for XLA AOT, you will need to define the inputs and outputs to the model. In this case we know there is only one input "features" which is shape 
`(BATCH_SIZE, N, 2)` or `(256, 32, 2)`. In addition, we know it is type float. In addition we know the output is "out_class" and type integer with shape `(256, 1)`. 

### Define inputs and outputs

In `gesture.config.pbtxt` we do
```
# Input has 3 dimensions
feed {
  id { node_name: "features" }
  shape {
    dim { size: 256 }
    dim { size: 32 }
    dim { size: 2 }
    
  }
}
# Fetch the output class
fetch {
  id { node_name: "out_class" }
}
```

### Define tf_library build rule

```
# Use the tf_library macro to compile your graph into executable code.
tf_library(
    name = "gesturelib",
    cpp_class = "Examples::Gesture",
    graph = "gesture_frozen.pb",
    config = "gesture.config.pbtxt",
)
```

### Define gesture_simple.cc and BUILD rule

#### Includes

We now add boilder plate includes

```
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL
#include "tensorflow/examples/xla/gesture/gesturelib.h"
```


The generated header has a class named `Examples::Gesture`.

#### Thread pool 

Now we setup the thread pool and instantiate the AOT compiled library
```cpp
int main(int argc,char* argv[]){
  Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());
  Examples::Gesture gesture;
  gesture.set_thread_pool(&device);
  ...
```

#### Instantiate the model

Now we make the Gesture model instance and bind the thread pool
```cpp
int main(int argc,char* argv[]){
	...
  Examples::Gesture gesture;
  gesture.set_thread_pool(&device);
  ...
```

#### Building the data set

```cpp
int main(int argc,char* argv[]){
	...
  float* vals = gesture.arg0_data();
  std::fill(vals,vals+BATCH_SIZE*N*2,0.f);
  float theta=0.f;
  int idx=0;
  for(int i=0;i<N;i++){
    vals[idx++] = cos(theta);
    vals[idx++] = sin(theta);
    theta += i/(2*M_PI);
  }
  ...
```
 
#### Running the model

```cpp
int main(int argc,char* argv[]){
	...
  gesture.Run();
  tensorflow::int64* data=gesture.result0_data();
  std::cout<<"test "<<0<<" is "<<data[0]<<std::endl;
```

#### BUILD rule

Now we finally setup the build rule.
```
# The executable code generated by tf_library can then be linked into your code.
cc_binary(
    name = "gesture_simple",
    srcs = [
      "gesture_simple.cc",  # include test_graph_tfmatmul.h to access the generated header
    ],
    deps = [
      ":gesturelib",  # link in the generated object file
      "//third_party/eigen3",
    ],
    includes = [
      "gesture.h"
    ] ,
    linkopts = [     "-lpthread"]
)
```

### Build and run `gesture_simple`

Now we can build our binary
```sh
bazel build -c opt //tensorflow/examples/xla/gesture:gesture_simple
```
Once that is done, run the binary
```
../../../../bazel-bin/tensorflow/examples/xla/gesture/gesture_simple
```
which should produce
```
test 0 is 0
test 1 is 2
test 2 is 1
```
which means (circle, misc, other) which is as is expected

## What about Android?

Instead of `main()` make a straight-C ABI function:
```cpp
extern "C"{
	int classifyGesture(float* x, float* y){
	  // Initialize instance and thread pool
	  ...
	  // Invoke the functionality
	  int idx=0;
	  float* features = arg0_data();
	  for(int i=0;i<32;i++){
		  features[idx++] = x[i];
		  features[idx++] = y[i];
		  idx++;
		}
	  gesture.Run();
	  // Extract the result
	  tensorflow::int64 my_class=gesture.result0_data()[0];
	  return my_class;
	}
}
```
This is necessary because JNI can only call C functions.

