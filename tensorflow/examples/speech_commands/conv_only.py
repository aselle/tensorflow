# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import tensorflow as tf
import wave
import struct
import numpy as np
import sys
graphdef = tf.GraphDef()

from tensorflow.python.tools.optimize_for_inference import optimize_for_inference_lib

# Make new placeholders which will be inputs to the model
input = tf.placeholder(dtype=tf.float32, shape=(16000,1))
sample_rate = tf.placeholder(dtype=tf.int32, shape=())
graphdef.ParseFromString(open("conv_actions_frozen.pb",'rb').read())
# Load and remap unsupported ops (decode wav  mostly)
labels, = tf.import_graph_def(graphdef, {"decoded_sample_data": input, "decoded_sample_data:1": sample_rate}, return_elements=["labels_softmax:0"],name="")

sess = tf.Session()

# Wrap shape shape to be (1,)
class DummyTensor:
  def __init__(self, x):
    self.name = x.name
    self.dtype = x.dtype
  def get_shape(self):
    return (1,)

# optimize graph
def removeout(x): return x.split(":")[0]
curr = optimize_for_inference_lib.optimize_for_inference(
    sess.graph_def,[removeout(input.name), removeout(sample_rate.name)],
    [removeout(labels.name)],
    [tf.float32.as_datatype_enum, tf.int32.as_datatype_enum], True)
# Convert and write the model
sample_rate = DummyTensor(sample_rate)
data = tf.contrib.lite.toco_convert(curr, [input, sample_rate], [labels], allow_custom_ops=True)
open("conv.tflite","wb").write(data)
# make sure it runs
foo = tf.contrib.lite.Interpreter(model_path="conv.tflite")
foo.allocate_tensors()
print foo.get_tensor(foo.get_input_details()[0]["index"]).shape
foo.set_tensor(foo.get_input_details()[0]["index"], np.zeros((16000, 1), np.float32))
print(foo.get_tensor(foo.get_input_details()[1]["index"]).shape)
foo.set_tensor(foo.get_input_details()[1]["index"], np.array((44100,), np.int32))

foo.invoke()
