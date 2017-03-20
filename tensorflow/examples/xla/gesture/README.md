# Gesture recognition with XLA AOT compilation

### Model parameters

The main idea is that we have a sequence of mouse or touch drags that form a path. We want to recognize if the path is a circle, line or something else. For simplicity, we train on synthetic data (noisy circles, lines and random paths). They look like this

![Images of example paths](examples.png)

The details of the model are specified in `gesture.py`. Each model is given a class number 0 is circle, 1 is line, and 2 is other. 

### Saving the model and checkpoints

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

### Training the model

### Freezing the model

Now, the only thing left to do to have a fully baked model that we can use is to freeze the checkpoint. This bakes all variable values from the checkpoint into constants that are used to replace the variables in a new graph. This graph is written to `gesture_frozen.pb`. This can be done with the
`freeze_graph.py` command:

```sh
python ../../../python/tools/freeze_graph.py \
  --input_checkpoint `pwd`/gesture.ckpt \
  --input_graph gesture.pb  \
  --output_node_names out_class --input_binary \
  --output_graph gesture_frozen.pb
```

### Training and baking our model

Now we are going to train the model:
```
cd tensorflow/examples/xla/gesture
# train the model... make sure training accuracy is above 90%
python gesture.py train
# make sure checkpoint works
python gesture.py eval         
```
Now we need to combine the checkpoint and graph
```
tensorflow/python/tools/freeze_graph.py \
  --input_checkpoint /tmp/model/aotc.ckpt \
  --input_graph /tmp/model/aotc.pbtxt  \
  --output_node_names y --input_binary \
  --output_graph /tmp/model/aotc_frozen.pb
```
(Don't forget --input_binary to indicate protos are binary)

### Building an XLA binary

Currently you need to build TensorFlow from source to use XLA. XLA must
be enabled in ./configure:
```

```

```sh
bazel build --config=opt //tensorflow/examples/xla/gesture:gesture
```