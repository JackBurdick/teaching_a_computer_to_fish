# Overview

## [Tensorflow](https://www.tensorflow.org/)
#### Elevator Pitch:
> blah
#### MNIST Example [source](https://www.tensorflow.org/tutorials/layers)
```python
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5],
                           padding="same", activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5],
                           padding="same", activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
logits = tf.layers.dense(inputs=dropout, units=10)
```

## [PyTorch](http://pytorch.org/)
#### Elevator Pitch:
> blah

## [Caffe2](https://caffe2.ai/)
#### Elevator Pitch:
> blah

## [Theano](http://deeplearning.net/software/theano/)
#### Elevator Pitch:
> blah

## [Keras](https://keras.io/)
#### Elevator Pitch:
> blah

## [Deeplearning4j](https://deeplearning4j.org/)
#### Elevator Pitch:
> blah

## [MxNet](http://mxnet.io/)
#### Elevator Pitch:
> blah
#### MNIST Example [source](https://mxnet.incubator.apache.org/tutorials/python/mnist.html)
```python
data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

```

## [PaddlePaddle](http://www.paddlepaddle.org/)
#### Elevator Pitch:
> blah

## [Microsoft Cognitive Toolkit](https://www.microsoft.com/en-us/cognitive-toolkit/)
#### Elevator Pitch:
> blah

## [Lasagne](https://lasagne.readthedocs.io/en/latest/)
#### Elevator Pitch:
> blah

## [BigDL](https://bigdl-project.github.io/master/)
#### Elevator Pitch:
> blah

## [DSSTNE](https://github.com/amzn/amazon-dsstne)
#### Elevator Pitch:
> Blah