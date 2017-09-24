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
#### MNIST Example [source](https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/401_CNN.py)
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
return output, x # return x for visualization
```

## [Caffe2](https://caffe2.ai/)
#### Elevator Pitch:
> blah
#### MNIST Example [source](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/MNIST.ipynb)
```python
def AddLeNetModel(model, data):
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax
```

## [Theano](http://deeplearning.net/software/theano/)
#### Elevator Pitch:
> blah
#### MNIST Example [source]()

## [Keras](https://keras.io/)
#### Elevator Pitch:
> blah
#### MNIST Example [source]()
```python

```

## [Deeplearning4j](https://deeplearning4j.org/)
#### Elevator Pitch:
> blah
#### MNIST Example [source]()
```python

```

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
#### MNIST Example [source]()
```python

```

## [Microsoft Cognitive Toolkit](https://www.microsoft.com/en-us/cognitive-toolkit/)
#### Elevator Pitch:
> blah
#### MNIST Example [source]()
```python

```

## [Lasagne](https://lasagne.readthedocs.io/en/latest/)
#### Elevator Pitch:
> blah
#### MNIST Example [source]()

## [BigDL](https://bigdl-project.github.io/master/)
#### Elevator Pitch:
> blah
#### MNIST Example [source]()
```python

```

## [DSSTNE](https://github.com/amzn/amazon-dsstne)
#### Elevator Pitch:
> Blah
#### MNIST Example [source]()
```python

```