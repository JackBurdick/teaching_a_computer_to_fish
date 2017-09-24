# Overview

## [Tensorflow](https://www.tensorflow.org/)
#### Elevator Pitch:
> blah

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
#### MNIST Example
[source](https://mxnet.incubator.apache.org/tutorials/python/mnist.html)
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