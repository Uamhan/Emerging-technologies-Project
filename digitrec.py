# a Python script that takes an image file
# containing a handwritten digit and identifies the digit using a supervised
# learning algorithm and the MNIST dataset.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)