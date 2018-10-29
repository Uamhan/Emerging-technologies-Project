# a Python script that takes an image file
# containing a handwritten digit and identifies the digit using a supervised
# learning algorithm and the MNIST dataset.

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

(data_training, label_training), (data_test, label_test) = tf.keras.datasets.mnist.load_data()


image_index = 7777 # You may select anything up to 60,000
print(label_training[image_index]) # The label is 8
plt.imshow(data_training[image_index], cmap='Greys')