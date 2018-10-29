# a Python script that takes an image file
# containing a handwritten digit and identifies the digit using a supervised
# learning algorithm and the MNIST dataset.

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
import keras.models

#Importing the training and test data from the tensoflow keras library.
(data_training, label_training), (data_test, label_test) = tf.keras.datasets.mnist.load_data()

# changing array to 4-dimesions to fit Keras API
data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)
data_training = data_training.reshape(data_training.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

#trys to load the a pre compiled model if not found in the current directory creates a new model and fits the dataset to it.
try:
    model = keras.models.load_model('MNIST_MODEL.h5')
except:
    #creating our sequential model. for our CNN(Convolutional neural network).
    #represents a linear stack of layers.
    model = Sequential()
    #adds a 2d convolutional layer to the model. with filter size of 28 and a kernal size of (5,5)
    model.add(Conv2D(28, kernel_size=(5,5), input_shape=input_shape))
    #adds a pooling lauer with a pool size of (3,3)
    model.add(MaxPooling2D(pool_size=(3, 3)))
    # Flattens arrays to fully connected layers
    model.add(Flatten()) 
    #adds standard densley conected NN layer
    model.add(Dense(128, activation=tf.nn.relu))
    #add drop out rate of 0.2 to prevent overfitting.
    model.add(Dropout(0.4))
    #final layer of the model with an out put of 10 to represent the 10 possible classifications 0-9
    model.add(Dense(10,activation=tf.nn.softmax))
    #compiles our model for use with the optimizer adam. The sparse categorical crossentropy loss algorythm.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=data_training,y=label_training, epochs=10)
    #saves the model so that we only need to compile once.
    model.save('MNIST_MODEL.h5')

#randomly picks a number from 0 to 9999 for the imgindex
imgIndex = np.random.randint(10000)
#plots the test data from at index imgindex
plt.imshow(data_test[imgIndex].reshape(28,28),cmap='Greys')
#use the model created previously to predict what the number the img represents is
prediction = model.predict(data_test[imgIndex].reshape(1, 28, 28,1))
#priints the predicted value.
print("predicted value = {}".format(prediction.argmax()))
#shows the plot of the image.
plt.show()

