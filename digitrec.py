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




def predict_image(userimage):
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
        #sequential model for mnist predictions
        model = Sequential()
        #adds a convolutional layer with 32 nodes relu activation
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        #adds a convolutional layer with 64 nodes relu activation
        model.add(Conv2D(64, (3, 3), activation='relu'))
        #maxpooling with a size of 2,2
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #drop out layer with 0.25 ratio
        model.add(Dropout(0.25))
        #flattens the current layer
        model.add(Flatten())
        #ads a dense layer with 128 nodes with relu activation
        model.add(Dense(128, activation='relu'))
        #droup out layer with a ratio of 0.5
        model.add(Dropout(0.5))
        #final dense layer that will produce one of 10 results
        model.add(Dense(10,activation=tf.nn.softmax))
        #compiles the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        #fits the model to the training data
        model.fit(data_training, label_training,epochs=10,validation_data=(data_test, label_test))
        #saves the model weights to a file to prevent need to refit on subsequent uses
        model.save('MNIST_MODEL.h5')

    
    #converts user inputed image
    npImageArray = np.array(userimage)
    #predicts the digit from user image using the model
    prediction = model.predict(npImageArray.reshape(1,28,28,1))
    #returns predicted value
    return(prediction.argmax())
    







    

