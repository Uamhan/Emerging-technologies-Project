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
from PIL import Image , ImageFilter
from PIL import ImageTk
from tkinter import filedialog
from tkinter import Tk,Label,Button,Frame
import cv2




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
        print("First Time training starting this may take a while. will only have to be run once")
        model.fit(data_training, label_training,epochs=10,validation_data=(data_test, label_test))
        #saves the model weights to a file to prevent need to refit on subsequent uses
        model.save('MNIST_MODEL.h5')

    
    #converts user inputed image
    npImageArray = np.array(userimage)
    #predicts the digit from user image using the model
    prediction = model.predict(npImageArray.reshape(1,28,28,1))
    #returns predicted value
    return(prediction.argmax())
    
def prep_image(image):
    #converts image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #resizes image to 28 by 28 pixels
    gray = cv2.resize(255-gray, (28, 28))
    #flattens the image ie reduces possible colour values from 0-255 to 0-1
    flatten = gray.flatten() / 255
    #returns modifyed image
    return flatten

def select_image():
    #gets the gui pannel for the image
    global panelA
    #gets the path of the user image through windows file explorer
    path = filedialog.askopenfilename()
    #if the path isint empty
    if len(path) > 0:
        #read in image from path
        image = cv2.imread(path)
        #convert the colour of the image to a format that can be displayed by pythons pillow library
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #preps the user image for predicition
        prepImage = prep_image(image)
        #predicts the digit in the user image
        prediction = predict_image(prepImage)
        #tempdisplay for pridiction
        print(prediction)
        #formats the image for display in the gui
        image = Image.fromarray(image)
        #reformats the image to a photoimage
        image = ImageTk.PhotoImage(image)
        #if panelA dosent exist initilise it
        if panelA is None:
            #panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
        # else update the image panel with new image
        else:
            # update the pannels
            panelA.configure(image=image)
            panelA.image = image

#initilise gui
root = Tk()
#image pannel
panelA = None
#button to call select image command	
btn = Button(root, text="Select an image", command=select_image)
#button formating
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
 
#initilise main gui loop
root.mainloop()






    

