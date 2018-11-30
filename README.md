# Emerging-technologies-assignments
___

Repository containing assignment from fourth year software development module emerging technologies.

This assignment consists of 4 jupyter notebooks and a python script with the over arching topic of machinelearning 

the first note book explores the numpy random functionality.

the second notebook explores the iris data set and how it can be used with machine learning to identify the species of iris.

the third notebook explores the MNIST data set and its applications in machinelearning.

the fourth and final notebook discusses the choices made and performance of our digit recognition python script

The python script uses the MNIST dataset and keras to predict the values of handwritten digits.

___
## How to view a jupyter notebook.

for This project the anoconda distribution was used which contains jupyter notebook as standard. it is highly recomended that you install anoconda to use with this project. anoconda installation instructions can be found at https://www.anaconda.com/download/ once downloaded and installed navigate to this repositorys cloned folder in the comand promt window and enter the comand.
```
 jupyter notebook
```
this will open the juypter notebook interface in your web browser. from here you will see the contents of the cloned repository and you may open them by simply clicking on them.

for those who do not wish to install anoconda. you can use pythons package manager to install juypter notebook on its own. with the pip command
```
pip install jupyter
```
this will install juypter notebook you may then use the instructions above to open the juypter notebooks.
___
## How to run the python script.

To run the python script in this repository you will need to install the anonda distribution as its packages are extensivly used throughout the script. this package is availble at https://www.anaconda.com/download/

this script also uses Two other packages for the user interface image display and image manipulation.

### Pillow
```
pip install Pillow
```

### Cv2
```
pip install opencv
```

once you have anoconda and these two packages instaled navigate to the repository location in comand prompt and enter the comand
```
python digitrec.py
```

this will run the digitrec.py file. when the script is run for the first time it must generate the weights of the model whitch depeding on your machine may take some time. this process will only have to be run once. if you simply wish to test the script faster the epoch value on line 56 can be lowered to greately descrese the training time at the cost of accuracy.


