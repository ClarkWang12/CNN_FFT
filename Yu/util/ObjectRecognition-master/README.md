# ORBID
## Object Reognition Based Image Description
Using on pre-trained (on imagenet data) Convolutional Neural Network for object recognition

There are 2 parts to this project. Kindly follow these steps to run the code. 

### PART 1
This part does a comparItive study on the different classifiers and gives the testing accuracy. This is only for a single object detection.

Code file - object_recognition.m

### PART 2
This part is the end to end demo of the ORBID system. Given an image, it goes through the following steps -

1. Identify the various prominent objects in the given image
2. Recognize the objects
3. Identify the position of the objects relative to each other (eg., Object A is to the left of ObjectB)
4. Build the caption using the object names and position

Code file - load_train.m, ORBID.m

Input image - Input4image.jpg
