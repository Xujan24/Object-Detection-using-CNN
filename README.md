# Simple Object Detection using Convolutional Neural Network
Object detection is one of the fundamental problem in computer vision. Given an image, the goal is to detect the objects within
the image, by generating a rectangular box (bounding box) around the objects. Obviously, there can be multiple objects in an 
image of same or different classes. Object detection deals with identifying each of these objects. However, in this project 
we are just concerned with a single object detection.

## Model Architecture
Our model consists of three convolutional layers and two fully connected layers. A kernel of size 5 with stride 1 is used in 
each of the convolutional layers and rectified linear units, ReLU, is used as activation function. A max pooling layer of filter 
size 2 with stride 2 is employed after each of the first two convolutional layers. 

## Training
We have trained the network for 50 epoch using stochastic gradient descent (SGD). For the first 30 epoch, the learning rate is 
set to 0.000001 and after that it is reduced to 0.0000001. We have also employed a momentum of 0.9. For regularization, dropout
(with p=0.5) is used.

## Dataset
The dataset we have used here is very simple and is generated in python. Each image is 100x100x1 and has a small rectangular 
box of random size and shape and at random positions. For background the color value is set to 0 and for box it is set to 255. 
The training dataset has 1000 of such images and the test dataset consists of 200 images. The corresponding ground truth 
information are stored in separate file.

To use the pretrained model, download the trained model from [this](https://1drv.ms/u/s!AoRMuqHDZuP8iyBdb4DMllcpqhil) link.
