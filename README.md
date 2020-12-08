# deep_learning_crash_guard
Transfer learning on AlexNet for crash avoidance in an autonomous RC Car

## Introduction

I train a Convolution Neural Network (CNN) using transfer learning to create an AI system to help an autonomous RC Car avoid crashes.

Video demo below:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/8gU-APmyHkg/0.jpg)](https://www.youtube.com/watch?v=8gU-APmyHkg)

The repo contains the following files

##  DataCollect.ipynb

This is an app to display video from the onboard csi camera and lets you save images into blocked and non blocked folders, which will be used to train the CNN.

##  Avoidance.ipynb

Script uses PyTorch framework to train using transfer learning on an AlexNet arhcitecture (and initial parameter values) to learn images that indicate a potential crash. The images are processed from the camera (see DataCollect) and pre-processed to make them ready for training. 


##  crashGuard.py

An infinite loop reading USB joystick commands with a background thread running the camera feed through the model. Background thread implements interrupt-like override on detection of a potential crash.

