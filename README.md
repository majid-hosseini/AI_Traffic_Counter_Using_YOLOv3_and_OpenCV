# AI traffic Counter Using YOLOv3 and OpenCV
An AI traffic counter is implemented in this project to detect and track vehicles on a video stream and count those going through a defined line on each direction of a highway. It utilizes the following two algorithms:

- YOLO algorithms to detect objects on each of the video frames. 
- SORT algorithms to track those objects over different frames.

## How it works
This AI traffic counter is composed of three main components: a detector, tracker and counter. A detector capable of processing a Realtime video to identify vehicles in a given frame of video and returns a list of bounding boxes around the objects was explained in my previous project titled [Realtime Object Detector](https://github.com/majid-hosseini/Realtime-Object-Detector). The tracker uses the bounding boxes to track the vehicles in subsequent frames. The detector is also used to update the trackers periodically to ensure that they are still tracking the vehicles correctly. Once the objects are detected and tracked over different frames of a traffic video stream, a mathematical calculation is applied to count the number of vehicles that their previous and current frame positions intersect with a defined line in the frames.


## YOLO Algorithm
In recent years, deep learning algorithms are offering cutting-edge improved results for object detection. YOLO algorithm is one of the most popular Convolutional Neural Networks with a single end-to-end model that can perform object detection in real-time. YOLO stands for, You Only Look Once and is an algorithm developed by Joseph Redmon, et al. and first described in the 2015 paper titled [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

The creation of the algorithm stemmed from the idea to place down a grid on the image and apply the image classification and localization algorithm to each of the grids.
Here the YOLOv3, a refined design which uses predefined anchor boxes to improve bounding box, is utilized for object detection in new images. Source code and pre-trained models of YOLOv3 is available in the [official DarkNet website](https://pjreddie.com/darknet/yolo/).

## SORT Algorithms
Simple Online and Realtime Tracking (SORT) is an implementation of tracking-by-detection framework, in which objects are detected in each frame and information of past and current frames are used to produce object identities on the fly. It is designed for online and real-time tracking applications. SORT was initially described in [this paper](http://arxiv.org/abs/1602.00763) by Alex Bewley et al.


## OpenCV
OpenCV is an open source library which provides tools to perform image and video processing for the computer vision, machine learning, and image processing applications. It is particularly popular for real-time operations which is very important in today’s systems. Integration with various libraries, such as Numpuy and python resulted in great capablities of processing the OpenCV array structures for analysis and mathematical operations.

## Project background and objectives

This project is an implementation of YOLOv3 and SORT algorithms to count vehicles crossing defined lined on both directions of a highway. Following are the steps to count vehicles in a real-time traffic video of highway:

* download the YOLOv3 pre-trained model weights
* define a OpenCV DNN module and load the downloaded model weights
* create a cv2.VideoCapture object to load video files and read it frame-by-frame 
* perform object detection process on each frame of the video file (similar to what was described in my previous project: [Realtime Object Detector](https://github.com/majid-hosseini/Realtime-Object-Detector).
* run the tracker which uses the bounding boxes to track the vehicles in subsequent frames
* The detector is also used to update the trackers periodically to ensure that they are still tracking the vehicles correctly
* a mathematical calculation is applied to count the number of vehicles passing two defined line on both directions of highway.
* processed frames are displayed using cv2.imshow method
* save the video of processed frames indicating detected objects and the counter
*close video files and destroy the window, which was created by the imshow method using cap.release() and cv2.destroyAllWindows() commands. 

## Sample ouput
Following is a sample output of this project showing detected vehicles and two counter indicating the number of vehicles on both directions of a highway traffic video.

![Sample-output](https://github.com/majid-hosseini/AI_traffic_counter_using_YOLOv3_and_Keras/blob/main/input/Output_5.gif)

 


# Requirement
* OpenCV 4.2.0
* Python 3.6


# Quick start
* **Download official** <a href="https://pjreddie.com/media/files/yolov3.weights" target="_blank">**YOLOv3 Pre-trained Model Weights**</a> **and place it in the same folder of project**



## Dependencies
* OpenCV
* Numpy


How to use?
===========
The project is developed in Jupyter Notebooks which is automatically rendered by GitHub. The developed codes and functions for each step are explained in the notebook.





