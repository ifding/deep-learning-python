
# Deep Learning using Python/C++/OpenCV

--- 

## Basics

- [Deep Learning Basics](./resources/deep-learning.md)
- [Components of Autonomous Driving System](./resources/autonomous-driving.md)
- [Datasets](./resources/datasets.md)
- [Train your own object detector with Faster-RCNN & PyTorch](./faster-rcnn-tutorial)


## Computer Vision and Deep Learning

#### [P1 - Detecting Lane Lines](./01_finding_lane_lines)
 - **Basic:** Detected highway lane lines on a video stream. Used OpencV image analysis techniques to identify lines, including Hough Transforms and Canny edge detection.
 - **Keywords:** Computer Vision, OpenCV
 
#### [P2 - Traffic Sign Classification](./02_traffic_sign_detector)
 - **Summary:** Built and trained a support vector machines (SVM) to classify traffic signs, using [dlib](http://dlib.net/). Google Street View images can be used to train the detectors. 25~40 images are sufficient to train a good detector.
 - **Keywords:** Computer Vision, Machine Learning
 
#### [P3 - Object Detection with OpenCV](./03_opencv_detection)
 - **Summary:** The provided API (for C++ and Python) is very easy to use, just load the network and run it. Multiple inputs/outputs are supported. Here are the examples: https://github.com/opencv/opencv/tree/master/samples/dnn.

#### [P4 - Vehicle Detection and Tracking](./04_vehicle_detection)
 - **Summary:** Created a vehicle detection and tracking pipeline with OpenCV, histogram of oriented gradients (HOG), and support vector machines (SVM). Implemented the same pipeline using a deep network to perform detection. Optimized and evaluated the model on video data from a automotive camera taken during highway driving.
 - **Keywords:** Computer Vision, Deep Learning, OpenCV

#### [P5 - Road Segmentation](./05_road_segmentation)
- **Summary:** Implement the road segmentation using a fully-convolutional network.
- **Keywords:** Deep Learning, Semantic Segmentation


## References

- <https://github.com/spmallick/learnopencv>
- <https://github.com/ndrplz/self-driving-car>

