
# Deep Learning for Autonomous Vehicles

--- 

## Basics

- [Deep Learning Basics](./resources/deep-learning.md)
- [Components of Autonomous Driving System](./resources/autonomous-driving.md)
- [Datasets](./resources/datasets.md)
- [ROS](./ROS/datasets.md)


## Computer Vision and Deep Learning

#### [P1 - Detecting Lane Lines](./01_finding_lane_lines)
 - **Basic:** Detected highway lane lines on a video stream. Used OpencV image analysis techniques to identify lines, including Hough Transforms and Canny edge detection.
 - **Keywords:** Computer Vision, OpenCV
 
#### [P2 - Traffic Sign Classification](./02_traffic_sign_detector)
 - **Summary:** Built and trained a support vector machines (SVM) to classify traffic signs, using [dlib](http://dlib.net/). Google Street View images can be used to train the detectors. 25~40 images are sufficient to train a good detector.
 - **Keywords:** Computer Vision, Machine Learning
 
#### [P3 - Behavioral Cloning](./03_behavioral_cloning)
 - **Summary:** Built and trained a convolutional neural network for end-to-end driving in a simulator, using TensorFlow and Keras. Used optimization techniques such as regularization and dropout to generalize the network for driving on multiple tracks.
 - **Keywords:** Deep Learning, Convolutional Neural Networks

#### [P4 - Vehicle Detection and Tracking](./04_vehicle_detection)
 - **Summary:** Created a vehicle detection and tracking pipeline with OpenCV, histogram of oriented gradients (HOG), and support vector machines (SVM). Implemented the same pipeline using a deep network to perform detection. Optimized and evaluated the model on video data from a automotive camera taken during highway driving.
 - **Keywords:** Computer Vision, Deep Learning, OpenCV

#### [P5 - Road Segmentation](./05_road_segmentation)
- **Summary:** Implement the road segmentation using a fully-convolutional network.
- **Keywords:** Deep Learning, Semantic Segmentation


## Sensor Fusion, Localization and Control

> More details: <https://github.com/ndrplz/self-driving-car>

- Extended Kalman Filter, Simulated lidar and radar measurements are used to detect a bicycle that travels around your vehicle. Kalman filter, lidar measurements and radar measurements are used to track the bicycle's position and velocity.

- Utilize an Unscented Kalman Filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Kalman filter, lidar measurements and radar measurements are used to track the bicycle's position and velocity.
 
- Implement a PID controller for keeping the car on track by appropriately adjusting the steering angle.

- Implement an MPC controller for keeping the car on track by appropriately adjusting the steering angle. Differently from previously implemented PID controller, MPC controller has the ability to anticipate future events and can take control actions accordingly. Indeed, future time steps are taking into account while optimizing current time slot.


## Path Planning and System Integration

#### [P6 - Path Planning](https://github.com/LiJiangnanBit/path_optimizer)
- **Summary:** The goal in this project is to build a path planner that is able to create smooth, safe trajectories for the car to follow. The highway track has other vehicles, all going different speeds, but approximately obeying the 50 MPH speed limit. The car transmits its location, along with its sensor fusion data, which estimates the location of all the vehicles on the same side of the road.
- **Keywords:** C++, Path Planning

## End-to-End Self Driving Car

- <https://github.com/Ansheel9/End-to-End-Self-Driving-Car>

## References

- <https://github.com/ndrplz/self-driving-car>
- <https://github.com/vsingla2/Self-Driving-Car-NanoDegree-Udacity>
