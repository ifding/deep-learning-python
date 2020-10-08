
# Finding Lane Lines on the Road


## Basic Lane Finding Project

For a self driving vehicle to stay in a lane, the first step is to identify lane lines before issuing commands to the control system. Since the lane lines can be of different colors (white, yellow) or forms (solid, dashed) this seemingly trivial task becomes increasingly difficult. Moreover, the situation is further exacerbated with variations in lighting conditions. Thankfully, there are a number of mathematical tools and approaches available nowadays to effectively extract lane lines from an image or dashcam video. 

### Methodology

Before attempting to detect lane lines in a video, a software pipeline is developed for lane detection in a series of images. Only after ensuring that it works satisfactorily for test images, the pipeline is employed for lane detection in a video. 

Consider the test image given below:

![](./test_images/solidWhiteRight.jpg)

1. The test image is first converted to grayscale from RGB using the helper function grayscale().

2. The grayscaled image is given a gaussian blur to remove noise or spurious gradients. 

3. Canny edge detection is applied on this blurred image and a binary image

4. A region of interest is defined to separate the lanes from sorrounding environment and a masked image containing only the lanes is extracted using cv2.bitwise_and() function.

5. This binary image of identified lane lines is finally merged with the original image using cv2.addweighted() function.


 - **Advanced:** Built an advanced lane-finding algorithm using distortion correction, image rectification, color transforms, and gradient thresholding. Identified lane curvature and vehicle displacement. Overcame environmental challenges such as shadows and pavement changes.

[Advanced Lane Finding Project](https://github.com/vsingla2/Self-Driving-Car-NanoDegree-Udacity/blob/master/Term1-Computer-Vision-and-Deep-Learning/Project4-Advanced-Lane_Lines/Advanced-Lane-Lines.ipynb)