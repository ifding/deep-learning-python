

This is a traffic sign detector and classifier that uses [dlib](http://dlib.net/) and its implementation of the Felzenszwalb's version of the Histogram of Oriented Gradients (HoG) detector.

The training examples used in this repository are from Brazilian road signs, but the classifier should work with any traffic signs, as long as you train it properly. Google Street View images can be used to train the detectors. 25~40 images are sufficient to train a good detector.

![](https://cloud.githubusercontent.com/assets/294960/7904020/7d216ae0-07c3-11e5-96fe-2b9d020fec4c.png)

Note: all programs accept `-h` as command-line parameter to show a help message.

## Build

```
mkdir build
(cd build; cmake .. && cmake --build .)
```

If you want to enable AVX instructions (make sure you have compatibility):

```
(cd build; cmake .. -DUSE_AVX_INSTRUCTIONS=ON && cmake --build .)
```

## Mark signs on images

1. Compile `imglab`:

```
cd dlib/tools/imglab
mkdir build
cd build
cmake ..
cmake --build .
```

2. Create XML from sample images and Train the fHOG detector:

Please check [transito-cv](https://github.com/fabioperez/transito-cv), there are methods with better results than HoG for traffic sign detector, such as Deep Learning architectures.


## Visualize HOG detectors

To visualize detectors, use the program `view_hog`. Usage:

```
build/view_hog svm_detectors/pare_detector.svm
```

![image](https://cloud.githubusercontent.com/assets/294960/8306983/6fa2ca40-1992-11e5-905d-04260fbfe128.png)


## Detect and Classify

To detect and classify images, run `detect` with the video frames as parameters, use the parameter `--wait` to wait for user input to show next image.

```
build/detect --wait examples/images/*.jpg
```

## Examples

To run the examples:

```
build/detect --wait -u1 examples/images/*
```

![image6](https://cloud.githubusercontent.com/assets/294960/8306981/6ef3e142-1992-11e5-91b0-e753737bcb5f.png)
![image7](https://cloud.githubusercontent.com/assets/294960/8306982/6f7f22c0-1992-11e5-8c2e-4079ddffec47.png)
![image8](https://cloud.githubusercontent.com/assets/294960/8306980/6edb6ae0-1992-11e5-9d77-ddbd0cd59a7b.png)


## References

- <https://github.com/fabioperez/transito-cv>
