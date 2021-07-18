
# Object detection in OpenCV


- Install OpenCV by following [here](https://thecodinginterface.com/blog/opencv-cpp-vscode/)

- Download `frozen_inference_graph.pb` from [learnopencv](https://github.com/spmallick/learnopencv/tree/master/Deep-Learning-with-OpenCV-DNN-Module/input)

- More models can be found in <https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV>

To run the code in C++:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
./build/demo
```
