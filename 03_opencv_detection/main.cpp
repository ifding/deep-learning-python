#include <fstream>
#include <sstream>
#include <mutex>
#include <thread>
#include <queue>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace dnn;

float confThreshold = 0.4, nmsThreshold = 0.4;
std::vector<std::string> classes;

inline void preprocess(const Mat& frame, Net& net, Size inpSize, float scale,
                       const Scalar& mean, bool swapRB);

void postprocess(Mat& frame, const std::vector<Mat>& out, Net& net);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);


template <typename T>
class QueueFPS : public std::queue<T>
{
public:
    QueueFPS() : counter(0) {}

    void push(const T& entry)
    {
        std::lock_guard<std::mutex> lock(mutex);

        std::queue<T>::push(entry);
        counter += 1;
        if (counter == 1)
        {
            // Start counting from a second frame (warmup).
            tm.reset();
            tm.start();
        }
    }

    T get()
    {
        std::lock_guard<std::mutex> lock(mutex);
        T entry = this->front();
        this->pop();
        return entry;
    }

    float getFPS()
    {
        tm.stop();
        double fps = counter / tm.getTimeSec();
        tm.start();
        return static_cast<float>(fps);
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex);
        while (!this->empty())
            this->pop();
    }

    unsigned int counter;

private:
    TickMeter tm;
    std::mutex mutex;
};


int main(int argc, char** argv)
{

    float scale = 1.0;
    Scalar mean = Scalar(104, 177, 123);
    bool swapRB = false;
    int inpWidth = 300;
    int inpHeight = 300;
    size_t asyncNumReq = 0;

    // Open file with classes names.
    std::string file = "./input/object_detection_classes_coco.txt"; 
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }

    std::string modelPath = "./input/frozen_inference_graph.pb";
    std::string configPath = "./input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt";
    std::string framework = "TensorFlow";    

    // Load a model.
    Net net = readNet(modelPath, configPath, framework);
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    // Create a window
    static const std::string kWinName = "Object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Open a video file or an image file or a camera stream.
    //VideoCapture cap;
    VideoCapture cap("./input/video_1.mp4");

    bool process = true;

    // Frames capturing thread
    QueueFPS<Mat> framesQueue;
    std::thread framesThread([&](){
        Mat frame;
        while (process)
        {
            cap >> frame;
            if (!frame.empty())
                framesQueue.push(frame.clone());
            else
                break;
        }
    });

    // Frames processing thread
    QueueFPS<Mat> processedFramesQueue;
    QueueFPS<std::vector<Mat> > predictionsQueue;
    std::thread processingThread([&](){
        std::queue<AsyncArray> futureOutputs;
        Mat blob;
        while (process)
        {
            // Get a next frame
            Mat frame;
            {
                if (!framesQueue.empty())
                {
                    frame = framesQueue.get();
                    if (asyncNumReq)
                    {
                        if (futureOutputs.size() == asyncNumReq)
                            frame = Mat();
                    }
                    else
                        framesQueue.clear();  // Skip the rest of frames
                }
            }

            // Process the frame
            if (!frame.empty())
            {
                preprocess(frame, net, Size(inpWidth, inpHeight), scale, mean, swapRB);
                processedFramesQueue.push(frame);

                if (asyncNumReq)
                {
                    futureOutputs.push(net.forwardAsync());
                }
                else
                {
                    std::vector<Mat> outs;
                    net.forward(outs, outNames);
                    predictionsQueue.push(outs);
                }
            }

            while (!futureOutputs.empty() &&
                   futureOutputs.front().wait_for(std::chrono::seconds(0)))
            {
                AsyncArray async_out = futureOutputs.front();
                futureOutputs.pop();
                Mat out;
                async_out.get(out);
                predictionsQueue.push({out});
            }
        }
    });

    // Postprocessing and rendering loop
    while (waitKey(1) < 0)
    {
        if (predictionsQueue.empty())
            continue;

        std::vector<Mat> outs = predictionsQueue.get();
        Mat frame = processedFramesQueue.get();

        postprocess(frame, outs, net);

        if (predictionsQueue.counter > 1)
        {
            std::string label = format("Camera: %.2f FPS", framesQueue.getFPS());
            putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

            label = format("Network: %.2f FPS", predictionsQueue.getFPS());
            putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

            label = format("Skipped frames: %d", framesQueue.counter - predictionsQueue.counter);
            putText(frame, label, Point(0, 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }
        imshow(kWinName, frame);
    }

    process = false;
    framesThread.join();
    processingThread.join();

    return 0;
}

inline void preprocess(const Mat& frame, Net& net, Size inpSize, float scale,
                       const Scalar& mean, bool swapRB)
{
    static Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0) inpSize.width = frame.cols;
    if (inpSize.height <= 0) inpSize.height = frame.rows;
    blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);

    // Run a model.
    net.setInput(blob, "", scale, mean);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        resize(frame, frame, inpSize);
        Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
}

void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;

    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    CV_Assert(outs.size() > 0);
    for (size_t k = 0; k < outs.size(); k++)
    {
        float* data = (float*)outs[k].data;
        for (size_t i = 0; i < outs[k].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > confThreshold)
            {
                int left   = (int)data[i + 3];
                int top    = (int)data[i + 4];
                int right  = (int)data[i + 5];
                int bottom = (int)data[i + 6];
                int width  = right - left + 1;
                int height = bottom - top + 1;
                if (width <= 2 || height <= 2)
                {
                    left   = (int)(data[i + 3] * frame.cols);
                    top    = (int)(data[i + 4] * frame.rows);
                    right  = (int)(data[i + 5] * frame.cols);
                    bottom = (int)(data[i + 6] * frame.rows);
                    width  = right - left + 1;
                    height = bottom - top + 1;
                }
                classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    }

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if (outLayers.size() > 1)
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confThreshold)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        std::vector<Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;
        for (auto it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            std::vector<Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            std::vector<int> nmsIndices;
            NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                size_t idx = nmsIndices[i];
                nmsBoxes.push_back(localBoxes[idx]);
                nmsConfidences.push_back(localConfidences[idx]);
                nmsClassIds.push_back(it->first);
            }
        }
        boxes = nmsBoxes;
        classIds = nmsClassIds;
        confidences = nmsConfidences;
    }

    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}
