/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-23 02:52:22
*/
#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <detector/YOLOv5Detector.h>

#include <tracker/BYTETracker.h>//bytetrack


const int nn_budget = 100;
const float max_cosine_distance = 0.2;

void test_bytetrack(cv::Mat& frame, std::vector<detect_result>& results, BYTETracker& tracker)
{
    std::vector<detect_result> objects;


    for (detect_result dr : results)
    {

        if (dr.class_id == 0) //person
        {
            objects.push_back(dr);
        }
    }


    std::vector<STrack> output_stracks = tracker.update(objects);

    for (unsigned long i = 0; i < output_stracks.size(); i++)
    {
        std::vector<float> tlwh = output_stracks[i].tlwh;
        bool vertical = tlwh[2] / tlwh[3] > 1.6;
        if (tlwh[2] * tlwh[3] > 20 && !vertical)
        {
            cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
            cv::putText(frame, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
                0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
        }
    }


}
int main(int argc, char* argv[])
{
    //bytetrack
    int fps = 20;
    BYTETracker bytetracker(fps, 30);
    //-----------------------------------------------------------------------
    // º”‘ÿ¿‡±√˚≥∆
    std::vector<std::string> classes;
    std::string file = R"(E:\repos\playground\cpp_playground\opencv_onnxrt_test\test_labels_list.txt)";
    std::ifstream ifs(file);
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
    //-----------------------------------------------------------------------

    std::cout << "classes:" << classes.size();
    std::shared_ptr<YOLOv5Detector> detector(new YOLOv5Detector());


    detector->init(k_detect_model_path);

    std::cout << "begin read video" << std::endl;
    cv::VideoCapture capture(R"(C:\Users\12486\Videos\temp\output\2024-04-26 00-01-51.mp4)");

    if (!capture.isOpened()) {
        printf("could not read this video file...\n");
        return -1;
    }
    std::cout << "end read video" << std::endl;
    std::vector<detect_result> results;
    int num_frames = 0;

    cv::VideoWriter video("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(1920, 1080));

    while (true)
    {
        cv::Mat frame;


        if (!capture.read(frame)) // if not success, break loop
        {
            std::cout << "\n Cannot read the video file. please check your video.\n";
            break;
        }

        num_frames++;
        //Second/Millisecond/Microsecond  √Îs/∫¡√Îms/Œ¢√Îus
        auto start = std::chrono::system_clock::now();
        detector->detect(frame, results);
        auto end = std::chrono::system_clock::now();
        auto detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();//ms
        std::cout << classes.size() << ":" << results.size() << ":" << num_frames << std::endl;


        //test_deepsort(frame, results,mytracker);
        test_bytetrack(frame, results, bytetracker);

        cv::imshow("YOLOv5-6.x", frame);

        video.write(frame);

        if (cv::waitKey(30) == 27) // Wait for 'esc' key press to exit
        {
            break;
        }

        results.clear();


    }
    capture.release();
    video.release();
    cv::destroyAllWindows();
}