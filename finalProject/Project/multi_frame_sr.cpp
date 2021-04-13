
#include <iostream>
#include <cstring>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/superres.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


class MultiFrameSource_CUDA : public cv::superres::FrameSource
{
public:
    MultiFrameSource_CUDA(const std::vector<cv::cuda::GpuMat>& frames);

    void nextFrame(cv::OutputArray frame);
    void reset();

private:
    int index_;
    std::vector<cv::cuda::GpuMat> frames_;
};

MultiFrameSource_CUDA::MultiFrameSource_CUDA(const std::vector<cv::cuda::GpuMat>& frames):
    frames_(frames), index_(0)
{

}

void MultiFrameSource_CUDA::nextFrame(cv::OutputArray _frame)
{
    if (index_ >= 0 && index_ < frames_.size()) {
        frames_[index_++].copyTo(_frame.getGpuMatRef());//这里默认是GpuMat
    } else {
        _frame.release();
    }
}

void MultiFrameSource_CUDA::reset()
{
    index_ = 0;
}

//OpenCV度量时间函数
#define MEASURE_TIME(op) \
{ \
    cv::TickMeter tm; \
    tm.start(); \
    op; \
    tm.stop(); \
    cout << tm.getTimeSec() << " sec" << endl; \
    }

static cv::Ptr<cv::superres::DenseOpticalFlowExt> createOptFlow(const string& name, bool useGpu)
{
    if (name == "farneback")
    {
        if (useGpu)
            return cv::superres::createOptFlow_Farneback_CUDA();
        else
            return cv::superres::createOptFlow_Farneback();
    }
    /*else if (name == "simple")
        return createOptFlow_Simple();*/
    else if (name == "tvl1")
    {
        if (useGpu)
            return cv::superres::createOptFlow_DualTVL1_CUDA();
        else
            return cv::superres::createOptFlow_DualTVL1();
    }
    else if (name == "brox")
        return cv::superres::createOptFlow_Brox_CUDA();
    else if (name == "pyrlk")
        return cv::superres::createOptFlow_PyrLK_CUDA();
    else
        std::cerr << "Incorrect Optical Flow algorithm - " << name << std::endl;

    return cv::Ptr<cv::superres::DenseOpticalFlowExt>();
}

// Laplacian Sharpen算法
cv::Mat sharpenImg2(const cv::Mat& img)
{
    cv::Mat result;
    result.create(img.size(), img.type());
    //Processing the inner edge of the pixel point, the image of the outer edge of the pixel should be additional processing
    for (int row = 1; row < img.rows-1; row++)
    {
        //Front row pixel
        const uchar* previous = img.ptr<const uchar>(row-1);
        //Current line to be processed
        const uchar* current = img.ptr<const uchar>(row);
        //new row
        const uchar* next = img.ptr<const uchar>(row+1);
        uchar *output = result.ptr<uchar>(row);
        int ch = img.channels();
        int starts = ch;
        int ends = (img.cols - 1) * ch;
        for (int col = starts; col < ends; col++)
        {
            //The traversing pointer of the output image is synchronized with the current row, and each channel value of each pixel in each row is given a increment, because the channel number of the image is to be taken into account.
            *output++ = cv::saturate_cast<unsigned char>(5 * current[col] - current[col-ch] - current[col+ch] - previous[col] - next[col]);
        }
    } //end loop
    //Processing boundary, the peripheral pixel is set to 0
    result.row(0).setTo(cv::Scalar::all(0));
    result.row(result.rows-1).setTo(cv::Scalar::all(0));
    result.col(0).setTo(cv::Scalar::all(0));
    result.col(result.cols-1).setTo(cv::Scalar::all(0));
    return result;
}

// 多帧超分辨率算法
int main(int argc, char** argv) {
    std::string optFlowName;
    std::string inputName;
    int iterations = 10;
    if (argc == 1) {
        optFlowName = "farneback";
        inputName = "city";
    } else {
        if (argc == 4) {
            optFlowName = string(argv[1]);
            inputName = string(argv[2]);
            iterations = atoi(argv[3]);
            if (iterations < 1) {
                iterations = 1;
            }
        } else {
            printf("./multi_frame_sr optFlowName inputName iterations\n");
            printf("\toptFlowName: farneback, tvl1, brox, pyrlk\n");
            printf("\tinputName: city, car, iso\n");
            printf("\titerations: integer, 1, 10, etc.\n");
            return -1;
        }
    }

    int scale = 2;
    int num_images = 5;
    int num_times = 10;
    int real_times = 5;
    std::string filenameFormat;
    if (inputName == "city") {
        num_images = 5;
        filenameFormat = "img_%06d.png";
    } else if (inputName == "car") {
        num_images = 4;
        filenameFormat = "car/%d.jpg";
    } else if (inputName == "iso") {
        num_images = 4;
        filenameFormat = "iso/%06d.png";
    } else {
        printf("wrong input\n");
        return -1;
    }

    cv::Ptr<cv::superres::SuperResolution> superRes = cv::superres::createSuperResolution_BTVL1_CUDA();
    int start_i = (num_times - real_times)*num_images;
    std::vector<cv::cuda::GpuMat> frames(num_images*num_times);
    char buf[BUFSIZ];
    for (int i = 0; i<num_images*num_times; i++) {
        memset(buf, '\0', BUFSIZ);
        sprintf(buf, filenameFormat.c_str(), i%num_images+1);
        frames[i] = cv::cuda::GpuMat(cv::imread(buf, cv::IMREAD_COLOR));
        std::cout << buf << ", " << frames[i].size() << "\n";
    }

    cv::TickMeter tm1;
    cv::Ptr<cv::superres::DenseOpticalFlowExt> opticalflow = createOptFlow(optFlowName, true);

    superRes->setOpticalFlow(opticalflow);
    superRes->setScale(scale);
    superRes->setIterations(iterations);
    superRes->setTemporalAreaRadius(1);
    cv::Ptr<cv::superres::FrameSource> frameSource = cv::makePtr<MultiFrameSource_CUDA>(frames);
    superRes->setInput(frameSource);

    cv::Mat result;
    for(int i=0; i<num_images*num_times; i++) {
        if(i==start_i) {
            tm1.start();
        }
        //std::cout << "[" << std::setw(3) << i << "]" << flush;

        //MEASURE_TIME(superRes->nextFrame(result));
        superRes->nextFrame(result);

        if (result.empty()) {
            break;
        }

        //    memset(buf, '\0', BUFSIZ);
        //    sprintf(buf, "result_%d", i);
        //    showImg(result, buf);
    }
    tm1.stop();
    std::cout << tm1.getTimeSec() << " sec" << std::endl;
    std::cout << static_cast<double>(num_images*num_times - start_i) / tm1.getTimeSec() << " FPS" << std::endl;
    cv::imwrite(inputName + "_" + optFlowName +"_sr_result.png", result);
    cv::Mat result2 = sharpenImg2(result);
    cv::imwrite(inputName + "_" + optFlowName +"_sr2_result.png", result2);
}
