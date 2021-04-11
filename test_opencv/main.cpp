#include <iostream>
#include <cstring>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/superres.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

#include <npp.h>
#include <nppcore.h>
#include <nppdefs.h>
#include <nppi.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_filtering_functions.h>
#include <nppi_geometry_transforms.h>
#include <nppi_linear_transforms.h>
#include <nppi_morphological_operations.h>
#include <nppi_statistics_functions.h>
#include <nppi_support_functions.h>
#include <nppi_threshold_and_compare_operations.h>
#include <npps.h>
#include <npps_arithmetic_and_logical_operations.h>
#include <npps_conversion_functions.h>
#include <npps_filtering_functions.h>
#include <npps_initialization.h>
#include <npps_statistics_functions.h>
#include <npps_support_functions.h>

#include <Eigen/Core>

#include <stdio.h>

extern "C" cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

///
/// \brief 测试opencv的mat的内存布局，详情见图片
/// \return
///
int test_opencv_mat_layout()
{
    cout << "Hello World!" << endl;

    string filename = "../subimg0000.png";

    cv::Mat im = cv::imread(filename, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

    int height = im.rows;
    int width = im.cols;
    int channels = im.channels();
    int step = im.step;  // width * channels = 1024 * 3 = 3072
    cout << "step = " << step << endl;

    // 这里表明opencv的mat内存布局
    uchar *data = im.data;
    for(int i=0; i<5;i++) {
        for(int j=0; j<5;j++) {
            cout << i << ", " << j << ": ";
            for (int k=0;k<channels; k++) {
                cout << static_cast<int>(data[i*width*channels + j*channels + k]) << ", ";
            }
            cout << endl;
        }
    }

    cout << "flatted: \n";
    for (int i =0; i < 25; i++) {
        cout << static_cast<int>(data[i]) << ", ";
    }
    cout << endl;

    cv::namedWindow("image");
    cv::imshow("image", im);
    cv::waitKey();

    return 0;
}

///
/// \brief 暗通道去雾算法 Kaiming He
/// \return
///
void dark_channel_prior_defog(const std::string& filename) {
    cv::Mat src = cv::imread(filename, cv::IMREAD_COLOR);
    cv::Mat src_n;
    src.convertTo(src_n, CV_32FC3, 1.0/255.0);
    int window_h = 15;
    int window_w = 15;
    cv::Mat src_expand;
    cv::copyMakeBorder(src_n, src_expand, window_h / 2, window_h/2,
                       window_w/2,window_w/2,cv::BORDER_REPLICATE);
    std::vector<cv::Mat> splits(3);
    cv::split(src_expand, splits);
    int height = src.rows;
    int width = src.cols;
    cv::Mat darkChannelImg(height, width, CV_32FC1);
    double tmp_min, tmp_min_pixel;
    for (int y = 0; y < height; ++y) {
        for(int x = 0; x<width; ++x) {
            tmp_min_pixel = 1.0;
            for(std::vector<cv::Mat>::iterator it = splits.begin();
                it != splits.end(); it++) {
                cv::Mat roi(*it, cv::Rect(x, y, window_w, window_h));
                cv::minMaxLoc(roi, &tmp_min);
                tmp_min_pixel = std::min(tmp_min, tmp_min_pixel);
            }
            darkChannelImg.at<float>(y, x) = static_cast<float>(tmp_min_pixel);
        }
    }

    cv::namedWindow("dark channel");
    cv::imshow("dark channel", darkChannelImg);
    cv::waitKey();

    // top bright pixels
    float num_top = 0.001f * height * width;
    cv::Mat darkChannelImgFlat = darkChannelImg.reshape(1,1);
    cv::Mat_<int> darkChannelImgFlatIndex;
    cv::sortIdx(darkChannelImgFlat, darkChannelImgFlatIndex,
                cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
    cv::Mat maskImg(darkChannelImgFlatIndex.rows,
                    darkChannelImgFlatIndex.cols,
                    CV_8UC1);
    for(int y=0; y<darkChannelImgFlatIndex.rows; y++) {
        for(int x=0; x<darkChannelImgFlatIndex.cols;x++) {
            if(darkChannelImgFlatIndex.at<int>(y,x) <= num_top) {
                maskImg.at<uchar>(y,x) = 1;
            } else {
                maskImg.at<uchar>(y,x) = 0;
            }
        }
    }

    cv::Mat darkChannelImgIndex = maskImg.reshape(1, height);
    std::vector<double> A(3);
    std::vector<double>::iterator itA = A.begin();
    std::vector<cv::Mat>::iterator it = splits.begin();
    std::vector<cv::Mat> expandImgVec(3);
    std::vector<cv::Mat>::iterator itAA = expandImgVec.begin();
    for(; it != splits.end() && itA !=A.end() && itAA != expandImgVec.end();
        it++, itA++, itAA++) {
        cv::Mat roi(*it, cv::Rect(window_w/2, window_h/2, width, height));
        cv::minMaxLoc(roi, 0, &(*itA), 0, 0, darkChannelImgIndex);
        (*itAA) = (*it)/(*itA);
    }

    cv::Mat darkChannelImgA(height, width, CV_32FC1);
    float omega = 0.95f;
    for(int y=0; y<height; y++) {
        for(int x = 0; x<width; x++) {
            tmp_min_pixel = 1.0;
            for(itAA = expandImgVec.begin();
                itAA != expandImgVec.end();
                itAA++) {
                cv::Mat roi(*itAA, cv::Rect(x, y, window_w, window_h));
                cv::minMaxLoc(roi, &tmp_min);
                tmp_min_pixel = std::min(tmp_min, tmp_min_pixel);
            }
            darkChannelImgA.at<float>(y,x) = static_cast<float>(tmp_min_pixel);
        }
    }

    cv::Mat tx = 1.0 - omega * darkChannelImgA;

    float t0 = 0.1f;
    cv::Mat jx(height, width, CV_32FC3);
    for(int y=0; y<height; y++) {
        for(int x=0; x<width;x++) {
            jx.at<cv::Vec3f>(y,x) = cv::Vec3f(
                        (src_n.at<cv::Vec3f>(y,x)[0]-A[0])/std::max(tx.at<float>(y,x),t0)+A[0],
                    (src_n.at<cv::Vec3f>(y,x)[1]-A[1])/std::max(tx.at<float>(y,x),t0)+A[1],
                    (src_n.at<cv::Vec3f>(y,x)[2]-A[2])/std::max(tx.at<float>(y,x),t0)+A[2]);
        }
    }

    cv::namedWindow("jx");
    cv::imshow("jx",jx);
    cv::waitKey();
}


///
/// \brief 暗通道去雾算法 Kaiming He
/// \return
///
void dark_channel_prior_defog_for_polar() {
    cv::Mat src1 = cv::imread("F:/wy/degree0.tiff", cv::IMREAD_ANYDEPTH);
    cv::namedWindow("src1");
    cv::imshow("src1", src1);
    cv::Mat src2 = cv::imread("F:/wy/degree45.tiff", cv::IMREAD_ANYDEPTH);
    cv::Mat src3 = cv::imread("F:/wy/degree90.tiff", cv::IMREAD_ANYDEPTH);
    std::vector<cv::Mat> srcVector;
    int height = src1.rows;
    int width = src1.cols;
    srcVector.push_back(src1);
    srcVector.push_back(src2);
    srcVector.push_back(src3);
    cv::Mat src(height, width, CV_8UC3);
    cv::merge(srcVector, src);
    cv::Mat src_n;
    src.convertTo(src_n, CV_32FC3, 1.0/255.0);
    int window_h = 25;
    int window_w = 25;
    cv::Mat src_expand;
    cv::copyMakeBorder(src_n, src_expand, window_h / 2, window_h/2,
                       window_w/2,window_w/2,cv::BORDER_REPLICATE);
    std::vector<cv::Mat> splits(3);
    cv::split(src_expand, splits);
    cv::Mat darkChannelImg(height, width, CV_32FC1);
    double tmp_min, tmp_min_pixel;
    for (int y = 0; y < height; ++y) {
        for(int x = 0; x<width; ++x) {
            tmp_min_pixel = 1.0;
            for(std::vector<cv::Mat>::iterator it = splits.begin();
                it != splits.end(); it++) {
                cv::Mat roi(*it, cv::Rect(x, y, window_w, window_h));
                cv::minMaxLoc(roi, &tmp_min);
                tmp_min_pixel = std::min(tmp_min, tmp_min_pixel);
            }
            darkChannelImg.at<float>(y, x) = static_cast<float>(tmp_min_pixel);
        }
    }

    cv::namedWindow("dark channel");
    cv::imshow("dark channel", darkChannelImg);
    cv::waitKey();

    // top bright pixels
    float num_top = 0.001f * height * width;
    cv::Mat darkChannelImgFlat = darkChannelImg.reshape(1,1);
    cv::Mat_<int> darkChannelImgFlatIndex;
    cv::sortIdx(darkChannelImgFlat, darkChannelImgFlatIndex,
                cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
    cv::Mat maskImg(darkChannelImgFlatIndex.rows,
                    darkChannelImgFlatIndex.cols,
                    CV_8UC1);
    for(int y=0; y<darkChannelImgFlatIndex.rows; y++) {
        for(int x=0; x<darkChannelImgFlatIndex.cols;x++) {
            if(darkChannelImgFlatIndex.at<int>(y,x) <= num_top) {
                maskImg.at<uchar>(y,x) = 1;
            } else {
                maskImg.at<uchar>(y,x) = 0;
            }
        }
    }

    cv::Mat darkChannelImgIndex = maskImg.reshape(1, height);
    std::vector<double> A(3);
    std::vector<double>::iterator itA = A.begin();
    std::vector<cv::Mat>::iterator it = splits.begin();
    std::vector<cv::Mat> expandImgVec(3);
    std::vector<cv::Mat>::iterator itAA = expandImgVec.begin();
    for(; it != splits.end() && itA !=A.end() && itAA != expandImgVec.end();
        it++, itA++, itAA++) {
        cv::Mat roi(*it, cv::Rect(window_w/2, window_h/2, width, height));
        cv::minMaxLoc(roi, 0, &(*itA), 0, 0, darkChannelImgIndex);
        (*itAA) = (*it)/(*itA);
    }

    cv::Mat darkChannelImgA(height, width, CV_32FC1);
    float omega = 0.95f;
    for(int y=0; y<height; y++) {
        for(int x = 0; x<width; x++) {
            tmp_min_pixel = 1.0;
            for(itAA = expandImgVec.begin();
                itAA != expandImgVec.end();
                itAA++) {
                cv::Mat roi(*itAA, cv::Rect(x, y, window_w, window_h));
                cv::minMaxLoc(roi, &tmp_min);
                tmp_min_pixel = std::min(tmp_min, tmp_min_pixel);
            }
            darkChannelImgA.at<float>(y,x) = static_cast<float>(tmp_min_pixel);
        }
    }

    cv::Mat tx = 1.0 - omega * darkChannelImgA;

    float t0 = 0.1f;
    cv::Mat jx(height, width, CV_32FC3);
    for(int y=0; y<height; y++) {
        for(int x=0; x<width;x++) {
            jx.at<cv::Vec3f>(y,x) = cv::Vec3f(
                        (src_n.at<cv::Vec3f>(y,x)[0]-A[0])/std::max(tx.at<float>(y,x),t0)+A[0],
                    (src_n.at<cv::Vec3f>(y,x)[1]-A[1])/std::max(tx.at<float>(y,x),t0)+A[1],
                    (src_n.at<cv::Vec3f>(y,x)[2]-A[2])/std::max(tx.at<float>(y,x),t0)+A[2]);
        }
    }

    cv::namedWindow("jx");
    cv::imshow("jx",jx);

    std::vector<cv::Mat> resultVector(3);
    cv::split(jx, resultVector);

    cv::namedWindow("jx1");
    cv::imshow("jx1", resultVector[0]);

    cv::waitKey();
}


int test_cuda_add(int argc, char** argv) {
    // std::string filename = std::string(argv[1]);
    // dark_channel_prior_defog(filename);
    // dark_channel_prior_defog_for_polar();

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
           c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}



int main1()
{
    string filename = "F:\\wy\\ImageStackAlignator\\pictures\\IMGP3003.PEF";
    FILE *in = fopen(filename.c_str(), "rb");
    fseek(in, 0, SEEK_END);
    unsigned fsz = ftell(in);
    unsigned char *buffer = (unsigned char *)malloc(fsz);
    if (!buffer)
        return 2;

    cout << fsz << endl;

    fseek(in, 0, SEEK_SET);
    unsigned readb = fread(buffer, 1, fsz, in);
    if (readb != fsz)
        return 3;


    if(buffer)
        free(buffer);

    return 0;
}

vector<float> gaussin_filter_1D(float sigma) {
    if (sigma <= 0) {
        return vector<float>({0, 0,0,0,1,0,0,0,0});
    }
    int size = (int)(sigma /0.6f - 0.4f) * 2 + 1 + 2;
    size = min(size, 99);
    vector<float> ret(size);

    int center = size / 2;
    for(int i=0;i<size;i++) {
        int x = i -center;
        ret[i] = (float)(exp(-(x*x)/(2*sigma*sigma)));
    }
    float sum=0;
    for(int i =0;i<size;i++) {
        sum+=ret[i];
    }
    for(int i=0; i<size;i++) {
        ret[i] /=sum;
    }
    return ret;
}

//测试NPP的图像旋转函数
void test_npp_rotate() {

    //读取图片
    string filename = "../subimg0000.png";

    cv::Mat im = cv::imread(filename, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

    int height = im.rows;
    int width = im.cols;
    int nchannels = im.channels();
    int step = im.step;  // width * channels = 1024 * 3 = 3072
    cout << "step = " << step << endl;

    // 这里表明opencv的mat内存布局
    uchar *data = im.data;

    unsigned char* pSrcData = new unsigned char[height * width * nchannels];
    unsigned char* pDstData = nullptr;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < nchannels; k++) {
                pSrcData[i * width * nchannels + j * nchannels + k] =
                        static_cast<int>(data[i*width*nchannels + j*nchannels + k]);
                // static_cast<unsigned char>((static_cast<double>(rand())/ RAND_MAX) * 255);
            }
        }
    }

    cudaError_t cuRet;
    NppStatus nppRet;
    Npp8u *pSrcDataDevice = nullptr;
    Npp8u *pDstDataDevice = nullptr;

    NppiSize oSrcSize = { 0 };
    NppiSize oDstSize = { 0 };
    NppiRect oSrcROI = { 0 };
    NppiRect oDstROI = { 0 };
    int nImgBpp = 0;
    int nSrcPitch = 0;
    int nDstPitch = 0;
    int nSrcPitchDevice = 0;
    int nDstPitchDevice = 0;
    double aBoundingBox[2][2] = { 0 };
    double nAngle = 0;

    nImgBpp = nchannels;
    oSrcSize.width = width;
    oSrcSize.height = height;
    nSrcPitch = width * nchannels;

    oSrcROI.x = oSrcROI.y = 0;
    oSrcROI.width = oSrcSize.width;
    oSrcROI.height = oSrcSize.height;

    // 分配显存
    pSrcDataDevice = nppiMalloc_8u_C3(oSrcSize.width * nImgBpp, oSrcSize.height, &nSrcPitchDevice);
    cudaMemcpy2D(pSrcDataDevice, nSrcPitchDevice, pSrcData, nSrcPitch,
                 oSrcSize.width * nImgBpp, oSrcSize.height, cudaMemcpyHostToDevice);

    //    nAngle = atof("90");
    //    nppiGetRotateBound(oSrcROI, aBoundingBox, nAngle, 0, 0);
    nAngle = atof("0");
    nppiGetRotateBound(oSrcROI, aBoundingBox, nAngle, 10, 10);
    oDstSize.width = static_cast<int>(ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])));
    oDstSize.height = static_cast<int>(ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])));

    pDstData = new unsigned char[oDstSize.height * oDstSize.width * nchannels];

    nDstPitch = oDstSize.width * nchannels;
    oDstROI.x = oDstROI.y = 0;
    oDstROI.width = oDstSize.width;
    oDstROI.height = oDstSize.height;

    pDstDataDevice = nppiMalloc_8u_C3(oDstSize.width, oDstSize.height, &nDstPitchDevice);
    cudaMemset2D(pDstDataDevice, nDstPitchDevice, 0, oDstSize.width * nDstPitch, oDstSize.height);
    nppRet = nppiRotate_8u_C3R(pSrcDataDevice, oSrcSize, nSrcPitchDevice, oSrcROI,
                               pDstDataDevice, nDstPitchDevice, oDstROI,
                               nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_CUBIC);
    cudaMemcpy2D(pDstData, nDstPitch, pDstDataDevice, nDstPitchDevice, oDstSize.width * nImgBpp, oDstSize.height, cudaMemcpyDeviceToHost);

    //    for (int i = 0; i < oDstSize.height; i++) {
    //        for (int j = 0; j < oDstSize.width; j++) {
    //            std::cout << i << ", " << j << ": ";
    //            for (int k = 0; k < nchannels; k++) {
    //                std::cout << static_cast<int>(pSrcData[i * oDstSize.width * nchannels + j * nchannels + k]) << ", ";
    //            }
    //            std::cout << std::endl;
    //        }
    //    }

    // 这里先分配一个图像，保存结果，然后直接用内存复制的方法将结果复制到图片中，然后保存
    cv::Mat dst = cv::Mat::zeros(oDstSize.height, oDstSize.width, CV_8UC3);
    memcpy(dst.data, pDstData, oDstSize.height * oDstSize.width * nchannels );
    cv::imwrite("dst.png", dst);

    delete[] pSrcData;
    delete[] pDstData;

    nppiFree(pSrcDataDevice);
    nppiFree(pDstDataDevice);

    cudaDeviceReset();
}

//显示图片
void showImg(const cv::Mat& mat, const std::string& title) {
    cv::namedWindow(title);
    cv::imshow(title, mat);
    cv::waitKey();
}

//从src中截取一个RotatedRect图像块
cv::Mat cropImg(const cv::Mat& src, int cx, int cy, int w, int h, float angle) {
    cv::Point2f center(static_cast<float>(cx),
                       static_cast<float>(cy));
    cv::Size size(w, h);
    cv::RotatedRect rect(center, size, angle);
    float rect_angle = rect.angle;
    cv::Size rect_size = rect.size;
    if (rect.angle < -45.0f) {
        rect_angle += 90.0f;
        std::swap(rect_size.width, rect_size.height);
    }
    cv::Mat M = cv::getRotationMatrix2D(rect.center, rect_angle, 1.0);
    cv::Mat rotated, cropped;
    cv::warpAffine(src, rotated, M, src.size(), cv::INTER_CUBIC);
    cv::getRectSubPix(rotated, rect_size, rect.center, cropped);
    return cropped;
}

cv::Mat sharpenImg(const cv::Mat& src) {
    // sharpen image using "unsharp mask" algorithm
    cv::Mat blurred;
    double sigma = 1, threshold = 5, amount = 1;
    cv::GaussianBlur(src, blurred, cv::Size(), sigma, sigma);
    cv::Mat lowContrastMask = cv::abs(src - blurred) < threshold;
    cv::Mat sharpened = src*(1+amount) + blurred*(-amount);
    src.copyTo(sharpened, lowContrastMask);
    return sharpened;
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

// 使用OpenCV自带的DNN超分辨算法进行超分辨率
void dnn_sr(int argc, char** argv) {
    std::string algo = string(argv[1]);
    std::string model_path = string(argv[2]);
    std::cout << algo << ", " << model_path << "\n";

    cv::Mat src = cv::imread("F:\\imreg_fmt\\build_release\\Release\\overlay_image.png", cv::IMREAD_COLOR);
    int scale = 2;
    cv::Mat src_up;

    cv::dnn_superres::DnnSuperResImpl sr;
    sr.readModel(model_path);
    sr.setModel(algo, scale);
    sr.upsample(src, src_up);

    showImg(src, "src");
    showImg(src_up, "src_up");

    char buf[BUFSIZ];
    memset(buf, '\0', BUFSIZ);
    sprintf(buf, "dnnsr_%s_%d.png", algo, scale);
    std::cout << "filename: " << buf << "\n";
    cv::imwrite(buf, src_up);
}


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

// OpenCV里面的视频超分辨率算法，多帧超分辨率算法
void cv_mfsr(int argc, char** argv) {

    std::string optFlowName = string(argv[1]);
    int scale = 2;

    std::string image_root = "F:/cuda-samples/Samples/testNPP/test_opencv";//在这个文件夹下加载序列图片
    int num_images = 5;
    cv::Ptr<cv::superres::SuperResolution> superRes = cv::superres::createSuperResolution_BTVL1_CUDA();
    std::vector<cv::cuda::GpuMat> frames(num_images*2);
    char buf[BUFSIZ];
    for (int i = 0; i<num_images*2; i++) {
        memset(buf, '\0', BUFSIZ);
        sprintf(buf, "%s/img_%06d.png", image_root.c_str(), i%num_images);
        frames[i] = cv::cuda::GpuMat(cv::imread(buf, cv::IMREAD_COLOR));
        std::cout << buf << ", " << frames[i].size() << "\n";
    }

    cv::Ptr<cv::superres::DenseOpticalFlowExt> opticalflow = createOptFlow(optFlowName, true);

    superRes->setOpticalFlow(opticalflow);
    superRes->setScale(scale);
    superRes->setIterations(100);
    superRes->setTemporalAreaRadius(1);
    cv::Ptr<cv::superres::FrameSource> frameSource =
            cv::makePtr<MultiFrameSource_CUDA>(frames);
    superRes->setInput(frameSource);

    cv::TickMeter tm1;
    cv::Mat result;
    for(int i=0; i<num_images*2; i++) {
        if(i==num_images) {
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
    std::cout << static_cast<double>(num_images) / tm1.getTimeSec() << " FPS" << std::endl;
    //保存结果
    showImg(result, "sr_result.png");
    cv::imwrite("sr_result.png", result);
}


cv::Mat getApodizationWindow(int rows, int cols, int radius) {
    int size = 2*radius;
    float *hanning_window = new float[size];
    for(int i=0; i<size;i++) {
        hanning_window[i] = 0.5 - 0.5 * std::cos((2 * static_cast<double>(CV_PI) * i)/(size - 1));
    }
    cv::Mat a = cv::Mat::ones(rows, 1, CV_32F);
    float* a_ptr = a.ptr<float>();
    memcpy(&a_ptr[0], &hanning_window[0], radius*sizeof(float));
    memcpy(&a_ptr[rows-radius], &hanning_window[radius], radius*sizeof(float));

    cv::Mat b = cv::Mat::ones(1, cols, CV_32F);
    float* b_ptr = b.ptr<float>();
    memcpy(&b_ptr[0], &hanning_window[0], radius*sizeof(float));
    memcpy(&b_ptr[cols-radius], &hanning_window[radius], radius*sizeof(float));

    cv::Mat c = a*b;  //矩阵相乘, (rows x 1) * (1 x cols) = (rows x cols)
    return c;
}

float* getHighPassFilter(int rows, int cols) {
    float* filter = new float[rows*cols];
    double row_step = CV_PI / static_cast<double>(rows-1);
    double col_step = CV_PI / static_cast<double>(cols-1);
    double t1, t2;
    double PI_2 = -CV_PI / static_cast<double>(2);
    for(int i=0;i<rows;i++) {
        t1 = i*row_step + PI_2 ;
        t1 *= t1;
        for(int j=0;j<cols;j++) {
            t2 = j*col_step + PI_2;
            t2 *= t2;
            t2 = std::cos(std::sqrt(t1 + t2));
            t2 *= t2;
            t2 = 1.0 - t2;
            filter[i*cols+j] = t2;
        }
    }
    return filter;
}

// 基于FFT的图像配准
extern "C" void copy_R2C(float* rs, cufftDoubleComplex* cs, int N);\
extern "C" void fftshift_2D(cufftDoubleComplex* cs, int width, int height);
extern "C" void high_pass_filtering(cufftDoubleComplex* input, float *output, int width, int height);
extern "C" void crossPowerSpectrum(cufftDoubleComplex* f1, cufftDoubleComplex* f2, int width, int height);
extern "C" void abs_and_normby(cufftDoubleComplex* f1, float* output, double normby, int width, int height);
void fftreg_phaseCorrelate(cv::cuda::GpuMat& im0,
                           cv::cuda::GpuMat& im1,
                           double& row,
                           double& col) {
    bool debug = true;
    int rows_ = im0.rows;
    int cols_ = im0.cols;
    //先测试对apodize0_g进行傅里叶变换
    //cv::Mat apodize0_copy = cv::Mat::zeros(rows_, cols_, CV_32FC1);
    float* data_ptr_d0 = im0.ptr<float>();
    float* data_ptr_d1 = im1.ptr<float>();
    //float* data_ptr_h = apodize0_copy.ptr<float>();
    //cudaMemcpy(data_ptr_h, data_ptr_d0, rows_*cols_*sizeof(float), cudaMemcpyDeviceToHost);
    cufftDoubleComplex *im0_g_c, *im1_g_c;
    cufftDoubleComplex *im0_g_c_o, *im1_g_c_o; // 傅里叶变换的输出
    cudaMalloc((void**)&im0_g_c, sizeof(cufftDoubleComplex)*rows_*cols_);
    cudaMalloc((void**)&im0_g_c_o, sizeof(cufftDoubleComplex)*rows_*cols_);
    cudaMalloc((void**)&im1_g_c, sizeof(cufftDoubleComplex)*rows_*cols_);
    cudaMalloc((void**)&im1_g_c_o, sizeof(cufftDoubleComplex)*rows_*cols_);
    copy_R2C(data_ptr_d0, im0_g_c, rows_*cols_); //复制到complex数据里面 GPU端
    copy_R2C(data_ptr_d1, im1_g_c, rows_*cols_); //复制到complex数据里面 GPU端

    cufftHandle plan;
    cufftPlan2d(&plan, rows_, cols_, CUFFT_Z2Z);
    cufftExecZ2Z(plan, im0_g_c, im0_g_c_o, CUFFT_FORWARD);//执行FFT
    cufftExecZ2Z(plan, im1_g_c, im1_g_c_o, CUFFT_FORWARD);//执行FFT

    crossPowerSpectrum(im0_g_c_o, im1_g_c_o, cols_, rows_); //结果存在im0_g_c_o里面了

    if (debug) {
        std::cout << "check cross ps:\n";
        cufftDoubleComplex *im0_g_c_o_h = new cufftDoubleComplex[rows_*cols_];
        cudaMemcpy(im0_g_c_o_h, im0_g_c_o, sizeof(cufftDoubleComplex)*rows_*cols_, cudaMemcpyDeviceToHost);
        for(int i=0; i<5;i++) {
            for(int j=0; j<5;j++) {
                const cufftDoubleComplex& temp = im0_g_c_o_h[i*cols_+j];
                std::cout << "(" << temp.x << "," << temp.y << "), ";
            }
            std::cout << "\n";
        }
        delete[] im0_g_c_o_h;
    }

    cufftExecZ2Z(plan, im0_g_c_o, im0_g_c, CUFFT_INVERSE);//执行FFT逆变换

    fftshift_2D(im0_g_c, cols_, rows_);  // OK

    float *abs_g_data;
    cudaMalloc((void**)&abs_g_data, sizeof(float)*rows_*cols_);
    abs_and_normby(im0_g_c, abs_g_data, static_cast<double>(rows_*cols_), cols_, rows_);

    cv::cuda::GpuMat abs_g(rows_, cols_, CV_32FC1, abs_g_data);
    cv::Point2i minLoc, maxLoc;
    double minMaxVals[2];
    cv::cuda::minMaxLoc(abs_g, &minMaxVals[0], &minMaxVals[1], &minLoc, &maxLoc);

    if (debug) {
        cv::Mat abs_h;
        abs_g.download(abs_h);
        float *abs_h_data = abs_h.ptr<float>();
        std::cout << "abs_h data:\n";
        for(int i=0;i<5;i++) {
            for(int j=0;j<5;j++) {
                std::cout << abs_h_data[i*cols_+j] << ", ";
            }
            std::cout << "\n";
        }

        std::cout << "minVal, maxVal: " << minMaxVals[0] << ", " << minMaxVals[1] << "\n";
        std::cout << "loc_h: " << minLoc << ", " << maxLoc << "\n";
    }

    // neibghborhood
    cv::Mat abs_h;
    abs_g.download(abs_h);
    int radius = 2;
    int size = 1 + 2*radius;
    int row_start = maxLoc.y - radius;
    int row_end = maxLoc.y + radius;
    int col_start = maxLoc.x - radius;
    int col_end = maxLoc.x + radius;
    cv::Range rowRange(maxLoc.y - radius, maxLoc.y + radius);
    cv::Range colRange(maxLoc.x - radius, maxLoc.x + radius);


    cudaFree(im0_g_c);
    cudaFree(im1_g_c);
    cudaFree(im0_g_c_o);
    cudaFree(im1_g_c_o);
    cudaFree(abs_g_data);
    cufftDestroy(plan);
}

void fft_image_registration(int argc, char** argv) {
    bool debug = true;
    cv::Mat im0 = cv::imread("F:/cuda-samples/Samples/testNPP/test_opencv/img_000002.png", cv::IMREAD_COLOR);
    cv::Mat im1 = cv::imread("F:/cuda-samples/Samples/testNPP/test_opencv/img_000000.png", cv::IMREAD_COLOR);
    cv::Mat gray0, gray1;
    cv::cvtColor(im0, gray0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(im1, gray1, cv::COLOR_BGR2GRAY);
    cv::Mat gray0f, gray1f;
    gray0.convertTo(gray0f, CV_32F, 1/255.0);
    gray1.convertTo(gray1f, CV_32F, 1/255.0);
    if(debug) {
        showImg(gray0, "gray0");
        showImg(gray1, "gray1");
    }

    int rows_ = im0.rows;
    int cols_ = im0.cols;
    int log_polar_size_ = std::max(rows_, cols_);
    int logPolarrows_ = log_polar_size_;
    int logPolarcols_ = log_polar_size_;
    double logBase_ = std::exp(std::log(rows_ * 1.1 / 2.0) / std::max(rows_, cols_));//根据图像的宽高得到对数基底
    float *scales = new float[logPolarcols_];
    float ellipse_coefficient = (float)(rows_) / cols_;  //得到椭圆系数
    for (int i = 0; i < logPolarcols_; i++)
    {
        scales[i] = std::pow(logBase_, i);  //向量中填充数值
    }
    float *scales_matrix = new float[logPolarrows_ * logPolarcols_];
    for (int j = 0; j < logPolarrows_; j++) {
        memcpy(&scales_matrix[j*logPolarcols_], scales, logPolarcols_*sizeof(float));
    }

    if (debug) {
        std::cout << "scales_matrix: \n";
        for(int i=0; i<5; i++) {
            for(int j=0; j<5;j++) {
                std::cout << scales_matrix[i*logPolarcols_+j] << ", ";
            }
            std::cout << "\n";
        }
    }

    double angle_step = CV_PI / static_cast<double>(logPolarrows_-1);
    float *angles_matrix = new float[logPolarrows_*logPolarcols_];
    for (int i = 0; i < logPolarrows_; i++) {
        //memset(&angles_matrix[i*logPolarcols_], -i*angle_step, logPolarcols_*sizeof(float));  // this is bad
        for(int j =0; j<logPolarcols_;j++) {
            angles_matrix[i*logPolarcols_+j] = -i*angle_step;
        }
    }

    if (debug) {
        std::cout << "angles_matrix: \n";
        for(int i=0; i<5; i++) {
            for(int j=0; j<5;j++) {
                std::cout << angles_matrix[i*logPolarcols_+j] << ", ";
            }
            std::cout << "\n";
        }
    }

    float center[2] = { cols_/2.0f, rows_/2.0f};  //中心
    float *xMap = new float[logPolarrows_*logPolarcols_];
    float *yMap = new float[logPolarrows_*logPolarcols_];
    int index;
    for (int i=0; i<logPolarrows_;i++) {
        for (int j=0; j<logPolarcols_;j++) {
            index = i*logPolarcols_+j;
            xMap[index] = scales_matrix[index] * std::cosf(angles_matrix[index]) + center[0];
            yMap[index] = scales_matrix[index] * std::sinf(angles_matrix[index]) + center[1];
        }
    }

    cv::Mat border_mask_ = cv::Mat(rows_, cols_, CV_8UC1, cv::Scalar(255));
    cv::Mat appodizationWindow = getApodizationWindow(rows_, cols_, (int)((0.12)*std::min(rows_, cols_)));
    if (debug) {
        std::cout << "appodizationWindow: \n";
        float* data_ptr = appodizationWindow.ptr<float>();
        for(int i=0; i<5; i++) {
            for(int j=0; j<5;j++) {
                std::cout << data_ptr[i*cols_+j] << ", ";
            }
            std::cout << "\n";
        }
    }

    float* highPassFilter = getHighPassFilter(rows_, cols_);

    if (false) {
        cv::Mat apodize0 = gray0f.mul(appodizationWindow);
        cv::Mat apodize1 = gray1f.mul(appodizationWindow);
        showImg(apodize0, "apodize0");
        showImg(apodize1, "apodize1");
    } else {
        cv::cuda::GpuMat gray0f_g, gray1f_g;
        cv::cuda::GpuMat appodizationWindow_g;
        gray0f_g.upload(gray0f);
        gray1f_g.upload(gray1f);
        appodizationWindow_g.upload(appodizationWindow);

        cv::cuda::GpuMat apodize0_g, apodize1_g;
        cv::cuda::multiply(gray0f_g, appodizationWindow_g, apodize0_g); //矩阵element-wise相乘
        cv::cuda::multiply(gray1f_g, appodizationWindow_g, apodize1_g);

        //        cv::Mat apodize0, apodize1;
        //        apodize0_g.download(apodize0);
        //        apodize1_g.download(apodize1);
        //        showImg(apodize0, "apodize0");
        //        showImg(apodize1, "apodize1");

        if (false) {//测试GpuMat的数据复制, OK! 说明
            cv::Mat apodize0_copy = cv::Mat::zeros(rows_, cols_, CV_32FC1);
            float* data_ptr_d = apodize0_g.ptr<float>();
            float* data_ptr_h = apodize0_copy.ptr<float>();
            cudaError_t cuda_result = cudaMemcpy(data_ptr_h, data_ptr_d, rows_*cols_*sizeof(float), cudaMemcpyDeviceToHost);
            if (cuda_result != cudaSuccess) {
                std::cout << "error in copy from device to host.\n";
            } else {
                showImg(apodize0_copy, "apodize0_copy");
            }
        }

        //先测试对apodize0_g进行傅里叶变换
        //cv::Mat apodize0_copy = cv::Mat::zeros(rows_, cols_, CV_32FC1);
        float* data_ptr_d0 = apodize0_g.ptr<float>();
        float* data_ptr_d1 = apodize1_g.ptr<float>();
        //float* data_ptr_h = apodize0_copy.ptr<float>();
        //cudaMemcpy(data_ptr_h, data_ptr_d0, rows_*cols_*sizeof(float), cudaMemcpyDeviceToHost);
        cufftDoubleComplex *apodize0_g_c, *apodize1_g_c;
        cufftDoubleComplex *apodize0_g_c_o, *apodize1_g_c_o; // 傅里叶变换的输出
        cudaMalloc((void**)&apodize0_g_c, sizeof(cufftDoubleComplex)*rows_*cols_);
        cudaMalloc((void**)&apodize0_g_c_o, sizeof(cufftDoubleComplex)*rows_*cols_);
        cudaMalloc((void**)&apodize1_g_c, sizeof(cufftDoubleComplex)*rows_*cols_);
        cudaMalloc((void**)&apodize1_g_c_o, sizeof(cufftDoubleComplex)*rows_*cols_);
        cufftDoubleComplex *apodize0_g_c_o_h = new cufftDoubleComplex[rows_*cols_]; // 傅里叶变换的输出的Host端存储，用于debug
        copy_R2C(data_ptr_d0, apodize0_g_c, rows_*cols_); //复制到complex数据里面 GPU端
        copy_R2C(data_ptr_d1, apodize1_g_c, rows_*cols_); //复制到complex数据里面 GPU端
        //        for(int i=0; i<rows_;i++) {
        //            for(int j=0;j<cols_;j++) {
        //                apodize0_g_c_o_h[i*cols_+j].x = static_cast<double>(data_ptr_h[i*cols_+j]);
        //                apodize0_g_c_o_h[i*cols_+j].y = 0.0;
        //            }
        //        }
        //        cudaMemcpy(apodize0_g_c, apodize0_g_c_o_h, sizeof(cufftDoubleComplex)*rows_*cols_, cudaMemcpyHostToDevice);

        cufftHandle plan;
        cufftPlan2d(&plan, rows_, cols_, CUFFT_Z2Z);
        cufftExecZ2Z(plan, apodize0_g_c, apodize0_g_c_o, CUFFT_FORWARD);//执行FFT
        cufftExecZ2Z(plan, apodize1_g_c, apodize1_g_c_o, CUFFT_FORWARD);//执行FFT

        if (debug){
            cudaMemcpy(apodize0_g_c_o_h, apodize0_g_c_o, sizeof(cufftDoubleComplex)*rows_*cols_, cudaMemcpyDeviceToHost);
            std::cout << "apodize0 after fft: \n";
            for(int i=0; i<5;i++) {
                for(int j=0; j<5;j++) {
                    const cufftDoubleComplex& temp = apodize0_g_c_o_h[i*cols_+j];
                    std::cout << "(" << temp.x << ", " << temp.y << "), ";
                }
                std::cout << "\n";
            }

            FILE *fp = fopen("cufft_result.txt", "w");
            for(int i=0; i<rows_;i++) {
                for(int j=0; j<cols_;j++) {
                    const cufftDoubleComplex& temp = apodize0_g_c_o_h[i*cols_+j];
                    fprintf(fp, "[%.6f, %.6f], ", temp.x, temp.y);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }

        //这里执行fftshift
        fftshift_2D(apodize0_g_c_o, cols_, rows_);  // OK
        fftshift_2D(apodize1_g_c_o, cols_, rows_);  // OK

        int block_rows = rows_/2;
        int block_cols = cols_/2;

        if (debug){
            cudaMemcpy(apodize0_g_c_o_h, apodize0_g_c_o, sizeof(cufftDoubleComplex)*rows_*cols_, cudaMemcpyDeviceToHost);
            std::cout << "apodize0 after fftshift: \n";
            for(int i=block_rows; i<block_rows+5;i++) {
                for(int j=block_cols; j<block_cols+5;j++) {
                    const cufftDoubleComplex& temp = apodize0_g_c_o_h[i*cols_+j];
                    std::cout << "(" << temp.x << ", " << temp.y << "), ";
                }
                std::cout << "\n";
            }

            FILE *fp = fopen("cufft_shift_result.txt", "w");
            for(int i=0; i<rows_;i++) {
                for(int j=0; j<cols_;j++) {
                    const cufftDoubleComplex& temp = apodize0_g_c_o_h[i*cols_+j];
                    fprintf(fp, "[%.6f, %.6f], ", temp.x, temp.y);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);

            //
            fp = fopen("highPassFilter.txt", "w");
            for(int i=0; i<rows_;i++) {
                for(int j=0; j<cols_;j++) {
                    fprintf(fp, "%.6f, ", highPassFilter[i*cols_+j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);

            std::cout << "fftshift * highpass result is: \n";
            for(int i=0; i<5;i++) {
                for(int j=0;j<5;j++) {
                    const cufftDoubleComplex& temp = apodize0_g_c_o_h[i*cols_+j];
                    double x = temp.x * highPassFilter[i*cols_+j];
                    double y = temp.y * highPassFilter[i*cols_+j];
                    x = sqrt(x*x+y*y);

                    std::cout << x << ", ";
                }
                std::cout << "\n";
            }
        }

        // fftshift之后的数据，再乘以highPassFilter
        float *im0_dft_g_data, *im1_dft_g_data;
        cudaMalloc((void**)&im0_dft_g_data, sizeof(float)*rows_*cols_);
        cudaMalloc((void**)&im1_dft_g_data, sizeof(float)*rows_*cols_);
        high_pass_filtering(apodize0_g_c_o, im0_dft_g_data, cols_, rows_);
        high_pass_filtering(apodize1_g_c_o, im1_dft_g_data, cols_, rows_);

        if (debug) {
            float *im0_dft_h = new float[rows_ * cols_];
            cudaMemcpy(im0_dft_h, im0_dft_g_data, sizeof(float)*rows_*cols_, cudaMemcpyDeviceToHost);
            std::cout << "download data from gpu\n";
            //cv::Mat im0_dft_h;
            //cv::cuda::GpuMat im0_dft_g(rows_, cols_, CV_32F);
            // cudaMemcpy(im0_dft_g.ptr<float>(), im0_dft_g_data, sizeof(float)*rows_*cols_, cudaMemcpyDeviceToDevice);
            //im0_dft_g.download(im0_dft_h);
            //float *im0_dft_h_data = im0_dft_h.ptr<float>();

            std::cout << "apodize0 after high pass filtering: \n";
            for(int i=0; i<5;i++) {
                for(int j=0; j<5;j++) {
                    //std::cout << im0_dft_h_data[i*cols_+j] << ", ";
                    std::cout << im0_dft_h[i*cols_+j] << ", ";
                }
                std::cout << "\n";
            }

            delete[] im0_dft_h;
        }

        if (debug) {
            cv::Mat im0_dft_h;
            cv::cuda::GpuMat im0_dft_g(rows_, cols_, CV_32F, im0_dft_g_data);
            im0_dft_g.download(im0_dft_h);

            float *im0_dft_h_data = im0_dft_h.ptr<float>();
            std::cout << "apodize0 after high pass filtering using GpuMat: \n";
            for(int i=0; i<5;i++) {
                for(int j=0; j<5;j++) {
                    std::cout << im0_dft_h_data[i*cols_+j] << ", ";
                }
                std::cout << "\n";
            }
        }

        //cv::remap(src, dst, cv_xMap, cv_yMap, cv::INTER_CUBIC & cv::INTER_MAX, cv::BORDER_CONSTANT, cv::Scalar());
        cv::cuda::GpuMat im0_dft_g(rows_, cols_, CV_32F, im0_dft_g_data);
        cv::cuda::GpuMat im1_dft_g(rows_, cols_, CV_32F, im1_dft_g_data);
        cv::cuda::GpuMat im0_log_polar_g(logPolarrows_, logPolarcols_, CV_32FC1);
        cv::cuda::GpuMat im1_log_polar_g(logPolarrows_, logPolarcols_, CV_32FC1);
        cv::Mat xMap_h(logPolarrows_, logPolarcols_, CV_32FC1, xMap);
        cv::Mat yMap_h(logPolarrows_, logPolarcols_, CV_32FC1, yMap);
        cv::cuda::GpuMat xMap_g, yMap_g;
        xMap_g.upload(xMap_h);
        yMap_g.upload(yMap_h);
        cv::cuda::remap(im0_dft_g, im0_log_polar_g, xMap_g, yMap_g, cv::INTER_CUBIC & cv::INTER_MAX, cv::BORDER_CONSTANT, cv::Scalar());
        cv::cuda::remap(im1_dft_g, im1_log_polar_g, xMap_g, yMap_g, cv::INTER_CUBIC & cv::INTER_MAX, cv::BORDER_CONSTANT, cv::Scalar());

        if (debug) {
            cv::Mat im0_log_polar_h, im1_log_polar_h;
            im0_log_polar_g.download(im0_log_polar_h);
            im1_log_polar_g.download(im1_log_polar_h);
            float *im0_log_polar_h_data = im0_log_polar_h.ptr<float>();
            std::cout << "im0_log_polar_h_data: \n";
            for(int i=0; i<5;i++) {
                for(int j=0; j<5;j++) {
                    std::cout << im0_log_polar_h_data[i*cols_+j] << ", ";
                }
                std::cout << "\n";
            }
            showImg(im0_log_polar_h, "im0_log_polar_h");
            showImg(im1_log_polar_h, "im1_log_polar_h");
        }

        double rs_row, rs_col;
        double t_row, t_col;
        double scale, rotation;
        fftreg_phaseCorrelate(im0_log_polar_g, im1_log_polar_g, rs_row, rs_col);

        delete[] apodize0_g_c_o_h;
        cudaFree(apodize0_g_c);
        cudaFree(apodize0_g_c_o);
        cudaFree(apodize1_g_c);
        cudaFree(apodize1_g_c_o);
        cufftDestroy(plan);
        cudaFree(im0_dft_g_data);
        cudaFree(im1_dft_g_data);
    }

    if (debug){
        std::cout << "highPassFilter: \n";
        for(int i=0; i<5; i++) {
            for(int j=0; j<5;j++) {
                std::cout << highPassFilter[i*cols_+j] << ", ";
            }
            std::cout << "\n";
        }
    }

    // 对apodize0和apodize1做fft，然后fftShift，然后


    // release
    delete[] scales_matrix;
    delete[] scales;
    delete[] angles_matrix;
    delete[] xMap;
    delete[] yMap;
    delete[] highPassFilter;

}

//测试Eigen的Complex的cwiseAbs是求模值还是求绝对值
void test_eigen() {
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ComplexMatrix;
    ComplexMatrix A(2, 2);
    A(0, 0) = std::complex<double>(1.0, 1.0);
    A(0, 1) = std::complex<double>(2.0, 2.0);
    A(1, 0) = std::complex<double>(3.0, 3.0);
    A(1, 1) = std::complex<double>(4.0, 4.0);
    std::cout << A << "\n";
    std::cout << A.cwiseAbs() << "\n";  //如果是complex，则是求sqrt(x.^2 + y.^2)

    Eigen::MatrixXd B(2, 2);
    B(0, 0) = 1.0;
    B(0, 1) = 2.0;
    B(1, 0) = 3.0;
    B(1, 1) = 4.0;
    std::cout << B << "\n";
    ComplexMatrix result = A.cwiseProduct(B);  //如果是一个Complex矩阵和一个普通矩阵相乘，则实部和虚部都会相乘
    std::cout << "result of A.cwiseProduct(B) is " << result << "\n";
    std::cout << result.cwiseAbs() << "\n";

    ComplexMatrix C(2, 2);
    C(0, 0) = std::complex<double>(1.0, 2.0);
    C(0, 1) = std::complex<double>(2.0, 3.0);
    C(1, 0) = std::complex<double>(3.0, 4.0);
    C(1, 1) = std::complex<double>(4.0, 5.0);

    std::cout << "A" << A << "\n";
    std::cout << "C" << C << "\n";
    std::cout << "C conjugate" << C.conjugate() << "\n";

    ComplexMatrix D = A.cwiseProduct(C.conjugate());
    std::cout << D << "\n";

    Eigen::MatrixXd Dabs = D.cwiseAbs();
    std::cout << "Dabs" << Dabs << "\n";
    int row, col;
    std::cout << Dabs.maxCoeff(&row, &col) << "\n";
    std::cout << row << ", " << col << "\n";
}

void dark_prior(cv::cuda::GpuMat &gpuimg, int radius, cv::cuda::GpuMat &dark_prior)
{
    int win_size = 2 * radius + 1;
    cv::cuda::GpuMat gpuimg_splitted[3];
    cv::cuda::GpuMat temp1, temp2;

    //GpuMat dark_prior;
    const cv::Mat kernel =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(win_size, win_size));

    cv::cuda::split(gpuimg, gpuimg_splitted);
    cv::cuda::min(gpuimg_splitted[0], gpuimg_splitted[1], temp1);
    cv::cuda::min(gpuimg_splitted[2], temp1, temp2);
    cv::Ptr<cv::cuda::Filter>minfilter =
            cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_32FC1, kernel);
    minfilter->apply(temp2, dark_prior);

}

void sum_by_indices(float *in, int *indices, int size, int width, int height, int channels,
                    float *out) {
    // out为RGB数组
    int row, col;
    for(int index=0;index<size;index++) {
        row = indices[index]/width;
        col = indices[index]%width;
        for(int c=0;c<channels;c++){
            out[c] += in[row*width*channels+col*channels+c];
        }
    }
}
void test_defog() {
    bool debug = false;


    cv::Mat Iper = cv::imread("F:/wy/ImageWorst_tiff16.tiff", cv::IMREAD_ANYCOLOR|cv::IMREAD_ANYDEPTH);
    cv::Mat Ipar = cv::imread("F:/wy/ImageBest_tiff16.tiff", cv::IMREAD_ANYCOLOR|cv::IMREAD_ANYDEPTH);
    int num_images = 1;
    cv::TickMeter tm;
    tm.start();
    for(int image_id=0; image_id<num_images; image_id++) {
        Iper.convertTo(Iper, CV_32FC3, 1/65535.0);
        Ipar.convertTo(Ipar, CV_32FC3, 1/65535.0);
        cv::cuda::GpuMat Iper_g;
        Iper_g.upload(Iper);
        //Ipar_g.upload(Ipar);

        cv::cuda::GpuMat Iper_dc_g;
        //cv::cuda::GpuMat Ipar_dc_g;
        dark_prior(Iper_g, 12, Iper_dc_g);
        //dark_prior(Ipar_g, 12, Ipar_dc_g);

        cv::Mat Iper_dc;
        Iper_dc_g.download(Iper_dc);
        //Ipar_dc_g.download(Ipar_dc);

        if (debug) {
            showImg(Iper_dc, "Iper_dc");
            //showImg(Ipar_dc, "Ipar_dc");
        }

        if (debug) {
            float *Iper_dc_ptr = Iper_dc.ptr<float>();
            FILE *fp = fopen("Iper_dc.txt", "w");
            for(int i=0; i<Iper_dc.rows; i++) {
                for(int j=0; j<Iper_dc.cols;j++) {
                    fprintf(fp, "%.6f, ", Iper_dc_ptr[i*Iper_dc.cols+j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }

        float percent = 0.005;
        int rows = Iper.rows;
        int cols = Iper.cols;
        int num_pixels = percent * rows * cols;
        //std::cout << "num_pixels = " << num_pixels << "\n";
        cv::Mat dark_temp = Iper_dc.reshape(1, 1); //一行的矩阵
        cv::Mat Idx;
        cv::sortIdx(dark_temp, Idx, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
        int *Idx_ptr = Idx.ptr<int>();

        if (debug) {
            cv::Mat mask = cv::Mat::zeros(rows, cols, CV_32FC1);
            std::cout << Idx.type() << "\n";

            float* mask_ptr = mask.ptr<float>();
            for(int i=0; i<num_pixels;i++) {
                int row = Idx_ptr[i]/cols;
                int col = Idx_ptr[i]%cols;
                mask_ptr[row*cols+col] = 1;
                std::cout << i << ": " << Idx_ptr[i] << "\n";
            }
            showImg(mask, "mask");
        }

        float *Iper_ptr = Iper.ptr<float>();
        float *Ipar_ptr = Ipar.ptr<float>();
        float Iper_infi_sum[3] = {0};
        float Ipar_infi_sum[3] = {0};
        sum_by_indices(Iper_ptr, Idx_ptr, num_pixels, cols, rows, 3, Iper_infi_sum);
        sum_by_indices(Ipar_ptr, Idx_ptr, num_pixels, cols, rows, 3, Ipar_infi_sum);

        float beta = 1.0f; //1.55f;
        float P[3], Ainfi[3];
        for(int i=0; i<3;i++) {
            P[i] = beta * (Iper_infi_sum[i] - Ipar_infi_sum[i]) / (Iper_infi_sum[i] + Ipar_infi_sum[i]);
            Ainfi[i] = (Iper_infi_sum[i] + Ipar_infi_sum[i]) / num_pixels;
        }


        cv::Mat Iper_vec[3], Ipar_vec[3], Itotal[3];
        cv::split(Iper, Iper_vec);
        cv::split(Ipar, Ipar_vec);
        std::vector<cv::Mat> A_vec(3), t_vec(3), R_vec(3);
        for (int i=0; i<3; i++) {
            A_vec[i] = (Iper_vec[i] - Ipar_vec[i])/P[i];
            Itotal[i] = Iper_vec[i] + Ipar_vec[i];
            t_vec[i] = 1.0 - A_vec[i] / Ainfi[i];
            R_vec[i] = (Itotal[i] - A_vec[i]) / t_vec[i];
            //cv::normalize(R_vec[i], R_vec[i], 1, 0, cv::NORM_MINMAX);
        }
        cv::Mat A, t, R;
        cv::merge(A_vec, A);
        cv::merge(t_vec, t);
        cv::merge(R_vec, R);

        double minVal, maxVal;
        cv::minMaxLoc(A, &minVal, &maxVal);
        std::cout << "A(minmax) is " << minVal << ", " << maxVal << "\n";
        cv::minMaxLoc(t, &minVal, &maxVal);
        std::cout << "t(minmax) is " << minVal << ", " << maxVal << "\n";
        cv::minMaxLoc(R, &minVal, &maxVal);
        std::cout << "R(minmax) is " << minVal << ", " << maxVal << "\n";

        if (debug) {
            std::cout << Iper_infi_sum[0] << ", " << Iper_infi_sum[1] << ", " << Iper_infi_sum[2] << "\n";
            std::cout << Ipar_infi_sum[0] << ", " << Ipar_infi_sum[1] << ", " << Ipar_infi_sum[2] << "\n";
            std::cout << P[0] << ", " << P[1] << ", " << P[2] << "\n";
            std::cout << Ainfi[0] << ", " << Ainfi[1] << ", " << Ainfi[2] << "\n";
            showImg(A, "A");
            showImg(t, "t");
            showImg(R, "R");
        }

        if(image_id == num_images - 1) {

            double minVal, maxVal;
            cv::minMaxLoc(R, &minVal, &maxVal);
            std::cout << minVal << ", " << maxVal << "\n";
            showImg(R, "R");
//            cv::Mat R1;
//            cv::normalize(R, R1, 1, 0, cv::NORM_MINMAX);
//            {
//                std::cout << "R1:\n";
//                float* R1_ptr = R1.ptr<float>();
//                for(int i=0; i<5; i++) {
//                    for(int j=0; j<5;j++) {
//                        int index = i*cols*3+j*3;
//                        std::cout << "(" << R1_ptr[index] << ", " << R1_ptr[index+1] << ", " << R1_ptr[index+2] << ") ";
//                    }
//                    std::cout << "\n";
//                }
//            }
//            R1.convertTo(R1, CV_8UC3, 255.0);
//            cv::imwrite("R.png", R1);
            cv::Mat tmp;
            R.convertTo(tmp, CV_8U, 255., 0.);
            cv::Mat finalImg;
            cv::cvtColor(tmp, finalImg, cv::COLOR_BGRA2BGR);
            cv::imwrite("R.png", finalImg);
        }
    }
    tm.stop();
    std::cout << tm.getTimeSec() << " sec" << std::endl;
    std::cout << static_cast<double>(num_images) / tm.getTimeSec() << " FPS" << std::endl;
}

extern "C" void defog_cuda(float* Iper, float* Ipar, int width, int height,
                float* A, float *t, float *R, float *P, float *Ainfi);
void test_defog_cuda() {
    bool debug = true;


    cv::Mat Iper = cv::imread("F:/wy/ImageWorst_tiff16.tiff", cv::IMREAD_ANYCOLOR|cv::IMREAD_ANYDEPTH);
    cv::Mat Ipar = cv::imread("F:/wy/ImageBest_tiff16.tiff", cv::IMREAD_ANYCOLOR|cv::IMREAD_ANYDEPTH);
    int num_images = 1;
    cv::TickMeter tm;
    tm.start();
    for(int image_id=0; image_id<num_images; image_id++) {
        Iper.convertTo(Iper, CV_32FC3, 1/65535.0);
        Ipar.convertTo(Ipar, CV_32FC3, 1/65535.0);
        cv::cuda::GpuMat Iper_g, Ipar_g;
        Iper_g.upload(Iper);
        Ipar_g.upload(Ipar);

        cv::cuda::GpuMat Iper_dc_g;
        //cv::cuda::GpuMat Ipar_dc_g;
        dark_prior(Iper_g, 12, Iper_dc_g);
        //dark_prior(Ipar_g, 12, Ipar_dc_g);

        cv::Mat Iper_dc;
        Iper_dc_g.download(Iper_dc);
        //Ipar_dc_g.download(Ipar_dc);

        if (debug) {
            showImg(Iper_dc, "Iper_dc");
            //showImg(Ipar_dc, "Ipar_dc");
        }

        if (debug) {
            float *Iper_dc_ptr = Iper_dc.ptr<float>();
            FILE *fp = fopen("Iper_dc.txt", "w");
            for(int i=0; i<Iper_dc.rows; i++) {
                for(int j=0; j<Iper_dc.cols;j++) {
                    fprintf(fp, "%.6f, ", Iper_dc_ptr[i*Iper_dc.cols+j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }

        float percent = 0.005;
        int rows = Iper.rows;
        int cols = Iper.cols;
        int num_pixels = percent * rows * cols;
        //std::cout << "num_pixels = " << num_pixels << "\n";
        cv::Mat dark_temp = Iper_dc.reshape(1, 1); //一行的矩阵
        cv::Mat Idx;
        cv::sortIdx(dark_temp, Idx, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
        int *Idx_ptr = Idx.ptr<int>();

        if (debug) {
            cv::Mat mask = cv::Mat::zeros(rows, cols, CV_32FC1);
            std::cout << Idx.type() << "\n";

            float* mask_ptr = mask.ptr<float>();
            for(int i=0; i<num_pixels;i++) {
                int row = Idx_ptr[i]/cols;
                int col = Idx_ptr[i]%cols;
                mask_ptr[row*cols+col] = 1;
                std::cout << i << ": " << Idx_ptr[i] << "\n";
            }
            showImg(mask, "mask");
        }

        float *Iper_ptr = Iper.ptr<float>();
        float *Ipar_ptr = Ipar.ptr<float>();
        float Iper_infi_sum[3] = {0};
        float Ipar_infi_sum[3] = {0};
        sum_by_indices(Iper_ptr, Idx_ptr, num_pixels, cols, rows, 3, Iper_infi_sum);
        sum_by_indices(Ipar_ptr, Idx_ptr, num_pixels, cols, rows, 3, Ipar_infi_sum);

        float beta = 1.55f;
        float P[3], Ainfi[3];
        for(int i=0; i<3;i++) {
            P[i] = beta * (Iper_infi_sum[i] - Ipar_infi_sum[i]) / (Iper_infi_sum[i] + Ipar_infi_sum[i]);
            Ainfi[i] = (Iper_infi_sum[i] + Ipar_infi_sum[i]) / num_pixels;
        }

        if (false) {
            cv::Mat Iper_vec[3], Ipar_vec[3], Itotal[3];
            cv::split(Iper, Iper_vec);
            cv::split(Ipar, Ipar_vec);
            std::vector<cv::Mat> A_vec(3), t_vec(3), R_vec(3);
            for (int i=0; i<3; i++) {
                A_vec[i] = (Iper_vec[i] - Ipar_vec[i])/P[i];
                Itotal[i] = Iper_vec[i] + Ipar_vec[i];
                t_vec[i] = 1.0 - A_vec[i] / Ainfi[i];
                R_vec[i] = (Itotal[i] - A_vec[i]) / t_vec[i];
            }
            cv::Mat A, t, R;
            cv::merge(A_vec, A);
            cv::merge(t_vec, t);
            cv::merge(R_vec, R);

            if (debug) {
                std::cout << Iper_infi_sum[0] << ", " << Iper_infi_sum[1] << ", " << Iper_infi_sum[2] << "\n";
                std::cout << Ipar_infi_sum[0] << ", " << Ipar_infi_sum[1] << ", " << Ipar_infi_sum[2] << "\n";
                std::cout << P[0] << ", " << P[1] << ", " << P[2] << "\n";
                std::cout << Ainfi[0] << ", " << Ainfi[1] << ", " << Ainfi[2] << "\n";
                showImg(A, "A");
                showImg(t, "t");
                showImg(R, "R");
            }
        }
        else {
            cv::cuda::GpuMat A_g(rows, cols, CV_32FC3);
            cv::cuda::GpuMat t_g(rows, cols, CV_32FC3);
            cv::cuda::GpuMat R_g(rows, cols, CV_32FC3);

            float *Iper_g_data = Iper_g.ptr<float>();
            float *Ipar_g_data = Ipar_g.ptr<float>();
            float *A_g_data = A_g.ptr<float>();
            float *t_g_data = t_g.ptr<float>();
            float *R_g_data = R_g.ptr<float>();
            float *P_g_data, *Ainfi_g_data;
            cudaMalloc((void**)&P_g_data, 3*sizeof(float));
            cudaMalloc((void**)&Ainfi_g_data, 3*sizeof(float));
            cudaMemcpy(P_g_data, P, 3*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(Ainfi_g_data, Ainfi, 3*sizeof(float), cudaMemcpyHostToDevice);
            defog_cuda(Iper_g_data, Ipar_g_data, cols, rows, A_g_data, t_g_data, R_g_data, P_g_data, Ainfi_g_data);
            cudaFree(P_g_data);
            cudaFree(Ainfi_g_data);

            if (debug) {
                cv::Mat A, t, R;
                A_g.download(A);
                t_g.download(t);
                R_g.download(R);
                showImg(A, "A");
                showImg(t, "t");
                showImg(R, "R");
            }

            if(image_id == num_images - 1) {
                cv::Mat R;
                R_g.download(R);
                showImg(R, "R");
    //            cv::Mat R1;
    //            cv::normalize(R, R1, 1, 0, cv::NORM_MINMAX);
    //            {
    //                std::cout << "R1:\n";
    //                float* R1_ptr = R1.ptr<float>();
    //                for(int i=0; i<5; i++) {
    //                    for(int j=0; j<5;j++) {
    //                        int index = i*cols*3+j*3;
    //                        std::cout << "(" << R1_ptr[index] << ", " << R1_ptr[index+1] << ", " << R1_ptr[index+2] << ") ";
    //                    }
    //                    std::cout << "\n";
    //                }
    //            }
    //            R1.convertTo(R1, CV_8UC3, 255.0);
    //            cv::imwrite("R.png", R1);
                cv::Mat tmp;
                R.convertTo(tmp, CV_8U, 255., 0.);
                cv::Mat finalImg;
                cv::cvtColor(tmp, finalImg, cv::COLOR_BGRA2BGR);
                cv::imwrite("R_gpu.png", finalImg);

            }

        }
    }
    tm.stop();
    std::cout << tm.getTimeSec() << " sec" << std::endl;
    std::cout << static_cast<double>(num_images) / tm.getTimeSec() << " FPS" << std::endl;
}



int main(int argc, char** argv){
    float SigmaDebayerTracking =0.5f;
    vector<float> filter = gaussin_filter_1D(SigmaDebayerTracking);

    for (auto value:filter) {
        std::cout << value << std::endl;
    }

    //test_npp_rotate();

    if (false) {//这里是根据一辐大图，对其下采样两倍，然后从中心扣取512x256的图，进行随机平移和旋转变换
        cv::Mat im = cv::imread("E:\\Downloads\\1.jpg", cv::IMREAD_COLOR);
        showImg(im, "original image");

        // 下采样图片
        cv::Mat im_resized;
        cv::resize(im, im_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        showImg(im_resized, "resized image");

        // 提取随机平移与旋转图像
        int H = im_resized.rows;
        int W = im_resized.cols;

        cv::Mat crop1 = cropImg(im_resized, W/2, H/2, 512, 256, 0.0f);
        showImg(crop1, "crop1");

        const int numImgs = 5;
        float low = -5.0f;
        float high = 5.0f;
        float angles[] = {0.0f, 0.0f, 5.0f, 10.0f, -15.0f};
        const int buffer_len = 1024;
        char buffer[buffer_len];
        for(int i = 0; i<numImgs; i++) {
            float randx = static_cast<float>(rand())/static_cast<float>(RAND_MAX) * (high - low) + low;
            float randy = static_cast<float>(rand())/static_cast<float>(RAND_MAX) * (high - low) + low;
            std::cout << randx << ", " << randy << "\n";
            int shiftx = static_cast<int>(std::floorf(randx));
            int shifty = static_cast<int>(std::floorf(randy));
            std::cout << shiftx << ", " << shifty << "\n";

            cv::Mat cropped = cropImg(im_resized, W/2+shiftx, H/2+shifty, 512, 256, angles[i]);
            showImg(cropped, "cropped");
            memset(buffer, '\0', buffer_len);
            sprintf(buffer, "img_%06d.png", i);
            cv::imwrite(buffer, cropped);
        }
    }

    if (false) { // 对一幅图进行sharpen操作
        cv::Mat src = cv::imread("F:\\VkResample\\build_release\\Release\\512_1024_upscaled.png", cv::IMREAD_COLOR);
        showImg(src, "src");

        cv::Mat sharpened = sharpenImg(src);
        showImg(sharpened, "sharpened");
        cv::imwrite("sharpened.png", sharpened);

        cv::Mat sharpened2 = sharpenImg2(src);
        showImg(sharpened2, "sharpened2");
        cv::imwrite("sharpened2.png", sharpened2);
    }

    if (false) {//测试DNN模型的超分辨率
        dnn_sr(argc, argv);
    }

    if (false) {// OpenCV的多帧超分辨率
        cv_mfsr(argc, argv);
    }

    if (false) { // FFT的图像配准
        fft_image_registration(argc, argv);
    }

    if (false) { //测试Eigen
        test_eigen();
    }

    if (true) {// 去雾
        test_defog();
        //test_defog_cuda();
    }

    return 0;
}

























