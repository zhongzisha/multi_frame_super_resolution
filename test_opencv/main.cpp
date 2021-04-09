#include <iostream>
#include <cstring>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/superres.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

/*
// rect is the RotatedRect (I got it from a contour...)
        RotatedRect rect;
        // matrices we'll use
        Mat M, rotated, cropped;
        // get angle and size from the bounding box
        float angle = rect.angle;
        Size rect_size = rect.size;
        // thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
        if (rect.angle < -45.) {
            angle += 90.0;
            swap(rect_size.width, rect_size.height);
        }
        // get the rotation matrix
        M = getRotationMatrix2D(rect.center, angle, 1.0);
        // perform the affine transformation
        warpAffine(src, rotated, M, src.size(), INTER_CUBIC);
        // crop the resulting image
        getRectSubPix(rotated, rect_size, rect.center, cropped);
*/

void showImg(const cv::Mat& mat, const std::string& title) {
    cv::namedWindow(title);
    cv::imshow(title, mat);
    cv::waitKey();
}

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

void dnn_sr(int argc, char** argv) {
    // 使用OpenCV自带的超分辨算法进行超分辨率
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
        frames_[index_++].copyTo(_frame.getGpuMatRef());
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
    superRes->setIterations(1);
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

    if (true) {
        cv_mfsr(argc, argv);
    }

    return 0;
}




















