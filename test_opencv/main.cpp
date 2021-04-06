#include <iostream>
#include <cstring>
using namespace std;

#include <opencv2/opencv.hpp>

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

    nAngle = atof("90");

    oSrcROI.x = oSrcROI.y = 0;
    oSrcROI.width = oSrcSize.width;
    oSrcROI.height = oSrcSize.height;

    // 分配显存
    pSrcDataDevice = nppiMalloc_8u_C3(oSrcSize.width * nImgBpp, oSrcSize.height, &nSrcPitchDevice);
    cudaMemcpy2D(pSrcDataDevice, nSrcPitchDevice, pSrcData, nSrcPitch,
                 oSrcSize.width * nImgBpp, oSrcSize.height, cudaMemcpyHostToDevice);

    nppiGetRotateBound(oSrcROI, aBoundingBox, nAngle, 0, 0);
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

int main(){
    float SigmaDebayerTracking =0.5f;
    vector<float> filter = gaussin_filter_1D(SigmaDebayerTracking);

    for (auto value:filter) {
        cout << value << endl;
    }

    test_npp_rotate();

    return 0;
}




















