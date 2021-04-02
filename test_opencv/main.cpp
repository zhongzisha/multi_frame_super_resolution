#include <iostream>
#include <cstring>
using namespace std;

#include <opencv2/opencv.hpp>

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



int main(int argc, char** argv) {
    // std::string filename = std::string(argv[1]);
    // dark_channel_prior_defog(filename);
    dark_channel_prior_defog_for_polar();
}




















