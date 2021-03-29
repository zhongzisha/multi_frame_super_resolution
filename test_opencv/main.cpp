#include <iostream>
#include <cstring>
using namespace std;

#include <opencv2/opencv.hpp>

int main()
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

    return 0;
}
