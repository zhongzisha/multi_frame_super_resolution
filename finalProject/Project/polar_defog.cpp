
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


extern "C"
void defog_cuda2(float* Iper, float* Ipar, int width, int height,
                float* A, float *t, float *R,
                float P0, float P1, float P2,
                 float A0, float A1, float A2);


//显示图片
void showImg(const cv::Mat& mat, const std::string& title) {
    cv::namedWindow(title);
    cv::imshow(title, mat);
    cv::waitKey();
}

void sum_by_indices2(float *in1, float *in2, int *indices, int size, int width, int height, int channels,
                    float *out1, float *out2) {
    int row, col;
    for(int index=0;index<size;index++) {
        row = indices[index]/width;
        col = indices[index]%width;
        for(int c=0;c<channels;c++){
            out1[c] += in1[row*width+col*channels+c];
            out2[c] += in2[row*width+col*channels+c];
        }
    }
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

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("./polar_defog debug inputType beta\n");
        printf("\tdebug: 0 or 1\n");
        printf("\tinputType: 1 or 2\n");
        printf("\tbeta: 1.55 for 1 and 10 for 2, need to adjust\n");
        return -1;
    }

    bool debug = static_cast<bool>(atoi(argv[1]));
    int inputType = atoi(argv[2]);
    float beta = atof(argv[3]);

    cv::Mat Iper_orig, Ipar_orig;
    switch(inputType) {
    case 1:{
        Iper_orig = cv::imread("ImageWorst_tiff16.tiff", cv::IMREAD_ANYCOLOR|cv::IMREAD_ANYDEPTH);
        Ipar_orig = cv::imread("ImageBest_tiff16.tiff", cv::IMREAD_ANYCOLOR|cv::IMREAD_ANYDEPTH);
    };
        break;
    case 2:{
        cv::Mat I0 = cv::imread("degree0.tiff", cv::IMREAD_GRAYSCALE);
        cv::Mat I45 = cv::imread("degree45.tiff", cv::IMREAD_GRAYSCALE);
        cv::Mat I90 = cv::imread("degree90.tiff", cv::IMREAD_GRAYSCALE);
        I0.convertTo(I0, CV_32FC1, 1/255.0);
        I45.convertTo(I45, CV_32FC1, 1/255.0);
        I90.convertTo(I90, CV_32FC1, 1/255.0);
        cv::Mat I135 = I0 + I90 - I45;

        if (debug) {

            std::vector<cv::Mat> inputs = {I0, I45, I90, I135};
            for(int i=0;i<inputs.size();i++){
                std::cout << inputs[i].size << "\n";
            }
            cv::Mat Ishown;
            cv::hconcat(inputs, Ishown);
            showImg(Ishown, "Ishown");
        }

        cv::Mat S0 = I0+I90;
        cv::Mat S1 = I0-I90;
        cv::Mat S2 = I45 - I135;
        cv::multiply(S1, S1, S1);
        cv::multiply(S2, S2, S2);
        cv::sqrt(S1 + S2, S1);
        cv::Mat D = S1 / (S0 + 1e-15);
        cv::Mat Iper1, Ipar1;
        cv::multiply(1.0f + D, S0/2, Iper1);
        cv::multiply(1.0f - D, S0/2, Ipar1);
        cv::normalize(Iper1, Iper1, 1, 0, cv::NORM_MINMAX);
        cv::normalize(Ipar1, Ipar1, 1, 0, cv::NORM_MINMAX);
        Iper1.convertTo(Iper1, CV_32FC1, 65535.0f);
        Ipar1.convertTo(Ipar1, CV_32FC1, 65535.0f);
        std::vector<cv::Mat> Ipers = {Iper1, Iper1, Iper1};
        std::vector<cv::Mat> Ipars = {Ipar1, Ipar1, Ipar1};
        cv::merge(Ipers, Iper_orig);
        cv::merge(Ipars, Ipar_orig);
    };
        break;
    }

    int warmupStep=32;
    int real_num_images = 256;

    if (debug) {
        warmupStep= 0;
        real_num_images=1;
    }

    int num_images = warmupStep+real_num_images;

    cv::TickMeter tm;

    for(int image_id=0; image_id<num_images; image_id++) {
        if(image_id==warmupStep) {
            tm.start();
        }

        cv::Mat Iper, Ipar;

        Iper_orig.convertTo(Iper, CV_32FC3, 1/65535.0f);
        Ipar_orig.convertTo(Ipar, CV_32FC3, 1/65535.0f);
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
        int widthStep = Iper.step;
        int step1 = Iper.step1();
        int num_pixels = percent * rows * cols;

        cv::Mat dark_temp = Iper_dc.reshape(1, 1); //一行的矩阵
        cv::Mat Idx;
        cv::sortIdx(dark_temp, Idx, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
        int *Idx_ptr = Idx.ptr<int>();

        if (debug) {
            std::cout << "num_pixels = " << num_pixels << "\n";
            std::cout << "height = " << rows << ", width = " << cols << ", widthStep is " << widthStep << ", step1 = " << step1 << "\n";

            cv::Mat mask = cv::Mat::zeros(rows, cols, CV_32FC1);
            std::cout << Idx.type() << "\n";

            float* mask_ptr = mask.ptr<float>();
            for(int i=0; i<num_pixels;i++) {
                int row = Idx_ptr[i]/cols;
                int col = Idx_ptr[i]%cols;
                mask_ptr[row*cols+col] = 1;
                // std::cout << i << ": " << Idx_ptr[i] << "\n";
            }
            showImg(mask, "mask");
        }

        float *Iper_ptr = Iper.ptr<float>();
        float *Ipar_ptr = Ipar.ptr<float>();
        float Iper_infi_sum[3] = {0};
        float Ipar_infi_sum[3] = {0};
        //sum_by_indices(Iper_ptr, Idx_ptr, num_pixels, cols, rows, 3, Iper_infi_sum);
        //sum_by_indices(Ipar_ptr, Idx_ptr, num_pixels, cols, rows, 3, Ipar_infi_sum);
        sum_by_indices2(Iper_ptr, Ipar_ptr, Idx_ptr, num_pixels, cols, rows, 3, Iper_infi_sum, Ipar_infi_sum);

        //float beta = 1.55f;
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
            defog_cuda2(Iper_g_data, Ipar_g_data, A_g.step1()/3, rows, A_g_data, t_g_data, R_g_data,
                        P[0], P[1], P[2], Ainfi[0], Ainfi[1], Ainfi[2]);

            if (debug) {
                cv::Mat A, t, R;
                A_g.download(A);
                t_g.download(t);
                R_g.download(R);
                showImg(A, "A");
                showImg(t, "t");
                showImg(R, "R");

                printf("A_g: %d, %d, %d, %d\n", A_g.rows, A_g.cols, A_g.step, A_g.step1());
                printf("A: %d, %d, %d, %d\n", A.rows, A.cols, A.step, A.step1());
                printf("t_g: %d, %d, %d, %d\n", t_g.rows, t_g.cols, t_g.step, t_g.step1());
                printf("t: %d, %d, %d, %d\n", t.rows, t.cols, t.step, t.step1());
                printf("R_g: %d, %d, %d, %d\n", R_g.rows, R_g.cols, R_g.step, R_g.step1());
                printf("R: %d, %d, %d, %d\n", R.rows, R.cols, R.step, R.step1());

                double minVal, maxVal;
                cv::minMaxLoc(A, &minVal, &maxVal);
                std::cout << "A(minmax) is " << minVal << ", " << maxVal << "\n";
                cv::minMaxLoc(t, &minVal, &maxVal);
                std::cout << "t(minmax) is " << minVal << ", " << maxVal << "\n";
                cv::minMaxLoc(R, &minVal, &maxVal);
                std::cout << "R(minmax) is " << minVal << ", " << maxVal << "\n";

                if(image_id == num_images - 1) {
                    cv::Mat R;
                    R_g.download(R);
                    showImg(R, "R");
                    cv::Mat tmp;
                    R.convertTo(tmp, CV_8U, 255., 0.);
                    cv::Mat finalImg;
                    cv::cvtColor(tmp, finalImg, cv::COLOR_BGRA2BGR);
                    cv::imwrite("R_gpu.png", finalImg);
                }
            }

        }
    }
    tm.stop();
    std::cout << tm.getTimeSec() << " sec" << std::endl;
    std::cout << static_cast<double>(real_num_images) / tm.getTimeSec() << " FPS" << std::endl;

    return 0;
}
