

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cfloat>

#include <cufft.h>

#define PI   3.1415926535897932384626433832795
#define PI_2 (-PI/2.0)
#define EPS (1e-15)


__global__ void defog_cuda_kernel2(float* Iper, float* Ipar, int width, int height,
                                  float* A, float *t, float *R,
                                   float P0, float P1, float P2,
                                   float A0, float A1, float A2) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int index;
    float P[3] = {P0, P1, P2};
    float Ainfi[3] = {A0, A1, A2};
    if(idx < width && idy < height)
    {
        index = idy*width*3+idx*3;
        for (int i=0; i<3; i++) {
//            A_vec[i] = (Iper_vec[i] - Ipar_vec[i])/P[i];
//            Itotal[i] = Iper_vec[i] + Ipar_vec[i];
//            t_vec[i] = 1.0 - A_vec[i] / Ainfi[i];
//            R_vec[i] = (Itotal[i] - A_vec[i]) / t_vec[i];
            A[index+i] = (Iper[index+i] - Ipar[index+i])/P[i];
            t[index+i] = 1.0f - A[index+i] / Ainfi[i];
            if (t[index+i] < 0.001f) {
                t[index+i] = 0.001f;
            }
            if (t[index+i] > 0.999f) {
                t[index+i] = 0.999f;
            }

            R[index+i] = (Iper[index+i] + Ipar[index+i] - A[index+i])/t[index+i];
            if (R[index+i] < 0.001f) {
                R[index+i] = 0.001f;
            }
            if (R[index+i] > 0.999f) {
                R[index+i] = 0.999f;
            }

        }
    }
}

extern "C"
void defog_cuda2(float* Iper, float* Ipar, int width, int height,
                float* A, float *t, float *R,
                float P0, float P1, float P2,
                 float A0, float A1, float A2) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize;
    gridSize.x = (width + blockSize.x - 1) / blockSize.x;
    gridSize.y = (height + blockSize.y - 1) / blockSize.y;
    gridSize.z = 1;
    defog_cuda_kernel2<<<gridSize, blockSize>>>(Iper, Ipar, width, height, A, t, R, P0, P1, P2, A0, A1, A2);
}
