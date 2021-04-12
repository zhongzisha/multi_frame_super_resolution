#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cfloat>

#include <cufft.h>

#define PI   3.1415926535897932384626433832795
#define PI_2 (-PI/2.0)
#define EPS (1e-15)


extern "C"
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


// Helper function for using CUDA to add vectors in parallel.
extern "C"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}


//将rs复制到cs的实部
extern "C"
__global__ void copy_R2C_kernel(float* rs, cufftDoubleComplex* cs, int N) {
    int index=threadIdx.x + blockIdx.x*blockDim.x;
    if (index < N) {
        cs[index].x = rs[index];
        cs[index].y = 0;
    }
}

extern "C"
void copy_R2C(float* rs, cufftDoubleComplex* cs, int N) {
    // Launch a kernel on the GPU with one thread for each element.
    int num_threads = 512;
    int num_blocks = (N+num_threads-1)/num_threads;
    copy_R2C_kernel<<<num_blocks, num_threads>>>(rs, cs, N);
}


__global__ void fftshift_2D_kernel(cufftDoubleComplex *IM, int imW, int imH)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = idy*imW + idx;
    int x, y, indshift;
    cufftDoubleComplex v;


    if(idx < imW && idy < imH/2)
    {
        if(idx<imW/2 && idy<imH/2)
        {
            x=idx+imW/2;
            y=idy+imH/2;
        }
        else if(idx>=imW/2 && idy<imH/2)
        {
            x=idx-imW/2;
            y=idy+imH/2;
        }

        indshift = y*imW+x;
        v.x = IM[ind].x;
        v.y = IM[ind].y;

        IM[ind].x = IM[indshift].x;
        IM[ind].y = IM[indshift].y;

        IM[indshift].x = v.x;
        IM[indshift].y = v.y;
    }
}
extern "C"
void fftshift_2D(cufftDoubleComplex* cs, int width, int height) {
    // Launch a kernel on the GPU with one thread for each element.
    dim3 blockSize(32, 32, 1);
    dim3 gridSize;
    gridSize.x = (width + blockSize.x - 1) / blockSize.x;
    gridSize.y = (height + blockSize.y - 1) / blockSize.y;
    gridSize.z = 1;
    fftshift_2D_kernel<<<gridSize, blockSize>>>(cs, width, height);
}

__global__ void high_pass_filtering_kernel(cufftDoubleComplex* input, float *output, int width, int height,
                                           double col_step, double row_step) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = idy*width + idx;
    double t1, t2;
    cufftDoubleComplex v;
    if(idx < width && idy < height)
    {
        t1 = idy*row_step+PI_2;
        t1 *= t1;
        t2 = idx*col_step+PI_2;
        t2 *= t2;
        t1 = cos(sqrt(t1+t2));
        t1 *= t1;
        t1 = 1.0 - t1;

        v.x = input[ind].x * t1;
        v.y = input[ind].y * t1;

        output[ind] = static_cast<float>(sqrt(v.x*v.x + v.y*v.y));
    }
}
extern "C"
void high_pass_filtering(cufftDoubleComplex* input, float *output, int width, int height) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize;
    gridSize.x = (width + blockSize.x - 1) / blockSize.x;
    gridSize.y = (height + blockSize.y - 1) / blockSize.y;
    gridSize.z = 1;
    double col_step = PI / static_cast<double>(width-1);
    double row_step = PI / static_cast<double>(height-1);
    high_pass_filtering_kernel<<<gridSize, blockSize>>>(input, output, width, height, col_step, row_step);
    //这原来是搞个double steps[2]的，cuda kernel不能传数组？
}

__global__ void crossPowerSpectrum_kernel(cufftDoubleComplex* f1, cufftDoubleComplex* f2, int width, int height) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = idy*width + idx;
    double denorm, t1, t2;
    if(idx < width && idy < height)
    {
        denorm = sqrt(f1[ind].x*f1[ind].x+f1[ind].y*f1[ind].y) * sqrt(f2[ind].x*f2[ind].x+f2[ind].y*f2[ind].y) + EPS;
        t1 = (f1[ind].x * f2[ind].x + f1[ind].y * f2[ind].y) / denorm;
        t2 = (f1[ind].y * f2[ind].x - f1[ind].x * f2[ind].y) / denorm;
        f1[ind].x = t1;
        f1[ind].y = t2;
    }
}
extern "C"
void crossPowerSpectrum(cufftDoubleComplex* f1, cufftDoubleComplex* f2, int width, int height) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize;
    gridSize.x = (width + blockSize.x - 1) / blockSize.x;
    gridSize.y = (height + blockSize.y - 1) / blockSize.y;
    gridSize.z = 1;
    crossPowerSpectrum_kernel<<<gridSize, blockSize>>>(f1, f2, width, height);
}

__global__ void abs_and_normby_kernel(cufftDoubleComplex* f1, float* output, double normby, int width, int height) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int ind = idy*width + idx;
    if(idx < width && idy < height)
    {
        f1[ind].x /= normby;
        f1[ind].y /= normby;
        output[ind] = sqrt(f1[ind].x*f1[ind].x+f1[ind].y*f1[ind].y);
    }
}
extern "C"
void abs_and_normby(cufftDoubleComplex* f1, float* output, double normby, int width, int height) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize;
    gridSize.x = (width + blockSize.x - 1) / blockSize.x;
    gridSize.y = (height + blockSize.y - 1) / blockSize.y;
    gridSize.z = 1;
    abs_and_normby_kernel<<<gridSize, blockSize>>>(f1, output, normby, width, height);
}

__global__ void defog_cuda_kernel(float* Iper, float* Ipar, int width, int height,
                                  float* A, float *t, float *R, float *P, float *Ainfi) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int index;
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
void defog_cuda(float* Iper, float* Ipar, int width, int height,
                float* A, float *t, float *R,
                float *P, float *Ainfi) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize;
    gridSize.x = (width + blockSize.x - 1) / blockSize.x;
    gridSize.y = (height + blockSize.y - 1) / blockSize.y;
    gridSize.z = 1;
    defog_cuda_kernel<<<gridSize, blockSize>>>(Iper, Ipar, width, height, A, t, R, P, Ainfi);
}

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


