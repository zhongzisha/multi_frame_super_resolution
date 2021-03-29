/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>


typedef unsigned short ushort;


bool printfNPPinfo(int argc, char *argv[]) {
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}


std::vector<float> gaussin_filter_1D(float sigma) {
    if (sigma <= 0) {
        return std::vector<float>({ 0, 0,0,0,1,0,0,0,0 });
    }
    int size = (int)(sigma / 0.6f - 0.4f) * 2 + 1 + 2;
    size = std::min(size, 99);
    std::vector<float> ret(size);

    const int center = size / 2;
    for (int i = 0; i < size; i++) {
        int x = i - center;
        ret[i] = (float)(exp(-(x * x) / (2 * sigma * sigma)));
    }
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += ret[i];
    }
    for (int i = 0; i < size; i++) {
        ret[i] /= sum;
    }
    return ret;
}


class PreAlignment {
public:
    Npp32f* d_inputImg;
    int inputImg_width;
    int inputImg_height;
    int inputImg_channels;
    int inputImg_step;
    int inputImg_pitch;

    Npp32f* d_imgToTrackRotated;
    float2* d_imgToTrackCplx;
    float2* d_imgRefCplx;
    byte* d_buffer;
    int* d_x;
    int* d_y;
    float* d_val;

    cufftHandle* d_plan;
    byte* d_bufferFFT;
    
    int width;
    int height;
    size_t FFTBufferSize;
    float highPass;
    bool memoryAllocated = false;

    void AllocateDeviceMemory() {
        int fftWidth = width / 2 + 1;
        cudaMalloc(&d_imgToTrackCplx, sizeof(float2) * fftWidth * height);
        cudaMalloc(&d_imgRefCplx, sizeof(float2) * fftWidth * height);
        cudaMalloc(&d_x, sizeof(int));
        cudaMalloc(&d_y, sizeof(int));
        cudaMalloc(&d_val, sizeof(float));

        cudaMalloc(&d_bufferFFT, sizeof(byte) * FFTBufferSize);

        int maxBufferSize;
        NppiSize oSizeROI = { inputImg_width, inputImg_height };
        nppiMaxIndxGetBufferHostSize_32f_C1R(oSizeROI, &maxBufferSize);
        int maxBufferSize2;
        nppiMinMaxGetBufferHostSize_32f_C1R(oSizeROI, &maxBufferSize2);
        maxBufferSize = MAX(maxBufferSize, maxBufferSize2);
        cudaMalloc(&d_buffer, sizeof(byte) * maxBufferSize);

        memoryAllocated = true;
    }

    PreAlignment(Npp32f* inputImg_,
        int inputImg_width_,
        int inputImg_height_,
        int inputImg_channels_,
        int inputImg_step_,
        int inputImg_pitch_) {
        d_inputImg = inputImg_;
        inputImg_width = inputImg_width_;
        inputImg_height = inputImg_height_;
        inputImg_channels = inputImg_channels_;
        inputImg_step = inputImg_step_;
        inputImg_pitch = inputImg_pitch_;
        

    }

    
};


int main(int argc, char *argv[]) {
  printf("%s Starting...\n\n", argv[0]);

  try {
    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false) {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    } else {
      filePath = sdkFindFilePath("Lena.pgm", argv[0]);
    }

    if (filePath) {
      sFilename = filePath;
    } else {
      sFilename = "Lena.pgm";
    }

    // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good()) {
      std::cout << "boxFilterNPP opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    } else {
      std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">"
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0) {
      exit(EXIT_FAILURE);
    }

    std::string sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename.rfind('.');

    if (dot != std::string::npos) {
      sResultFilename = sResultFilename.substr(0, dot);
    }

    sResultFilename += "_boxFilter.png";

    if (checkCmdLineFlag(argc, (const char **)argv, "output")) {
      char *outputFilePath;
      getCmdLineArgumentString(argc, (const char **)argv, "output",
                               &outputFilePath);
      sResultFilename = outputFilePath;
    }


    //∂¡»°Õº∆¨
    int height = 1024;
    int width = 1024;
    int nchannels = 3;
    unsigned char* pSrcData = new unsigned char[height * width * nchannels];
    unsigned char* pDstData = nullptr;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < nchannels; k++) {
                pSrcData[i * width * nchannels + j * nchannels + k] =
                    static_cast<unsigned char>((static_cast<double>(rand())/ RAND_MAX) * 255);
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

    // ∑÷≈‰œ‘¥Ê
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
    
    for (int i = 0; i < oDstSize.height; i++) {
        for (int j = 0; j < oDstSize.width; j++) {
            std::cout << i << ", " << j << ": ";
            for (int k = 0; k < nchannels; k++) {
                    std::cout << static_cast<int>(pSrcData[i * oDstSize.width * nchannels + j * nchannels + k]) << ", ";
            }
            std::cout << std::endl;
        }
    }
    
    
    delete[] pSrcData;
    delete[] pDstData;

    nppiFree(pSrcDataDevice);
    nppiFree(pDstDataDevice);

    cudaDeviceReset();

    exit(EXIT_SUCCESS);
  } catch (npp::Exception &rException) {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  } catch (...) {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
