set NVCCFLAGS=--compiler-options --use-fast-math --Wno-deprecated-gpu-targets

rem nvcc %NVCCFLAGS% -I"%CUDA_PATH%/include" -m64 -g -G -arch=sm_75 -c -o kernel.obj kernel.cu
nvcc %NVCCFLAGS% -I"%CUDA_PATH%/include" -m64 -g -G -arch=sm_75 -c -o myKernels.obj myKernels.cu

set OPENCV_LIBS=%OPENCV_PATH%/x64/vc16/lib/opencv_core451.lib ^
%OPENCV_PATH%/x64/vc16/lib/opencv_imgcodecs451.lib ^
%OPENCV_PATH%/x64/vc16/lib/opencv_highgui451.lib ^
%OPENCV_PATH%/x64/vc16/lib/opencv_imgproc451.lib ^
%OPENCV_PATH%/x64/vc16/lib/opencv_world451.lib

set CUDA_LIBS="%CUDA_PATH%/lib/x64/cuda.lib" "%CUDA_PATH%/lib/x64/cudart.lib" "%CUDA_PATH%/lib/x64/cufft.lib"

set COMMON_INSTALL="F:/softwares/install/"

set NPP_LIBS="%CUDA_PATH%/lib/x64/nppc.lib" "%CUDA_PATH%/lib/x64/nppial.lib" "%CUDA_PATH%/lib/x64/nppicc.lib" "%CUDA_PATH%/lib/x64/nppidei.lib" "%CUDA_PATH%/lib/x64/nppif.lib" "%CUDA_PATH%/lib/x64/nppig.lib" "%CUDA_PATH%/lib/x64/nppim.lib" "%CUDA_PATH%/lib/x64/nppist.lib" "%CUDA_PATH%/lib/x64/nppisu.lib" "%CUDA_PATH%/lib/x64/nppitc.lib" "%CUDA_PATH%/lib/x64/npps.lib"

cl /std:c17 -I "%COMMON_INSTALL%/include" ^
-I "%COMMON_INSTALL%/include/eigen3" ^
-I "%OPENCV_PATH%/include" ^
-I "%CUDA_PATH%/include" ^
main.cpp /link myKernels.obj ^
%OPENCV_LIBS% %CUDA_LIBS% %NPP_LIBS%
