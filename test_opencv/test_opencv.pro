TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp


INCLUDEPATH += "F:/softwares/install/include/"
INCLUDEPATH += "F:/softwares/install/include/eigen3"
INCLUDEPATH += "F:/opencv-4.5.1/build_release/install/include"
INCLUDEPATH += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_core451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_imgcodecs451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_highgui451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_imgproc451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_world451.lib"

## This makes the .cu files appear in your project
#CUDA_SOURCES +=  \
#DeBayerKernels.cu \
#RobustnessModell.cu \
#ShiftMinimizerKernels.cu \
#kernel.cu \
#opticalFlow.cu

## Project dir and outputs
#OBJECTS_DIR = $$PWD
#DESTDIR = $$PWD

## Path to cuda toolkit install
#CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2"
## GPU architecture
#CUDA_ARCH = sm_75
## nvcc flags (ptxas option verbose is always useful)
#NVCCFLAGS = --compiler-options -use-fast-math --Wno-deprecated-gpu-targets
## include paths
#INCLUDEPATH += $$CUDA_DIR/include
## lib dirs
#QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
## libs - note than i'm using a x_86_64 machine
#LIBS += -lcuda -lcudart
## join the includes in a line
#CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

#cuda.input = CUDA_SOURCES
#cuda.output = ${OBJECTS_DIR}/${QMAKE_FILE_BASE}_cuda.o
#cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#cuda.dependency_type = TYPE_C
#cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
## Tell Qt that we want add more stuff to the Makefile
#QMAKE_EXTRA_UNIX_COMPILERS += cuda


# This makes the .cu files appear in your project
OTHER_FILES +=  \
DeBayerKernels.cu \
RobustnessModell.cu \
ShiftMinimizerKernels.cu \
kernel.cu \
opticalFlow.cu

DISTFILES += \
    build.bat \
    myKernels.cu




