TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
        multi_frame_sr.cpp \
        polar_defog.cpp

DISTFILES += \
    compile.sh \
    myKernels.cu \
    runall.sh


INCLUDEPATH += "F:/softwares/install/include/"
INCLUDEPATH += "F:/softwares/install/include/eigen3"
INCLUDEPATH += "F:/opencv-4.5.1/build_release/install/include"
INCLUDEPATH += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_core451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_imgcodecs451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_highgui451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_imgproc451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_world451.lib"
