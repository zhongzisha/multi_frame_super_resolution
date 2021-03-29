TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp


INCLUDEPATH += "F:/opencv-4.5.1/build_release/install/include"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_core451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_imgcodecs451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_highgui451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_imgproc451.lib"
LIBS += "F:/opencv-4.5.1/build_release/install/x64/vc16/lib/opencv_ccalib451.lib"
