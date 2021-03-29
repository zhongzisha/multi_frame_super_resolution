TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp


INCLUDEPATH += "C:/msys64/mingw64/include/opencv4"
LIBS += "C:/msys64/mingw64/lib/libopencv_core.dll.a"
LIBS += "C:/msys64/mingw64/lib/libopencv_imgcodecs.dll.a"
LIBS += "C:/msys64/mingw64/lib/libopencv_imgproc.dll.a"
LIBS += "C:/msys64/mingw64/lib/libopencv_highgui.dll.a"
