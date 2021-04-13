
export OPENCV_LIBS="-L${OPENCV_DIR}/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_cudafilters -lopencv_cudaarithm -lopencv_superres"
export CUDA_LIBS="-L${CUDA_PATH}/lib64 -lcuda -lcudart"

nvcc -O3 -m64 -g -G -arch=sm_72 -I$CUDA_PATH/include -I${OPENCV_DIR}/include/opencv4 ${OPENCV_LIBS} \
${CUDA_LIBS} myKernels.cu polar_defog.cpp -o polar_defog

nvcc -O3 -m64 -g -G -arch=sm_72 -I$CUDA_PATH/include -I${OPENCV_DIR}/include/opencv4 ${OPENCV_LIBS} \
${CUDA_LIBS} multi_frame_sr.cpp -o multi_frame_sr
