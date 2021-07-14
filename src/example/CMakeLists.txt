cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

set(CMAKE_PREFIX_PATH=/home/hongfeng/libtorch)
find_package(Torch REQUIRED)
find_package(OpenCV 3.2 REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(/usr/local/cuda-11.1/targets/x86_64-linux/include)
include_directories(/home/hongfeng/CV/build_a_slam_bymyself/src/example)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app ${OpenCV_LIBRARIES})
target_link_libraries(example-app ${TORCH_LIBRARIES})
target_link_libraries(example-app Construct ${catkin_LIBRARIES})
target_link_libraries(example-app ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so)
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

add_library(Construct Construct.cpp)
