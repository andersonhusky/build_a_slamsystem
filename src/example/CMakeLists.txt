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

add_executable(example-app example-app.cpp)
target_link_libraries(example-app ${OpenCV_LIBRARIES})
target_link_libraries(example-app ${TORCH_LIBRARIES})
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
