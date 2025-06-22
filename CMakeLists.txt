cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0012 NEW)
project(anime_project)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(anime_test main.cpp CustomDatasetTorch.cpp)
target_link_libraries(anime_test "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
set_property(TARGET anime_test PROPERTY CXX_STANDARD 17) 