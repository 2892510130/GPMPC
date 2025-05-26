cmake_minimum_required(VERSION 3.26)

project(GPMPC)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Eigen3 REQUIRED)

file(GLOB_RECURSE MODEL_SOURCE_FILE 
     "src/model/*.cpp"
)

add_library(ModelLibrary
    ${MODEL_SOURCE_FILE} 
)

target_include_directories(ModelLibrary PUBLIC include)

target_link_libraries (ModelLibrary Eigen3::Eigen)

# Add test exes
add_executable(env_test src/test/env_test.cpp)
target_link_libraries(env_test Eigen3::Eigen)

add_executable(model_test src/test/model_test.cpp)
target_link_libraries(model_test ModelLibrary)

