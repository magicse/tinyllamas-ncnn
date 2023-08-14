cmake_minimum_required(VERSION 3.14)
project(tinyllamas-ncnn)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

Set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(ncnn
    GIT_REPOSITORY https://github.com/Tencent/ncnn
    GIT_PROGRESS TRUE GIT_SHALLOW TRUE GIT_TAG 20230517
    USES_TERMINAL_DOWNLOAD TRUE)
FetchContent_MakeAvailable(ncnn)

add_executable(tinyllamas tinyllamas.cpp)
target_link_libraries(tinyllamas ncnn)
