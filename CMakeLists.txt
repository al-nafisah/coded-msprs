cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0054 NEW)

project(msprs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

set(EXECUTABLE_OUTPUT_PATH bin/)

# Data paths resolve against this, so a binary works from any directory.
get_filename_component(MSPRS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(cpptrace CONFIG 0.4.1)
find_package(AFF3CT CONFIG 4.1.1 REQUIRED)

add_library(msprs_core INTERFACE)
target_include_directories(msprs_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(msprs_core INTERFACE aff3ct::aff3ct-static-lib Threads::Threads)
target_compile_definitions(msprs_core INTERFACE MSPRS_REPO_ROOT="${MSPRS_REPO_ROOT}")
