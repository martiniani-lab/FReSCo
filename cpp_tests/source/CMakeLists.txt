cmake_minimum_required(VERSION 3.19.1)

project(hyperalg)

# if not specified by user, the standard build type is release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release Coverage." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

enable_language(CXX)

#set OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -g -Wall -ansi -pedantic -std=c++17")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
if(COMPILER_SUPPORTS_AVX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif(COMPILER_SUPPORTS_AVX)

# Add and compile the gtest library
set(gtest_src ../gtest)
set(gtest_binary_dir googletest)
set(gtest_include ${gtest_src}/include/)
include_directories(${gtest_include})
message("gtest include directory ${gtest_include}")
add_subdirectory(../gtest ${gtest_binary_dir})
set(gtest_libs gtest gtest_main)
message("gtest_libs ${gtest_libs}")

# set the hyperalg include directory
set(hyperalg_include ../../source)
include_directories(${hyperalg_include})
message("hyperalg include directory ${hyperalg_include}")

## build the hyperalg library
#FILE(GLOB hyperalg_sources ${hyperalg_include}/*.c*)
#add_library(hyperalg_lib ${hyperalg_sources})
## add hyperalg_lib to target_link_libraries

# get all the source files
FILE(GLOB sources *.c*)
message("test files ${sources}")

add_executable(test_main ${sources})

target_link_libraries(test_main gtest gtest_main pthread)
