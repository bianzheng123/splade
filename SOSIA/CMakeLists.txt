cmake_minimum_required(VERSION 3.18)

#set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_C_COMPILER "gcc")

project(SOSIA LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Green "${Esc}[32m")
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message("${Green}Release mode${ColourReset}")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message("${Green}Clang${ColourReset}")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message("${Green}GNU${ColourReset}")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message("${Green}MSVC${ColourReset}")
    else ()
        message("${Green}Unknown compiler${ColourReset}")
    endif ()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -DNDEBUG -openmp -march=native -fpic -w -ftree-vectorize -ftree-vectorizer-verbose=0 ${OpenMP_CXX_FLAGS}")
    add_definitions(-Ofast -lrt -DNDEBUG -openmp -march=native -fpic -w -ftree-vectorizer-verbose=0 -ftree-vectorize ${OpenMP_CXX_FLAGS})
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -fsanitize=address -lasan ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native ${OpenMP_CXX_FLAGS}")
    #    add_definitions(-DNDEBUG -march=native)
    message("${Green}Debug mode${ColourReset}")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native ${OpenMP_CXX_FLAGS}")
    message("${Green}Unknown mode${ColourReset}")
endif ()

#find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#
#find_package(Threads REQUIRED)
#
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
#find_package(OpenBLAS REQUIRED)
#
#find_package(spdlog REQUIRED)

find_package(OpenMP REQUIRED)

#pybind11_add_module(SimpleNRATopKRankingItem src/app/insight/SimpleNRATopKRankingItem.cpp)
#target_link_libraries(SimpleNRATopKRankingItem PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen spdlog::spdlog ${OpenBLAS_LIB})
#target_include_directories(SimpleNRATopKRankingItem PRIVATE ${OpenBLAS_INCLUDE_DIR})

set(EXE src/alg.cpp src/basis.cpp src/Preprocess.cpp)

add_executable(sos src/main.cpp ${EXE})
target_link_libraries(sos PRIVATE OpenMP::OpenMP_CXX)


#add_executable(t1 test.cpp)
#target_link_libraries(t1 PRIVATE spdlog::spdlog)

