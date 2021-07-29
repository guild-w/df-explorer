cmake_minimum_required(VERSION 3.10)

project(df-explorer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_LIST_DIR}/thirdparty)
include_directories(${CMAKE_CURRENT_LIST_DIR}/thirdparty/CGBN/include)

find_package(workflow CONFIG HINTS thirdparty/workflow REQUIRED)
include_directories(${WORKFLOW_INCLUDE_DIR})
link_directories(${WORKFLOW_LIB_DIR})

set(SRC_FILE
        src/server.cc
        src/mimc_constants.cc
        src/darkforest.cc
        src/mimc.cu
        src/mimc_mpz.cc)

# add the executable
add_executable(df-explorer ${SRC_FILE})

target_link_libraries(df-explorer
        gmp
        workflow
        )

add_subdirectory(samples)