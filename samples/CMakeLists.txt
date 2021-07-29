cmake_minimum_required(VERSION 3.10)

project(df-explorer-standalone LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/../src)

set(TEST_SRC_FILE
        standalone.cc
        ${PROJECT_SOURCE_DIR}/../src/mimc_constants.cc
        ${PROJECT_SOURCE_DIR}/../src/darkforest.cc
        ${PROJECT_SOURCE_DIR}/../src/mimc_mpz.cc
        ${PROJECT_SOURCE_DIR}/../src/mimc.cu)

# add the executable
add_executable(standalone ${TEST_SRC_FILE})

target_link_libraries(standalone gmp)
