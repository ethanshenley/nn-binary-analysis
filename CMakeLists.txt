cmake_minimum_required(VERSION 3.15)
project(nn_binary_analysis VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM 15.0 REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/nn_binary_analysis/include
    /usr/include/llvm-15
    ${LLVM_INCLUDE_DIRS}
)

add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(llvm_libs
    Support Core IRReader MCParser MC MCDisassembler
    Target AllTargetsDescs AllTargetsDisassemblers AllTargetsInfos
    AsmParser AsmPrinter AllTargets
)

add_library(nn_binary_analysis
    nn_binary_analysis/src/core/binary_parser.cpp
    nn_binary_analysis/src/core/memory_analyzer.cpp
    nn_binary_analysis/src/core/transformer_detector.cpp
    nn_binary_analysis/src/core/framework_detector.cpp
)

target_link_libraries(nn_binary_analysis PUBLIC ${llvm_libs} ${ZSTD_LIBRARIES})

add_executable(nn_detect tools/nn_detect/main.cpp)
target_link_libraries(nn_detect PRIVATE nn_binary_analysis)

add_executable(nn_analyze tools/nn_analyze/main.cpp)
target_link_libraries(nn_analyze PRIVATE nn_binary_analysis)