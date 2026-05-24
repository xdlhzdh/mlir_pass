cmake_minimum_required(VERSION 3.20)
project(AICompilerDemo LANGUAGES CXX C)

# For clangd / IDE (compile_commands.json in build/).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

set(STABLEHLO_INCLUDE_DIR "/usr/local/include" CACHE PATH "StableHLO headers")
set(STABLEHLO_LIB_DIR "/usr/local/lib" CACHE PATH "StableHLO libraries")
include_directories(${STABLEHLO_INCLUDE_DIR})

file(GLOB _stablehlo_lib_files
  "${STABLEHLO_LIB_DIR}/libStablehlo*.a"
  "${STABLEHLO_LIB_DIR}/libChlo*.a"
  "${STABLEHLO_LIB_DIR}/libVhlo*.a")
set(STABLEHLO_IMPORTED "${_stablehlo_lib_files}" CACHE INTERNAL "All StableHLO static libraries")

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

# One-click pipeline demo: per-stage IR files under output/pipeline-dumps/latest/
add_custom_target(
  run-pipeline-demo
  COMMAND ${CMAKE_COMMAND} -E env AI_COMPILER_DEMO=${CMAKE_BINARY_DIR}/tools/ai-compiler-demo/ai-compiler-demo
          bash ${CMAKE_SOURCE_DIR}/scripts/run_pipeline_demo.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS ai-compiler-demo
  COMMENT "Run full pipeline on all test .mlir files; write IR dumps to output/pipeline-dumps/latest/")
