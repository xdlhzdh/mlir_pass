cmake_minimum_required(VERSION 3.20)

# LLVM installs clang-cl into /usr/local/bin. On Linux/WSL it uses the MSVC
# driver and cannot link; CMake may pick it when CMAKE_PREFIX_PATH=/usr/local.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_CXX_COMPILER)
  find_program(_AICOMPILER_CXX c++ clang++
    HINTS /usr/bin /usr/local/bin
    NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH)
  if(_AICOMPILER_CXX AND NOT _AICOMPILER_CXX MATCHES "clang-cl$")
    set(CMAKE_CXX_COMPILER "${_AICOMPILER_CXX}" CACHE STRING "C++ compiler")
  endif()
  unset(_AICOMPILER_CXX)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_C_COMPILER AND CMAKE_CXX_COMPILER)
  get_filename_component(_AICOMPILER_CXX_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
  find_program(_AICOMPILER_C cc clang
    HINTS "${_AICOMPILER_CXX_DIR}" /usr/bin
    NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH)
  if(_AICOMPILER_C AND NOT _AICOMPILER_C MATCHES "clang-cl$")
    set(CMAKE_C_COMPILER "${_AICOMPILER_C}" CACHE STRING "C compiler")
  endif()
  unset(_AICOMPILER_C)
  unset(_AICOMPILER_CXX_DIR)
endif()

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
  run_pipeline_demo
  COMMAND ${CMAKE_COMMAND} -E env PIPE_DEMO=${CMAKE_BINARY_DIR}/tools/pipe-demo/pipe-demo
          bash ${CMAKE_SOURCE_DIR}/scripts/run_pipeline_demo.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS pipe-demo
  COMMENT "Run full pipeline on all test .mlir files; write IR dumps to output/pipeline-dumps/latest/")
