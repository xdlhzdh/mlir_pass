get_property(_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(_plugin_pass_srcs
  AICompilerPlugin.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/StablehloConstantFold.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/ConvBNFusion.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/ConvBNReluFusion.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/SoftmaxLegalize.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/RMSNormLegalize.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/AttentionLegalize.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/RoPELegalize.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/LayerNormLegalize.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/GeluLegalize.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/SwiGLULegalize.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/QdqLegalize.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/MatMulBiasFusion.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/HorizontalGemmFusion.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/ElementwiseChainLegalize.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/ProducerConsumerLegalize.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/LayoutBridgeLegalize.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/KVCacheLegalize.cpp
)

add_library(AICompilerPlugin SHARED ${_plugin_pass_srcs})
add_dependencies(AICompilerPlugin AICompilerPassIncGen)

target_include_directories(AICompilerPlugin PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/lib/Transforms
  ${STABLEHLO_INCLUDE_DIR}
)

target_compile_options(AICompilerPlugin PRIVATE -fno-rtti)

target_link_libraries(AICompilerPlugin PRIVATE
  -Wl,--whole-archive
  MLIRIR
  MLIRRewrite
  -Wl,--no-whole-archive
  MLIRPass
  MLIRTransforms
  ${_dialect_libs}
  -Wl,--start-group
  ${STABLEHLO_IMPORTED}
  -Wl,--end-group
)

if(DEFINED MLIR_DIR)
  get_filename_component(_mlir_bin "${MLIR_DIR}/../../bin" ABSOLUTE)
  if(EXISTS "${_mlir_bin}/mlir-opt")
    set(MLIR_OPT_EXE "${_mlir_bin}/mlir-opt")
  else()
    set(MLIR_OPT_EXE "mlir-opt")
  endif()
else()
  set(MLIR_OPT_EXE "mlir-opt")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/run_aicom_mlir_opt.sh.in"
  "${CMAKE_CURRENT_BINARY_DIR}/run_aicom_mlir_opt.sh"
  @ONLY)

add_custom_target(test_mlir_opt_plugin
  COMMAND sh "${CMAKE_CURRENT_BINARY_DIR}/run_aicom_mlir_opt.sh"
          "${MLIR_OPT_EXE}"
          "${CMAKE_SOURCE_DIR}/test/lit/constant_fold.mlir"
          "${CMAKE_CURRENT_BINARY_DIR}/constant_fold_fused.mlir"
  DEPENDS AICompilerPlugin
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running mlir-opt with AICompilerPlugin (aicom-fusion)")
