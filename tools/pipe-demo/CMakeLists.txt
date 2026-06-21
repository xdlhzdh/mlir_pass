get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_executable(pipe-demo main.cpp)

target_link_libraries(pipe-demo PRIVATE
  AICompilerPipeline
  AICompilerTransforms
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
  MLIRExecutionEngine
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  ${dialect_libs}
  ${conversion_libs}
  -Wl,--start-group
  ${STABLEHLO_IMPORTED}
  -Wl,--end-group
)

target_include_directories(pipe-demo PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Transforms
)
