get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_executable(ai-compiler-demo main.cpp)

target_link_libraries(ai-compiler-demo PRIVATE
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

target_include_directories(ai-compiler-demo PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Transforms
)
