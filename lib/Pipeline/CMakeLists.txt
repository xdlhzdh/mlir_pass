add_library(AICompilerPipeline STATIC
  Pipeline.cpp
  PipelineStages.cpp
)

target_include_directories(AICompilerPipeline PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../Transforms
)

target_link_libraries(AICompilerPipeline PUBLIC
  AICompilerTransforms
)
