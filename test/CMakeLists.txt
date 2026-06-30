# Optional LIT tests — enabled when lit + FileCheck are found.
# Search order (see README "工具链"):
#   1. Same LLVM build tree as MLIR (LLVM_DEFAULT_EXTERNAL_LIT / LLVM_TOOLS_BINARY_DIR)
#   2. LLVM_TOOLS_BINARY_DIR env override
#   3. PATH (e.g. apt llvm-*-tools)
set(_llvm_tool_hints "${LLVM_TOOLS_BINARY_DIR}")
if(DEFINED LLVM_DEFAULT_EXTERNAL_LIT AND LLVM_DEFAULT_EXTERNAL_LIT)
  get_filename_component(_lit_from_build "${LLVM_DEFAULT_EXTERNAL_LIT}" DIRECTORY)
  list(PREPEND _llvm_tool_hints "${_lit_from_build}")
endif()

find_program(FILECHECK_EXE NAMES FileCheck
  HINTS ${_llvm_tool_hints} ENV LLVM_TOOLS_BINARY_DIR)
find_program(LIT_EXE NAMES lit llvm-lit
  HINTS ${_llvm_tool_hints} ENV LLVM_TOOLS_BINARY_DIR)

if(FILECHECK_EXE AND LIT_EXE)
  set(_lit_dir "${CMAKE_CURRENT_BINARY_DIR}/lit")
  file(MAKE_DIRECTORY "${_lit_dir}")

  configure_file(lit/lit.site.cfg.py.in "${_lit_dir}/lit.site.cfg.py" @ONLY)
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/conv_bn_fusion.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/conv_bn_relu_fusion.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/softmax_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/affine_lowering.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/vector_lowering.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/dynamic_batch.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/constant_fold.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/attention_fusion_e2e.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/rmsnorm_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/attention_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/rmsnorm_fusion_e2e.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/rope_fusion_e2e.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/rope_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/bufferize_smoke.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/softmax_fusion_e2e.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/layernorm_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/layernorm_fusion_e2e.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/linalg_fusion_smoke.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/loops_smoke.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/gelu_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/canonicalize_cse_smoke.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/swiglu_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/qdq_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/matmul_bias_fusion.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/horizontal_gemm_fusion.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/elementwise_chain_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/producer_consumer_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/transformer_block_fusion_e2e.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/layout_bridge_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/kvcache_legalize.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/graph_partition_smoke.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/broadcast_simplify.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/gelu_linalg_smoke.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/shape_refine_batch.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/linalg_tile_smoke.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/shape_get_dimension_size.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/linalg_tile_fuse_smoke.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/flash_attention_tile.mlir"
       DESTINATION "${_lit_dir}")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/decode_loop.mlir"
       DESTINATION "${_lit_dir}")

  add_custom_target(test_jit_golden
    COMMAND ${CMAKE_COMMAND} -E env "MLIR_PASS_BUILD=${CMAKE_BINARY_DIR}"
            bash ${CMAKE_SOURCE_DIR}/scripts/run_jit_golden.sh
    DEPENDS pipe-demo
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "JIT numerical golden (pipe-demo vs NumPy)")

  add_custom_target(test_partition_smoke
    COMMAND ${CMAKE_COMMAND} -E env
            "MLIR_COMPILER_BUILD=${CMAKE_SOURCE_DIR}/../mlir_compiler/build"
            "MLIR_PASS_BUILD=${CMAKE_BINARY_DIR}"
            bash ${CMAKE_SOURCE_DIR}/scripts/run_partition_smoke.sh
    DEPENDS pipe-demo
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "P13 graph partition demo + mlir_pass partition smoke")

  add_custom_target(test_layout_e2e
    COMMAND ${CMAKE_COMMAND} -E env
            "MLIR_COMPILER_BUILD=${CMAKE_SOURCE_DIR}/../mlir_compiler/build"
            "MLIR_PASS_BUILD=${CMAKE_BINARY_DIR}"
            bash ${CMAKE_SOURCE_DIR}/scripts/run_layout_e2e.sh
    DEPENDS pipe-demo
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Layout conv P4 export -> mlir_pass layout-bridge e2e")

  add_custom_target(test_attention_e2e
    COMMAND ${CMAKE_COMMAND} -E env
            "MLIR_COMPILER_BUILD=${CMAKE_SOURCE_DIR}/../mlir_compiler/build"
            "MLIR_PASS_BUILD=${CMAKE_BINARY_DIR}"
            bash ${CMAKE_SOURCE_DIR}/scripts/run_attention_e2e.sh
    DEPENDS pipe-demo
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Cross-repo attention ONNX->StableHLO->mlir_pass fusion e2e")

  add_custom_target(test_transformer_e2e
    COMMAND ${CMAKE_COMMAND} -E env
            "MLIR_COMPILER_BUILD=${CMAKE_SOURCE_DIR}/../mlir_compiler/build"
            "MLIR_PASS_BUILD=${CMAKE_BINARY_DIR}"
            bash ${CMAKE_SOURCE_DIR}/scripts/run_transformer_e2e.sh
    DEPENDS pipe-demo
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Cross-repo Transformer ONNX->StableHLO->mlir_pass fusion e2e")

  add_custom_target(test_quant_e2e
    COMMAND ${CMAKE_COMMAND} -E env
            "MLIR_COMPILER_BUILD=${CMAKE_SOURCE_DIR}/../mlir_compiler/build"
            "MLIR_PASS_BUILD=${CMAKE_BINARY_DIR}"
            bash ${CMAKE_SOURCE_DIR}/scripts/run_quant_e2e.sh
    DEPENDS pipe-demo
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Q/DQ MatMul LIT + P4 export -> mlir_pass qdq-legalize e2e")

  add_custom_target(test_dynamic_e2e
    COMMAND ${CMAKE_COMMAND} -E env
            "MLIR_COMPILER_BUILD=${CMAKE_SOURCE_DIR}/../mlir_compiler/build"
            "MLIR_PASS_BUILD=${CMAKE_BINARY_DIR}"
            bash ${CMAKE_SOURCE_DIR}/scripts/run_dynamic_e2e.sh
    DEPENDS pipe-demo
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Dynamic batch P4 export -> mlir_pass fusion/linalg e2e")

  add_custom_target(test_broadcast_e2e
    COMMAND ${CMAKE_COMMAND} -E env
            "MLIR_COMPILER_BUILD=${CMAKE_SOURCE_DIR}/../mlir_compiler/build"
            "MLIR_PASS_BUILD=${CMAKE_BINARY_DIR}"
            bash ${CMAKE_SOURCE_DIR}/scripts/run_broadcast_e2e.sh
    DEPENDS pipe-demo
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "P4 numpy broadcast ONNX -> mlir_pass fusion/linalg e2e")

  add_custom_target(test_torch_e2e
    COMMAND ${CMAKE_COMMAND} -E env
            "MLIR_COMPILER_BUILD=${CMAKE_SOURCE_DIR}/../mlir_compiler/build"
            "MLIR_PASS_BUILD=${CMAKE_BINARY_DIR}"
            bash ${CMAKE_SOURCE_DIR}/scripts/run_torch_e2e.sh
    DEPENDS pipe-demo
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "torch-mlir export (or fixture) -> mlir_pass fusion e2e")

  add_custom_target(test_kvcache_e2e
    COMMAND ${CMAKE_COMMAND} -E env
            "MLIR_COMPILER_BUILD=${CMAKE_SOURCE_DIR}/../mlir_compiler/build"
            "MLIR_PASS_BUILD=${CMAKE_BINARY_DIR}"
            bash ${CMAKE_SOURCE_DIR}/scripts/run_kvcache_e2e.sh
    DEPENDS pipe-demo
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "P4 decode_step export -> mlir_pass fusion + P12 memplan e2e")

  add_custom_target(test_lit_filecheck
    COMMAND ${CMAKE_COMMAND} -E env "LIT_EXE=${LIT_EXE}" "FILECHECK_EXE=${FILECHECK_EXE}"
            bash ${CMAKE_SOURCE_DIR}/scripts/test_lit_filecheck.sh
    DEPENDS pipe-demo
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Running AI compiler LIT tests")

  add_custom_target(test_all
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/test_all.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS pipe-demo
    COMMENT "Shell regression + LIT/FileCheck")
else()
  message(STATUS "lit / FileCheck not found — test_lit_filecheck target skipped")
endif()

# Shell regression (always available when pipe-demo is built).
add_custom_target(test_shell_regression
  COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/test_shell_regression.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS pipe-demo
  COMMENT "Shell regression (pipeline + fusion + JIT)")
