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
  get_filename_component(_fc_dir "${FILECHECK_EXE}" DIRECTORY)

  set(_lit_dir "${CMAKE_CURRENT_BINARY_DIR}/lit")
  file(MAKE_DIRECTORY "${_lit_dir}")

  configure_file(lit/lit.site.cfg.py.in "${_lit_dir}/lit.site.cfg.py" @ONLY)
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lit/conv_bn_fusion.mlir"
       DESTINATION "${_lit_dir}")

  add_custom_target(check-ai-compiler
    COMMAND ${CMAKE_COMMAND} -E env "PATH=${_fc_dir}"
            "${LIT_EXE}" -sv "${_lit_dir}"
    DEPENDS ai-compiler-demo
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Running AI compiler LIT tests")

  add_custom_target(test-ai-compiler-all
    COMMAND bash ${CMAKE_SOURCE_DIR}/test/run_tests.sh --all
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ai-compiler-demo check-ai-compiler
    COMMENT "Shell regression + LIT/FileCheck")
else()
  message(STATUS "lit / FileCheck not found — check-ai-compiler target skipped")
endif()

# Shell regression (always available when ai-compiler-demo is built).
add_custom_target(test-ai-compiler
  COMMAND bash ${CMAKE_SOURCE_DIR}/test/run_tests.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS ai-compiler-demo
  COMMENT "Shell regression tests (pipeline + fusion + JIT)")
