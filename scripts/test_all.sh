#!/usr/bin/env bash
# Run Shell regression followed by LIT/FileCheck.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT}/scripts/test_shell_regression.sh"
bash "${ROOT}/scripts/test_lit_filecheck.sh"
