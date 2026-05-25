#!/usr/bin/env bash
# Run shell regression followed by LIT/FileCheck.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT}/scripts/test_pipeline_demo.sh"
bash "${ROOT}/scripts/test_lit_filecheck.sh"
