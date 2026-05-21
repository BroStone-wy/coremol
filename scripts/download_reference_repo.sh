#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/third_party/CurvFlow-Transformer"

mkdir -p "${ROOT_DIR}/third_party"
if [ -d "${TARGET_DIR}/.git" ]; then
  git -C "${TARGET_DIR}" pull --ff-only
else
  git clone https://github.com/juniormaidao/CurvFlow-Transformer "${TARGET_DIR}"
fi

