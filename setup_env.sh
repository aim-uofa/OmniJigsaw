#!/usr/bin/env bash

set -e
set -o pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGES_DIR="${PROJECT_ROOT}/packages"
VENV_DIR="${PROJECT_ROOT}/.venv"
VLLM_ARCHIVE="${PACKAGES_DIR}/vllm-qwen3-omni.zip"
VLLM_SOURCE_DIR="${PACKAGES_DIR}/vllm"
FLASH_ATTN_WHEEL="${PACKAGES_DIR}/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$(basename -- "${FLASH_ATTN_WHEEL}")"

cd "${PROJECT_ROOT}"
mkdir -p "${PACKAGES_DIR}"

command -v uv >/dev/null 2>&1 || {
    echo "Error: uv is required but was not found in PATH." >&2
    exit 1
}
command -v unzip >/dev/null 2>&1 || {
    echo "Error: unzip is required but was not found in PATH." >&2
    exit 1
}
command -v git >/dev/null 2>&1 || {
    echo "Error: git is required but was not found in PATH." >&2
    exit 1
}

if [ ! -f "${VLLM_ARCHIVE}" ]; then
    echo "Error: vLLM source archive not found: ${VLLM_ARCHIVE}" >&2
    exit 1
fi
if [ ! -f "${FLASH_ATTN_WHEEL}" ]; then
    command -v wget >/dev/null 2>&1 || {
        echo "Error: wget is required to download the flash-attn wheel." >&2
        exit 1
    }
    echo "Downloading the flash-attn wheel from the official release..."
    wget -c -O "${FLASH_ATTN_WHEEL}.part" "${FLASH_ATTN_URL}"
    mv -- "${FLASH_ATTN_WHEEL}.part" "${FLASH_ATTN_WHEEL}"
fi

rm -rf "${VLLM_SOURCE_DIR}"
unzip -q "${VLLM_ARCHIVE}" -d "${PACKAGES_DIR}"
if [ ! -d "${VLLM_SOURCE_DIR}/.git" ]; then
    echo "Error: the extracted vLLM source does not contain Git metadata." >&2
    exit 1
fi

if [ -e "${VENV_DIR}" ]; then
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
        echo "Error: ${VENV_DIR} exists but is not a valid virtual environment." >&2
        exit 1
    fi
    if ! "${VENV_DIR}/bin/python" -c \
        'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
        echo "Error: the existing virtual environment does not use Python 3.12: ${VENV_DIR}" >&2
        exit 1
    fi
    echo "Reusing existing virtual environment: ${VENV_DIR}"
else
    uv venv "${VENV_DIR}" --python 3.12
fi
source "${VENV_DIR}/bin/activate"
export UV_INDEX_URL="https://pypi.org/simple/"
uv pip install -e '.[vllm]'
uv pip install -r requirements.txt
uv pip uninstall vllm

VLLM_SAFE_DIRECTORY_ADDED=0
cleanup_vllm_safe_directory() {
    if [ "${VLLM_SAFE_DIRECTORY_ADDED}" -eq 1 ]; then
        git config --global --fixed-value --unset-all \
            safe.directory "${VLLM_SOURCE_DIR}" || true
    fi
}
trap cleanup_vllm_safe_directory EXIT
if ! git config --global --fixed-value --get-all \
    safe.directory "${VLLM_SOURCE_DIR}" >/dev/null; then
    git config --global --add safe.directory "${VLLM_SOURCE_DIR}"
    VLLM_SAFE_DIRECTORY_ADDED=1
fi

cd "${VLLM_SOURCE_DIR}"
VLLM_USE_PRECOMPILED=1 \
    uv pip install -e . -v --no-build-isolation
cd "${PROJECT_ROOT}"
cleanup_vllm_safe_directory
trap - EXIT

uv pip install "${FLASH_ATTN_WHEEL}"
# Resolve the compatible dataset dependencies together with Transformers.
uv pip install transformers==4.57.1 datasets==5.0.1 fsspec==2026.6.0
python env_fix/env_fix.py

echo
echo "Environment setup completed successfully."
echo "Activate it before training with:"
echo "  source ${VENV_DIR}/bin/activate"
