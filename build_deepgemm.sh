#!/bin/bash
set -euo pipefail
# Get repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel)
export DG_JIT_CACHE_DIR=$REPO_ROOT/.cache
export DG_JIT_USE_RUNTIME_API=1
export DG_JIT_COMPILER_COMMAND=1
export DG_USE_LOCAL_VERSION=0
export DG_DEBUG_BUILD=1

# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Link CUTLASS includes
ln -sf $script_dir/third-party/cutlass/include/cutlass deep_gemm/include
ln -sf $script_dir/third-party/cutlass/include/cute deep_gemm/include

# # Remove old dist file, build files, and build
# rm -rf build dist
# rm -rf *.egg-info
# python setup.py build

# uv pip install --no-build-isolation -v -e . --force-reinstall 2>&1 | tee build.log

# # Find the .so file in build directory and create symlink in current directory
so_file=$(find build -name "*.so" -type f | head -n 1)
if [ -n "$so_file" ]; then
    ln -sf "../$so_file" deep_gemm/
else
    echo "Error: No SO file found in build directory" >&2
    exit 1
fi

# # Open users' original directory
# cd "$original_dir"
