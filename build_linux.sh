#!/bin/bash
# WSL/Linux build script for MYTHOS.cpp
# PHASE B Task 49: WSL Linux build setup
# Usage: bash build_linux.sh

set -e

echo "=== MYTHOS.cpp Linux Build ==="

# Check prerequisites
command -v cmake >/dev/null 2>&1 || { echo "Installing cmake..."; sudo apt-get update && sudo apt-get install -y cmake; }
command -v g++ >/dev/null 2>&1 || { echo "Installing g++..."; sudo apt-get install -y g++; }
command -v ninja >/dev/null 2>&1 || { echo "Ninja not found, using Makefiles"; NINJA_FLAG=""; }
command -v ninja >/dev/null 2>&1 && { NINJA_FLAG="-G Ninja"; echo "Using Ninja"; }

# Configure
echo "Configuring..."
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release \
    -DMYTHOS_USE_CUDA=OFF \
    -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -Werror" \
    $NINJA_FLAG

# Build
echo "Building..."
cmake --build build-linux --parallel $(nproc)

# Test
echo "Running tests..."
cd build-linux
ctest --output-on-failure
cd ..

# Verify executables
echo "Verifying executables..."
for exe in build-linux/oil_train build-linux/oil_infer build-linux/oil_info build-linux/oil_bench; do
    if [ -f "$exe" ]; then
        SIZE=$(stat -c%s "$exe")
        if [ "$SIZE" -gt 0 ]; then
            echo "  OK: $exe ($SIZE bytes)"
        else
            echo "  FAIL: $exe is zero-size"
            exit 1
        fi
    fi
done

echo "=== Linux Build SUCCESS ==="
