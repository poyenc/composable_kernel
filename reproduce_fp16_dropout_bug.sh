#!/bin/bash
# Reproducer for fp16 d256 dropout miscompilation on gfx90a
#
# Bug: The AMDGPU register allocator in ROCm 7.1.1 clang generates incorrect
# code for the fp16 d256 dropout FMHA kernel on gfx90a. The dropout RANDVAL
# output (buffer_store_byte) contains wrong data for wave lanes 32-63.
#
# This script builds the test twice:
#   1. Without workaround → fp16 dropout FAILS
#   2. With -DCK_TILE_FMHA_FWD_REGALLOC_WORKAROUND → fp16 dropout PASSES
#
# The workaround adds a conditional branch before store_tile() that changes
# the register allocator's live range splitting. It does not change VGPR
# count, spills, or scratch size.
#
# Prerequisites: ROCm 7.1.1 with gfx90a GPU, CMake, Ninja
#
# Usage:
#   cd composable_kernel
#   bash reproduce_fp16_dropout_bug.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEST_FILTER='TestCkTileFmhaFwd/Dropout.FmhaFwdFp16/96'

echo "=== Build 1: WITHOUT workaround ==="
rm -rf build && mkdir build && cd build
../script/cmake-ck-dev.sh .. 'gfx90a' -G Ninja
ninja test_ck_tile_fmha_fwd_fp16

echo ""
echo "=== Test 1: fp16 dropout (EXPECT FAIL) ==="
RESULT_WITHOUT=PASS
./bin/test_ck_tile_fmha_fwd_fp16 --gtest_filter="$TEST_FILTER" 2>&1 \
    || RESULT_WITHOUT=FAIL

echo ""
echo "=== Build 2: WITH workaround (-DCK_TILE_FMHA_FWD_REGALLOC_WORKAROUND) ==="
cd "$SCRIPT_DIR"
rm -rf build && mkdir build && cd build
../script/cmake-ck-dev.sh .. 'gfx90a' -G Ninja \
    '-DCMAKE_HIP_FLAGS=-DCK_TILE_FMHA_FWD_REGALLOC_WORKAROUND'
ninja test_ck_tile_fmha_fwd_fp16

echo ""
echo "=== Test 2: fp16 dropout (EXPECT PASS) ==="
RESULT_WITH=PASS
./bin/test_ck_tile_fmha_fwd_fp16 --gtest_filter="$TEST_FILTER" 2>&1 \
    || RESULT_WITH=FAIL

echo ""
echo "=== Results ==="
echo ""
echo "  Without workaround: $RESULT_WITHOUT (expected FAIL)"
echo "  With workaround:    $RESULT_WITH (expected PASS)"
echo ""
echo "The workaround (CK_TILE_FMHA_FWD_REGALLOC_WORKAROUND) adds a conditional"
echo "branch in block_dropout.hpp before store_tile(). This changes the register"
echo "allocator's live range splitting without affecting register pressure."
