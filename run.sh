#!/bin/bash
set -ex

for n in $(seq 5); do

../example/ck_tile/01_fmha/script/benchmark_fwd.sh 2>&1 | tee perf-$n.txt

done
