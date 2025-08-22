#!/bin/sh
# TODO: run this script from CK root or build directory
EXE="$(find . -name tile_example_fmha_fwd_v3 -type f | head -n 1)"
VALID=0

for prec in "fp16" "bf16" ; do
for hdim in 128 ; do
for perm in 0 ; do

$EXE -prec=$prec -b=32 -h=16        -s=512   -d=$hdim -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=16 -h=16        -s=1024  -d=$hdim -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=8  -h=16        -s=2048  -d=$hdim -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=4  -h=16        -s=4096  -d=$hdim -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=2  -h=16        -s=8192  -d=$hdim -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=1  -h=16        -s=16384 -d=$hdim -iperm=$perm -operm=$perm -v=$VALID
                                          
$EXE -prec=$prec -b=1  -h=64        -s=16384 -d=$hdim -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=1  -h=16 -h_k=1 -s=65536 -d=$hdim -iperm=$perm -operm=$perm -v=$VALID
$EXE -prec=$prec -b=1  -h=40        -s=37200 -d=$hdim -iperm=$perm -operm=$perm -v=$VALID

done
done
done
