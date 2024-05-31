#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=./bin/tile_example_fmha_fwd
VALID=1
set -ex

KNAME=1


for prec in "fp16" ; do
for perm in 0 1 ; do
for hdim in 128 ; do
for num_splits in 2 4 8 ; do

$EXE -prec=$prec -b=64  -h=8  -h_k=1 -d=$hdim -s=1 -s_k=8192   -iperm=$perm -operm=$perm -kname=$KNAME -v=$VALID -num_splits=$num_splits
$EXE -prec=$prec -b=128 -h=16 -h_k=1 -d=$hdim -s=1 -s_k=32768  -iperm=$perm -operm=$perm -kname=$KNAME -v=$VALID -num_splits=$num_splits

nhead=$((2048 / $hdim))     # follow fav2 setup
# $EXE -prec=$prec -b=32 -h=$nhead -d=$hdim -s=128 -s_k=512   -iperm=$perm -operm=$perm -kname=$KNAME -v=$VALID -num_splits=$num_splits
# $EXE -prec=$prec -b=16 -h=$nhead -d=$hdim -s=128 -s_k=1024  -iperm=$perm -operm=$perm -kname=$KNAME -v=$VALID -num_splits=$num_splits
# $EXE -prec=$prec -b=8  -h=$nhead -d=$hdim -s=128 -s_k=2048  -iperm=$perm -operm=$perm -kname=$KNAME -v=$VALID -num_splits=$num_splits
# $EXE -prec=$prec -b=4  -h=$nhead -d=$hdim -s=128 -s_k=4096  -iperm=$perm -operm=$perm -kname=$KNAME -v=$VALID -num_splits=$num_splits
# $EXE -prec=$prec -b=2  -h=$nhead -d=$hdim -s=128 -s_k=8192  -iperm=$perm -operm=$perm -kname=$KNAME -v=$VALID -num_splits=$num_splits
# $EXE -prec=$prec -b=1  -h=$nhead -d=$hdim -s=128 -s_k=16384 -iperm=$perm -operm=$perm -kname=$KNAME -v=$VALID -num_splits=$num_splits

done
done
done
done

exit

for perm in 0 1 ; do

$EXE -prec=fp8 -squant=1 -b=32 -h=16 -d=128 -s=512   -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=$KNAME -v=$VALID ; sleep 3
$EXE -prec=fp8 -squant=1 -b=16 -h=16 -d=128 -s=1024  -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=$KNAME -v=$VALID ; sleep 3
$EXE -prec=fp8 -squant=1 -b=8  -h=16 -d=128 -s=2048  -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=$KNAME -v=$VALID ; sleep 3
$EXE -prec=fp8 -squant=1 -b=4  -h=16 -d=128 -s=4096  -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=$KNAME -v=$VALID ; sleep 3
$EXE -prec=fp8 -squant=1 -b=2  -h=16 -d=128 -s=8192  -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=$KNAME -v=$VALID ; sleep 3
$EXE -prec=fp8 -squant=1 -b=1  -h=16 -d=128 -s=16384 -iperm=$perm -operm=$perm -vlayout=c -range_q=240 -range_k=240 -range_v=240 -range_p=240 -range_o=240 -kname=$KNAME -v=$VALID ; sleep 3

done
