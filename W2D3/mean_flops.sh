#!/bin/bash

log_file=$1

awk '
match($0, /Step Time : [0-9.]+s.*GPU utilization: [0-9.]+/) {
    n++
    if (n == 1) next   # skip warmup / outlier

    t = $0
    gsub(/.*Step Time : /, "", t)
    gsub(/s.*/, "", t)

    g = $0
    gsub(/.*GPU utilization: /, "", g)
    gsub(/TFLOP.*/, "", g)

    sum_time += t
    sum_gpu  += g
    count++
}
END {
    printf "Average Step Time: %.3fs\n", sum_time/count
    printf "Average GPU Utilization: %.2f TFLOP/s/GPU\n", sum_gpu/count
}
' $1
MEM_ALLOC=$(awk -F'mem-allocated-gigabytes: ' '/mem-allocated-gigabytes:/ {split($2, a, " "); print a[1]}' $1)
echo "Allocated Memory (GB): ${MEM_ALLOC}"


