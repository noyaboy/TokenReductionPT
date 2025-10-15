#!/bin/bash
# Profile baseline (no TR) with bs=8, is=448
# This captures the large batch case where TR works well

echo "========================================="
echo "Profiling BASELINE (no TR) - Large Batch"
echo "Batch size: 8, Input size: 448"
echo "========================================="

# Create output directory
mkdir -p nsys_traces

# Run with nsys profiling
nsys profile \
    --output=nsys_traces/baseline_bs8_is448 \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --sample=cpu \
    --cpuctxsw=none \
    --backtrace=none \
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model deit_base_patch16_224.fb_in1k \
        --input-size 448 \
        --debugging \
        --batch-size 8 \
        --test_multiple 1 \
        --dummy_loader

echo ""
echo "Profiling complete! Output saved to: nsys_traces/baseline_bs8_is448.nsys-rep"
