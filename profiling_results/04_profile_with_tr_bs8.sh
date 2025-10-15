#!/bin/bash
# Profile with Token Reduction (TR) with bs=8, is=448
# This is the case where TR performs well

echo "========================================="
echo "Profiling WITH TOKEN REDUCTION - Large Batch"
echo "Batch size: 8, Input size: 448"
echo "Keep rate: 0.1, Reduction locations: 3,6,9"
echo "========================================="

# Create output directory
mkdir -p nsys_traces

# Run with nsys profiling
nsys profile \
    --output=nsys_traces/with_tr_bs8_is448 \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --sample=cpu \
    --cpuctxsw=none \
    --backtrace=none \
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model topk_deit_base_patch16_224.fb_in1k \
        --input-size 448 \
        --debugging \
        --batch-size 8 \
        --test_multiple 1 \
        --keep_rate 0.1 \
        --reduction_loc 3 6 9 \
        --dummy_loader

echo ""
echo "Profiling complete! Output saved to: nsys_traces/with_tr_bs8_is448.nsys-rep"
