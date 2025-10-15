#!/bin/bash
# Profile with Token Reduction (TR) with bs=1, is=224
# This captures the full timeline with nsys

echo "========================================="
echo "Profiling WITH TOKEN REDUCTION"
echo "Batch size: 1, Input size: 224"
echo "Keep rate: 0.1, Reduction locations: 3,6,9"
echo "========================================="

# Create output directory
mkdir -p nsys_traces

# Run with nsys profiling
nsys profile \
    --output=nsys_traces/with_tr_bs1_is224 \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --sample=cpu \
    --cpuctxsw=none \
    --backtrace=none \
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model topk_deit_base_patch16_224.fb_in1k \
        --input-size 224 \
        --debugging \
        --batch-size 1 \
        --test_multiple 1 \
        --keep_rate 0.1 \
        --reduction_loc 3 6 9 \
        --dummy_loader

echo ""
echo "Profiling complete! Output saved to: nsys_traces/with_tr_bs1_is224.nsys-rep"
echo ""
echo "To analyze:"
echo "  1. Download the .nsys-rep file"
echo "  2. Open in NVIDIA Nsight Systems GUI"
echo "  3. Or use: nsys stats nsys_traces/with_tr_bs1_is224.nsys-rep"
