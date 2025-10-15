#!/bin/bash
# Deep kernel analysis with NCU for baseline (no TR)
# This profiles specific kernels for compute/memory characteristics

echo "========================================="
echo "NCU Deep Kernel Analysis - BASELINE"
echo "Batch size: 1, Input size: 224"
echo "========================================="

# Create output directory
mkdir -p ncu_reports

# Run with ncu profiling
# We'll profile the most expensive kernels with detailed metrics
ncu \
    --set full \
    --target-processes all \
    --kernel-name-base=demangled \
    --launch-skip 100 \
    --launch-count 10 \
    --export ncu_reports/baseline_bs1_detailed \
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model deit_base_patch16_224.fb_in1k \
        --input-size 224 \
        --debugging \
        --batch-size 1 \
        --test_multiple 1 \
        --dummy_loader

echo ""
echo "NCU profiling complete! Output saved to: ncu_reports/baseline_bs1_detailed.ncu-rep"
echo ""
echo "Key metrics to check:"
echo "  - SM Efficiency (compute utilization)"
echo "  - Memory Throughput vs Peak"
echo "  - Warp Occupancy"
echo "  - Compute vs Memory Bound classification"
