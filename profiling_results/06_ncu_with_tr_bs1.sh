#!/bin/bash
# Deep kernel analysis with NCU for Token Reduction case
# Focus on TR-specific operations (topk, gather, indexing)

echo "========================================="
echo "NCU Deep Kernel Analysis - WITH TR"
echo "Batch size: 1, Input size: 224"
echo "========================================="

# Create output directory
mkdir -p ncu_reports

# Run with ncu profiling
ncu \
    --set full \
    --target-processes all \
    --kernel-name-base=demangled \
    --launch-skip 100 \
    --launch-count 10 \
    --export ncu_reports/with_tr_bs1_detailed \
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
echo "NCU profiling complete! Output saved to: ncu_reports/with_tr_bs1_detailed.ncu-rep"
echo ""
echo "Look for TR-specific kernels:"
echo "  - topk selection kernels"
echo "  - gather/index operations"
echo "  - Check their occupancy and efficiency"
