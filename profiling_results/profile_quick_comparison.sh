#!/bin/bash
# Quick nsys profiling for root cause analysis
# Profiles 4 key configurations to understand overhead

echo "========================================="
echo "Quick Profiling: Root Cause Analysis"
echo "========================================="
echo ""
echo "This will profile 4 configurations:"
echo "  1. is=224, bs=1, baseline (fast)"
echo "  2. is=224, bs=1, with TR (slow)"
echo "  3. is=448, bs=8, baseline (slow)"
echo "  4. is=448, bs=8, with TR (fast)"
echo ""
echo "Estimated time: ~10-15 minutes"
echo ""

mkdir -p nsys_traces

# Configuration 1: is=224, bs=1, baseline
echo "--- Config 1/4: is=224, bs=1, baseline ---"
nsys profile \
    --output=nsys_traces/quick_224_bs1_baseline \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --backtrace=none \
    --capture-range=cudaProfilerApi \
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model deit_base_patch16_224.fb_in1k \
        --input-size 224 \
        --debugging \
        --batch-size 1 \
        --test_multiple 1 \
        --warmup_iters 10 \
        --dummy_loader

echo ""
echo "✓ Config 1/4 complete"
echo ""

# Configuration 2: is=224, bs=1, with TR
echo "--- Config 2/4: is=224, bs=1, with TR ---"
nsys profile \
    --output=nsys_traces/quick_224_bs1_with_tr \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --backtrace=none \
    --capture-range=cudaProfilerApi \
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model topk_deit_base_patch16_224.fb_in1k \
        --input-size 224 \
        --debugging \
        --batch-size 1 \
        --test_multiple 1 \
        --keep_rate 0.1 \
        --reduction_loc 3 6 9 \
        --warmup_iters 10 \
        --dummy_loader

echo ""
echo "✓ Config 2/4 complete"
echo ""

# Configuration 3: is=448, bs=8, baseline
echo "--- Config 3/4: is=448, bs=8, baseline ---"
nsys profile \
    --output=nsys_traces/quick_448_bs8_baseline \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --backtrace=none \
    --capture-range=cudaProfilerApi \
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model deit_base_patch16_224.fb_in1k \
        --input-size 448 \
        --debugging \
        --batch-size 8 \
        --test_multiple 1 \
        --warmup_iters 10 \
        --dummy_loader

echo ""
echo "✓ Config 3/4 complete"
echo ""

# Configuration 4: is=448, bs=8, with TR
echo "--- Config 4/4: is=448, bs=8, with TR ---"
nsys profile \
    --output=nsys_traces/quick_448_bs8_with_tr \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --backtrace=none \
    --capture-range=cudaProfilerApi \
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model topk_deit_base_patch16_224.fb_in1k \
        --input-size 448 \
        --debugging \
        --batch-size 8 \
        --test_multiple 1 \
        --keep_rate 0.1 \
        --reduction_loc 3 6 9 \
        --warmup_iters 10 \
        --dummy_loader

echo ""
echo "✓ Config 4/4 complete"
echo ""

echo "========================================="
echo "Quick profiling complete!"
echo "========================================="
echo ""
echo "Analyzing kernel counts..."
echo ""

# Quick analysis
for trace in nsys_traces/quick_*.nsys-rep; do
    echo "--- $(basename $trace .nsys-rep) ---"
    nsys stats --report cuda_gpu_kern_sum $trace 2>/dev/null | head -15 | tail -10 || echo "Analysis pending..."
    echo ""
done

echo "To analyze in detail:"
echo "  nsys stats --report cuda_gpu_kern_sum nsys_traces/quick_224_bs1_baseline.nsys-rep"
echo "  nsys stats --report cuda_gpu_kern_sum nsys_traces/quick_224_bs1_with_tr.nsys-rep"
echo ""
echo "Or open in Nsight Systems GUI for visual comparison"
