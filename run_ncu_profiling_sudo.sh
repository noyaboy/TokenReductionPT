#!/bin/bash
# NCU profiling with sudo - requires password input
# This script must be run directly in terminal, not in background

echo "========================================="
echo "NCU Profiling with sudo (Deep Kernel Analysis)"
echo "========================================="
echo ""
echo "This will profile 5 key configurations:"
echo "  1. is=224, bs=1, baseline"
echo "  2. is=224, bs=1, with TR (overhead analysis)"
echo "  3. is=448, bs=8, with TR (speedup analysis)"
echo "  4. is=448, bs=4, with TR (anomaly investigation)"
echo "  5. is=448, bs=1, with TR (best speedup case)"
echo ""
echo "Estimated time: 30-60 minutes total"
echo ""

mkdir -p profiling_results/ncu_reports

# Config 1: is=224, bs=1, baseline
echo "--- Config 1/5: is=224, bs=1, baseline ---"
sudo ncu \
    --set full \
    --export profiling_results/ncu_reports/ncu_224_bs1_baseline \
    --force-overwrite \
    --target-processes all \
    /home/noah/miniconda3/bin/python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model deit_base_patch16_224.fb_in1k \
        --input-size 224 \
        --debugging \
        --batch-size 1 \
        --test_multiple 1 \
        --warmup_iters 5 \
        --dummy_loader

echo "✓ Config 1/5 complete"
echo ""

# Config 2: is=224, bs=1, with TR
echo "--- Config 2/5: is=224, bs=1, with TR ---"
sudo ncu \
    --set full \
    --export profiling_results/ncu_reports/ncu_224_bs1_with_tr \
    --force-overwrite \
    --target-processes all \
    /home/noah/miniconda3/bin/python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model topk_deit_base_patch16_224.fb_in1k \
        --input-size 224 \
        --debugging \
        --batch-size 1 \
        --test_multiple 1 \
        --keep_rate 0.1 \
        --reduction_loc 3 6 9 \
        --warmup_iters 5 \
        --dummy_loader

echo "✓ Config 2/5 complete"
echo ""

# Config 3: is=448, bs=8, with TR
echo "--- Config 3/5: is=448, bs=8, with TR ---"
sudo ncu \
    --set full \
    --export profiling_results/ncu_reports/ncu_448_bs8_with_tr \
    --force-overwrite \
    --target-processes all \
    /home/noah/miniconda3/bin/python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model topk_deit_base_patch16_224.fb_in1k \
        --input-size 448 \
        --debugging \
        --batch-size 8 \
        --test_multiple 1 \
        --keep_rate 0.1 \
        --reduction_loc 3 6 9 \
        --warmup_iters 5 \
        --dummy_loader

echo "✓ Config 3/5 complete"
echo ""

# Config 4: is=448, bs=4, with TR (anomaly)
echo "--- Config 4/5: is=448, bs=4, with TR ---"
sudo ncu \
    --set full \
    --export profiling_results/ncu_reports/ncu_448_bs4_with_tr \
    --force-overwrite \
    --target-processes all \
    /home/noah/miniconda3/bin/python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model topk_deit_base_patch16_224.fb_in1k \
        --input-size 448 \
        --debugging \
        --batch-size 4 \
        --test_multiple 1 \
        --keep_rate 0.1 \
        --reduction_loc 3 6 9 \
        --warmup_iters 5 \
        --dummy_loader

echo "✓ Config 4/5 complete"
echo ""

# Config 5: is=448, bs=1, with TR (best speedup)
echo "--- Config 5/5: is=448, bs=1, with TR ---"
sudo ncu \
    --set full \
    --export profiling_results/ncu_reports/ncu_448_bs1_with_tr \
    --force-overwrite \
    --target-processes all \
    /home/noah/miniconda3/bin/python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model topk_deit_base_patch16_224.fb_in1k \
        --input-size 448 \
        --debugging \
        --batch-size 1 \
        --test_multiple 1 \
        --keep_rate 0.1 \
        --reduction_loc 3 6 9 \
        --warmup_iters 5 \
        --dummy_loader

echo "✓ Config 5/5 complete"
echo ""

echo "========================================="
echo "NCU Profiling Complete!"
echo "========================================="
echo ""
echo "Reports saved in: profiling_results/ncu_reports/"
echo ""
echo "Generated files:"
ls -lh profiling_results/ncu_reports/*.ncu-rep 2>/dev/null || echo "  (checking...)"
echo ""
echo "To analyze in GUI:"
echo "  ncu-ui profiling_results/ncu_reports/ncu_224_bs1_with_tr.ncu-rep"
echo ""
echo "To view CLI summary:"
echo "  ncu --import profiling_results/ncu_reports/ncu_224_bs1_with_tr.ncu-rep --page details"
echo ""
