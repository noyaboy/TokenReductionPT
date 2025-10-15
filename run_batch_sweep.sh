#!/bin/bash
# Batch size sweep - runs from project root directory
# Fixed to work with proper paths

OUTPUT_FILE="profiling_results/batch_size_sweep_results.csv"

echo "========================================="
echo "Testing Intermediate Batch Sizes"
echo "Finding TR crossover point"
echo "========================================="
echo ""

# Write CSV header
echo "batch_size,model,tp,latency_ms,flops" > $OUTPUT_FILE

# Test batch sizes: 1, 2, 4, 8, 16
for BS in 1 2 4 8 16; do
    echo "--- Testing Batch Size: $BS (baseline) ---"

    # Baseline (no TR)
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model deit_base_patch16_224.fb_in1k \
        --input-size 224 \
        --debugging \
        --batch-size $BS \
        --test_multiple 1 \
        --dummy_loader 2>&1 | grep "^cub_" | tail -1 > /tmp/result.txt

    if [ -s /tmp/result.txt ]; then
        TP=$(cat /tmp/result.txt | cut -d',' -f2)
        LAT=$(cat /tmp/result.txt | cut -d',' -f3)
        FLOPS=$(cat /tmp/result.txt | cut -d',' -f5)
        echo "$BS,baseline,$TP,$LAT,$FLOPS" >> $OUTPUT_FILE
        echo "  Baseline: $TP img/s, $LAT ms, $FLOPS GFLOPs"
    else
        echo "  ERROR: No output captured"
        echo "$BS,baseline,,," >> $OUTPUT_FILE
    fi

    echo ""
    echo "--- Testing Batch Size: $BS (with TR) ---"

    # With TR
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model topk_deit_base_patch16_224.fb_in1k \
        --input-size 224 \
        --debugging \
        --batch-size $BS \
        --test_multiple 1 \
        --keep_rate 0.1 \
        --reduction_loc 3 6 9 \
        --dummy_loader 2>&1 | grep "^cub_" | tail -1 > /tmp/result.txt

    if [ -s /tmp/result.txt ]; then
        TP=$(cat /tmp/result.txt | cut -d',' -f2)
        LAT=$(cat /tmp/result.txt | cut -d',' -f3)
        FLOPS=$(cat /tmp/result.txt | cut -d',' -f5)
        echo "$BS,with_tr,$TP,$LAT,$FLOPS" >> $OUTPUT_FILE
        echo "  With TR: $TP img/s, $LAT ms, $FLOPS GFLOPs"
    else
        echo "  ERROR: No output captured"
        echo "$BS,with_tr,,," >> $OUTPUT_FILE
    fi

    echo ""
done

echo "========================================="
echo "Batch size sweep complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "========================================="
