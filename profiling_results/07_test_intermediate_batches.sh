#!/bin/bash
# Test intermediate batch sizes to find crossover point
# Where does TR start becoming beneficial?

echo "========================================="
echo "Testing Intermediate Batch Sizes"
echo "Finding TR crossover point"
echo "========================================="

OUTPUT_FILE="batch_size_sweep_results.csv"

# Write CSV header
echo "batch_size,model,tp,latency_ms,flops" > $OUTPUT_FILE

# Test batch sizes: 1, 2, 4, 8, 16
for BS in 1 2 4 8 16; do
    echo ""
    echo "--- Testing Batch Size: $BS (baseline) ---"

    # Baseline (no TR)
    RESULT=$(python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model deit_base_patch16_224.fb_in1k \
        --input-size 224 \
        --debugging \
        --batch-size $BS \
        --test_multiple 1 \
        --dummy_loader 2>&1 | grep "^cub_" | tail -1)

    # Parse result
    TP=$(echo $RESULT | cut -d',' -f2)
    LAT=$(echo $RESULT | cut -d',' -f3)
    FLOPS=$(echo $RESULT | cut -d',' -f5)

    echo "$BS,baseline,$TP,$LAT,$FLOPS" >> $OUTPUT_FILE
    echo "  Baseline: $TP img/s, $LAT ms, $FLOPS GFLOPs"

    echo ""
    echo "--- Testing Batch Size: $BS (with TR) ---"

    # With TR
    RESULT=$(python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model topk_deit_base_patch16_224.fb_in1k \
        --input-size 224 \
        --debugging \
        --batch-size $BS \
        --test_multiple 1 \
        --keep_rate 0.1 \
        --reduction_loc 3 6 9 \
        --dummy_loader 2>&1 | grep "^cub_" | tail -1)

    # Parse result
    TP=$(echo $RESULT | cut -d',' -f2)
    LAT=$(echo $RESULT | cut -d',' -f3)
    FLOPS=$(echo $RESULT | cut -d',' -f5)

    echo "$BS,with_tr,$TP,$LAT,$FLOPS" >> $OUTPUT_FILE
    echo "  With TR: $TP img/s, $LAT ms, $FLOPS GFLOPs"
done

echo ""
echo "========================================="
echo "Batch size sweep complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "========================================="
echo ""
echo "Analyzing crossover point..."
python - << 'EOF'
import pandas as pd

df = pd.read_csv('batch_size_sweep_results.csv')
baseline = df[df['model'] == 'baseline'].set_index('batch_size')
with_tr = df[df['model'] == 'with_tr'].set_index('batch_size')

print("\nComparison (Speedup with TR):")
print("=" * 50)
print("Batch Size | Baseline TP | TR TP | Speedup | Better?")
print("-" * 50)

for bs in baseline.index:
    base_tp = float(baseline.loc[bs, 'tp'])
    tr_tp = float(with_tr.loc[bs, 'tp'])
    speedup = tr_tp / base_tp
    better = "✓ YES" if speedup > 1.0 else "✗ NO"
    print(f"{bs:10d} | {base_tp:11.2f} | {tr_tp:5.2f} | {speedup:7.3f}x | {better}")

print("=" * 50)
print("\nCrossover point: Find where speedup > 1.0")
EOF
