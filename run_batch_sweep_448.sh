#!/bin/bash
# Batch size sweep with INPUT SIZE 448 to validate hypothesis
# Hypothesis: TR should be FASTER at large input sizes

OUTPUT_FILE="profiling_results/batch_size_sweep_448_results.csv"

echo "========================================="
echo "Testing Intermediate Batch Sizes"
echo "INPUT SIZE: 448 (4x more tokens than 224)"
echo "Validating hypothesis: TR faster at large input size"
echo "========================================="
echo ""

# Write CSV header
echo "batch_size,model,tp,latency_ms,flops,memory" > $OUTPUT_FILE

# Test batch sizes: 1, 2, 4, 8 (skip 16 - may OOM)
for BS in 1 2 4 8; do
    echo "--- Testing Batch Size: $BS (baseline, is=448) ---"

    # Baseline (no TR)
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model deit_base_patch16_224.fb_in1k \
        --input-size 448 \
        --debugging \
        --batch-size $BS \
        --test_multiple 1 \
        --dummy_loader 2>&1 | grep "^cub_" | tail -1 > /tmp/result.txt

    if [ -s /tmp/result.txt ]; then
        TP=$(cat /tmp/result.txt | cut -d',' -f2)
        LAT=$(cat /tmp/result.txt | cut -d',' -f3)
        FLOPS=$(cat /tmp/result.txt | cut -d',' -f5)
        MEM=$(cat /tmp/result.txt | cut -d',' -f6)
        echo "$BS,baseline,$TP,$LAT,$FLOPS,$MEM" >> $OUTPUT_FILE
        echo "  Baseline: $TP img/s, $LAT ms, $FLOPS GFLOPs, $MEM MB"
    else
        echo "  ERROR: No output captured"
        echo "$BS,baseline,,,," >> $OUTPUT_FILE
    fi

    echo ""
    echo "--- Testing Batch Size: $BS (with TR, is=448) ---"

    # With TR
    python speed_test.py \
        --cfg configs/cub_ft_weakaugs.yaml \
        --model topk_deit_base_patch16_224.fb_in1k \
        --input-size 448 \
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
        MEM=$(cat /tmp/result.txt | cut -d',' -f6)
        echo "$BS,with_tr,$TP,$LAT,$FLOPS,$MEM" >> $OUTPUT_FILE
        echo "  With TR: $TP img/s, $LAT ms, $FLOPS GFLOPs, $MEM MB"
    else
        echo "  ERROR: No output captured"
        echo "$BS,with_tr,,,," >> $OUTPUT_FILE
    fi

    echo ""
done

echo "========================================="
echo "Batch size sweep (is=448) complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "========================================="
echo ""
echo "Comparing with is=224 results..."
echo ""

# Quick comparison analysis
python3 << 'EOF'
import pandas as pd

print("=" * 80)
print("COMPARISON: INPUT SIZE 224 vs 448")
print("=" * 80)
print()

# Load both datasets
try:
    df_224 = pd.read_csv('profiling_results/batch_size_sweep_results.csv')
    df_448 = pd.read_csv('profiling_results/batch_size_sweep_448_results.csv')

    print("Input Size 224 (Small - TR Expected to be SLOWER):")
    print("-" * 80)
    for bs in [1, 2, 4, 8]:
        base = df_224[(df_224['batch_size']==bs) & (df_224['model']=='baseline')]
        tr = df_224[(df_224['batch_size']==bs) & (df_224['model']=='with_tr')]
        if not base.empty and not tr.empty:
            speedup = float(tr['tp'].iloc[0]) / float(base['tp'].iloc[0])
            result = "✓ FASTER" if speedup > 1.0 else "✗ SLOWER"
            print(f"  bs={bs:2d}: {speedup:.3f}x {result}")

    print()
    print("Input Size 448 (Large - TR Expected to be FASTER):")
    print("-" * 80)
    for bs in [1, 2, 4, 8]:
        base = df_448[(df_448['batch_size']==bs) & (df_448['model']=='baseline')]
        tr = df_448[(df_448['batch_size']==bs) & (df_448['model']=='with_tr')]
        if not base.empty and not tr.empty:
            speedup = float(tr['tp'].iloc[0]) / float(base['tp'].iloc[0])
            result = "✓ FASTER" if speedup > 1.0 else "✗ SLOWER"
            pct = (speedup - 1) * 100
            print(f"  bs={bs:2d}: {speedup:.3f}x {result} ({pct:+.1f}%)")

    print("=" * 80)

except Exception as e:
    print(f"Error during analysis: {e}")
EOF
