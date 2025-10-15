#!/bin/bash
# Analyze profiling results and extract key metrics

echo "========================================="
echo "Analyzing Profiling Results"
echo "========================================="

echo ""
echo "--- 1. NSYS Statistics: Baseline vs TR (bs=1) ---"
echo ""

if [ -f "nsys_traces/baseline_bs1_is224.nsys-rep" ]; then
    echo "Baseline (no TR) - bs=1, is=224:"
    nsys stats --report cuda_gpu_kern_sum nsys_traces/baseline_bs1_is224.nsys-rep | head -20
    echo ""
    BASELINE_KERNELS=$(nsys stats --report cuda_gpu_kern_sum nsys_traces/baseline_bs1_is224.nsys-rep | grep -c "void ")
    echo "Total unique kernels (baseline): $BASELINE_KERNELS"
else
    echo "Baseline trace not found. Run: bash 01_profile_baseline_bs1.sh"
fi

echo ""
echo "---"
echo ""

if [ -f "nsys_traces/with_tr_bs1_is224.nsys-rep" ]; then
    echo "With TR - bs=1, is=224:"
    nsys stats --report cuda_gpu_kern_sum nsys_traces/with_tr_bs1_is224.nsys-rep | head -20
    echo ""
    TR_KERNELS=$(nsys stats --report cuda_gpu_kern_sum nsys_traces/with_tr_bs1_is224.nsys-rep | grep -c "void ")
    echo "Total unique kernels (with TR): $TR_KERNELS"
else
    echo "TR trace not found. Run: bash 02_profile_with_tr_bs1.sh"
fi

echo ""
echo "========================================="
echo "--- 2. Key Findings from Batch Sweep ---"
echo "========================================="

if [ -f "batch_size_sweep_results.csv" ]; then
    python3 - << 'EOF'
import pandas as pd
import sys

try:
    df = pd.read_csv('batch_size_sweep_results.csv')
    baseline = df[df['model'] == 'baseline'].set_index('batch_size')
    with_tr = df[df['model'] == 'with_tr'].set_index('batch_size')

    print("\nThroughput Comparison:")
    print("=" * 70)
    print(f"{'BS':>4} | {'Baseline (img/s)':>18} | {'With TR (img/s)':>15} | {'Speedup':>8} | {'Better?':>8}")
    print("-" * 70)

    crossover_bs = None
    for bs in sorted(baseline.index):
        base_tp = float(baseline.loc[bs, 'tp'])
        tr_tp = float(with_tr.loc[bs, 'tp'])
        speedup = tr_tp / base_tp
        better = "✓ YES" if speedup > 1.0 else "✗ NO"

        if speedup > 1.0 and crossover_bs is None:
            crossover_bs = bs

        print(f"{bs:4d} | {base_tp:18.2f} | {tr_tp:15.2f} | {speedup:7.3f}x | {better:>8}")

    print("=" * 70)

    if crossover_bs:
        print(f"\n✓ CROSSOVER POINT: Batch size {crossover_bs}")
        print(f"  TR becomes beneficial at BS >= {crossover_bs}")
    else:
        print("\n✗ No crossover point found in tested range")
        print("  TR is slower at all tested batch sizes")

    print("\nFLOPs Reduction:")
    print("=" * 50)
    for bs in sorted(baseline.index):
        base_flops = float(baseline.loc[bs, 'flops'])
        tr_flops = float(with_tr.loc[bs, 'flops'])
        reduction = (1 - tr_flops / base_flops) * 100
        print(f"BS {bs:2d}: {base_flops:6.2f} → {tr_flops:6.2f} GFLOPs ({reduction:.1f}% reduction)")
    print("=" * 50)

except FileNotFoundError:
    print("ERROR: batch_size_sweep_results.csv not found")
    sys.exit(1)
EOF
else
    echo "Batch sweep results not found. Run: bash 07_test_intermediate_batches.sh"
fi

echo ""
echo "========================================="
echo "--- 3. NCU Analysis Summary ---"
echo "========================================="

if [ -f "ncu_reports/baseline_bs1_detailed.ncu-rep" ]; then
    echo "Baseline kernel analysis available"
    echo "Open in Nsight Compute GUI for detailed metrics"
else
    echo "NCU baseline report not found. Run: bash 05_ncu_baseline_bs1.sh"
fi

if [ -f "ncu_reports/with_tr_bs1_detailed.ncu-rep" ]; then
    echo "TR kernel analysis available"
    echo "Open in Nsight Compute GUI for detailed metrics"
else
    echo "NCU TR report not found. Run: bash 06_ncu_with_tr_bs1.sh"
fi

echo ""
echo "========================================="
echo "Analysis complete!"
echo "========================================="
echo ""
echo "Manual analysis steps:"
echo "  1. Open .nsys-rep files in Nsight Systems"
echo "     - Compare kernel counts and durations"
echo "     - Look for CPU-GPU sync overhead"
echo "     - Check memory transfer patterns"
echo ""
echo "  2. Open .ncu-rep files in Nsight Compute"
echo "     - Check SM Efficiency"
echo "     - Compare Memory vs Compute bound"
echo "     - Look at TR operation (topk, gather) efficiency"
echo ""
