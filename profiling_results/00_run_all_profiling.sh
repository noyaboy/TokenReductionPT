#!/bin/bash
# Master script to run all profiling experiments
# Run this on a CUDA-enabled machine

echo "========================================================"
echo "Token Reduction Performance Profiling - Full Suite"
echo "========================================================"
echo ""
echo "This will run:"
echo "  1. Nsys profiling (4 configurations)"
echo "  2. NCU deep kernel analysis (2 configurations)"
echo "  3. Batch size sweep (5 batch sizes × 2 configs)"
echo ""
echo "Estimated time: 30-60 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Phase 1: Nsys Timeline Profiling
echo ""
echo "========== PHASE 1: TIMELINE PROFILING (nsys) =========="
echo ""

echo "1/4: Baseline bs=1..."
bash 01_profile_baseline_bs1.sh

echo ""
echo "2/4: With TR bs=1..."
bash 02_profile_with_tr_bs1.sh

echo ""
echo "3/4: Baseline bs=8..."
bash 03_profile_baseline_bs8.sh

echo ""
echo "4/4: With TR bs=8..."
bash 04_profile_with_tr_bs8.sh

# Phase 2: NCU Deep Kernel Analysis
echo ""
echo "========== PHASE 2: KERNEL ANALYSIS (ncu) =========="
echo ""

echo "1/2: Baseline detailed..."
bash 05_ncu_baseline_bs1.sh

echo ""
echo "2/2: With TR detailed..."
bash 06_ncu_with_tr_bs1.sh

# Phase 3: Batch Size Sweep
echo ""
echo "========== PHASE 3: BATCH SIZE SWEEP =========="
echo ""

bash 07_test_intermediate_batches.sh

# Summary
echo ""
echo "========================================================"
echo "ALL PROFILING COMPLETE!"
echo "========================================================"
echo ""
echo "Generated files:"
echo "  - nsys_traces/*.nsys-rep (timeline traces)"
echo "  - ncu_reports/*.ncu-rep (kernel analysis)"
echo "  - batch_size_sweep_results.csv (batch sweep data)"
echo ""
echo "Next steps:"
echo "  1. Review batch_size_sweep_results.csv for crossover point"
echo "  2. Open nsys traces in Nsight Systems GUI"
echo "  3. Open ncu reports in Nsight Compute GUI"
echo "  4. Run analysis: bash 08_analyze_results.sh"
echo ""
