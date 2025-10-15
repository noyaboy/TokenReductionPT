# Token Reduction Profiling Guide

## Quick Start

On a CUDA-enabled machine with GPU access:

```bash
cd profiling_results
bash 00_run_all_profiling.sh  # Runs all experiments
bash 08_analyze_results.sh     # Analyzes collected data
```

## Problem Overview

Token Reduction (TR) exhibits paradoxical performance behavior:

| Configuration | Without TR | With TR (0.1 keep) | Result |
|--------------|------------|-------------------|---------|
| **bs=1, is=224** | 169.7 img/s, 17.57 GF | 140.9 img/s, 5.55 GF | ❌ 68% ↓FLOPs but 17% SLOWER |
| **bs=8, is=448** | 108.7 img/s, 78.52 GF | 321.3 img/s, 24.73 GF | ✅ 69% ↓FLOPs and 3x FASTER |

**Goal**: Understand why TR overhead dominates at small batch sizes despite drastically reducing computation.

## Profiling Scripts

### Phase 1: Timeline Profiling (nsys)

Captures system-wide execution traces showing:
- Kernel launch patterns and counts
- CPU-GPU synchronization
- Memory transfers
- Overall timeline visualization

**Scripts:**
- `01_profile_baseline_bs1.sh` - Baseline without TR (bs=1, is=224)
- `02_profile_with_tr_bs1.sh` - With TR (bs=1, is=224, keep=0.1)
- `03_profile_baseline_bs8.sh` - Baseline large batch (bs=8, is=448)
- `04_profile_with_tr_bs8.sh` - With TR large batch (bs=8, is=448, keep=0.1)

**Outputs:** `nsys_traces/*.nsys-rep`

**Analysis:**
```bash
# Command-line stats
nsys stats --report cuda_gpu_kern_sum nsys_traces/baseline_bs1_is224.nsys-rep

# Or open in Nsight Systems GUI for visualization
# Compare:
#   - Number of kernel launches (TR adds topk, gather, indexing kernels)
#   - Kernel durations
#   - CPU overhead between kernels
#   - Memory operation patterns
```

### Phase 2: Deep Kernel Analysis (ncu)

Profiles individual kernels in detail:
- SM occupancy and efficiency
- Memory vs compute bound classification
- Achieved vs theoretical bandwidth
- Warp execution efficiency

**Scripts:**
- `05_ncu_baseline_bs1.sh` - Deep analysis baseline
- `06_ncu_with_tr_bs1.sh` - Deep analysis with TR

**Outputs:** `ncu_reports/*.ncu-rep`

**Analysis:**
Open in Nsight Compute GUI and compare:

1. **Baseline Attention/MLP kernels:**
   - SM Efficiency
   - Memory Throughput
   - Occupancy levels
   - Compute vs Memory bound

2. **TR-specific kernels (topk, gather, index):**
   - Are they memory-bound?
   - Low occupancy due to small problem size?
   - Overhead vs actual work done

### Phase 3: Batch Size Sweep

Systematically tests bs=1,2,4,8,16 to find crossover point.

**Script:** `07_test_intermediate_batches.sh`

**Output:** `batch_size_sweep_results.csv`

**Analysis:**
Script automatically generates comparison table showing:
- Throughput comparison at each batch size
- Speedup factor (TR / baseline)
- Crossover point where TR becomes beneficial
- FLOPs reduction verification

## Expected Findings

### Hypothesis 1: Kernel Launch Overhead
At small batch sizes:
- TR adds many additional kernels (topk, gather, scatter operations)
- Fixed kernel launch overhead (~5-10μs per launch) dominates
- Baseline has fewer, simpler kernel launches
- **Evidence**: Count kernel launches in nsys traces

### Hypothesis 2: Low GPU Utilization
Small batch means small problem size:
- Insufficient parallelism to saturate GPU
- Low SM occupancy (<50%)
- Memory-bound operations with small transfers
- **Evidence**: Check SM efficiency and occupancy in ncu reports

### Hypothesis 3: Dynamic Operations Cost
TR involves dynamic operations:
- `torch.topk`: Selection with sorting/heap operations
- `torch.gather`: Irregular memory access patterns
- Index-based operations: Poor cache locality
- **Evidence**: Profile TR-specific kernels, check memory patterns

### Hypothesis 4: Memory Overhead
TR operations may have poor memory characteristics:
- Gather operations: scattered reads (bad coalescing)
- Small tensor sizes: Unable to amortize memory latency
- Additional intermediate tensors
- **Evidence**: Check memory throughput and transaction efficiency

## Key Metrics to Extract

### From nsys (Timeline Analysis):
1. **Kernel Count Comparison**
   ```
   Baseline: ~X kernels per forward pass
   With TR: ~Y kernels per forward pass
   Added overhead: (Y-X) * ~10μs kernel launch overhead
   ```

2. **Total GPU Time**
   ```
   Compare sum of all kernel durations
   Look for CPU gaps (synchronization overhead)
   ```

3. **Memory Operations**
   ```
   D2H/H2D transfers
   Peer-to-peer copies
   ```

### From ncu (Kernel Analysis):

1. **SM Efficiency**
   ```
   Baseline: Should be >60% for GEMM operations
   TR ops: Check if topk/gather have <30% efficiency
   ```

2. **Occupancy**
   ```
   High batch: >70% occupancy → good utilization
   Low batch: <40% occupancy → underutilization
   ```

3. **Memory Bound vs Compute Bound**
   ```
   Baseline attention: Likely compute-bound at large batch
   TR gather ops: Likely memory-bound due to irregular access
   ```

4. **Achieved Bandwidth**
   ```
   Compare against theoretical peak
   Low % means inefficient memory patterns
   ```

## Analysis Workflow

1. **Run Experiments**
   ```bash
   bash 00_run_all_profiling.sh
   ```

2. **Quick Analysis**
   ```bash
   bash 08_analyze_results.sh
   ```

3. **Deep Dive - Timeline**
   - Open `nsys_traces/baseline_bs1_is224.nsys-rep` in Nsight Systems
   - Open `nsys_traces/with_tr_bs1_is224.nsys-rep` in Nsight Systems
   - Compare side-by-side:
     - CUDA HW row: Look at GPU utilization patterns
     - CUDA API row: Count kernel launches
     - Zoom into a single forward pass
     - Measure total time and gaps

4. **Deep Dive - Kernels**
   - Open `ncu_reports/baseline_bs1_detailed.ncu-rep` in Nsight Compute
   - Open `ncu_reports/with_tr_bs1_detailed.ncu-rep` in Nsight Compute
   - Focus on:
     - Top 10 most expensive kernels
     - TR-specific kernels (topk, gather, index_select)
     - Check "Details" page for occupancy and bottleneck analysis

5. **Document Findings**
   - Create `findings_report.md` in this directory
   - Include screenshots from nsys/ncu
   - Quantify overhead sources
   - Propose optimizations

## Expected Optimizations

Based on findings, potential solutions:

### If Kernel Launch Overhead Dominates:
- **Kernel Fusion**: Fuse topk + gather into single operation
- **Conditional TR**: Disable TR for bs < threshold
- **Vectorized Operations**: Batch multiple TR operations

### If Low Occupancy is the Issue:
- **Delayed TR**: Apply TR less frequently at small batch
- **Different Reduction Strategy**: Use different method for small batches
- **Increase Effective Batch**: Accumulate multiple samples before TR

### If Memory Patterns are Poor:
- **Optimized Gather**: Use specialized gather kernels
- **Reordering**: Reorganize tensor layout for better coalescing
- **Caching**: Cache selected tokens to avoid repeated gathering

## Troubleshooting

**CUDA not available:**
- Verify: `nvidia-smi`
- Check PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure CUDA version matches PyTorch compilation

**Profiling tools not found:**
```bash
which nsys  # Should return /usr/local/bin/nsys or similar
which ncu   # Should return /usr/bin/ncu or similar
```

**Out of memory during profiling:**
- Reduce `--launch-count` in ncu scripts
- Profile fewer kernels with `--kernel-regex`
- Use smaller test_multiple value

## Next Steps After Analysis

1. **Document Root Cause**
   - Write up findings with evidence
   - Include profiling screenshots
   - Quantify each overhead source

2. **Implement Optimizations**
   - Based on findings, implement targeted fixes
   - Re-profile to validate improvements

3. **Propose Adaptive Strategy**
   - Automatic batch size detection
   - Conditional TR based on workload characteristics
   - Runtime decision making

4. **Paper/Report**
   - This analysis makes excellent empirical material
   - Shows deep understanding of GPU performance
   - Demonstrates systematic debugging methodology
