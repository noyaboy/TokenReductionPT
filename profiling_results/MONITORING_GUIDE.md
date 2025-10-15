# Real-Time Monitoring Guide

## Quick Start

```bash
# Terminal 1: Start profiling (runs in background)
bash run_with_logging.sh &

# Terminal 2: Monitor progress
tail -f profiling.log
```

## Monitoring Commands

### Watch the Full Log
```bash
tail -f profiling.log
```

### Watch Only Important Lines
```bash
# See progress updates, errors, and summaries
tail -f profiling.log | grep -E "(===|✓|✗|Progress|Complete|Error|run_name)"
```

### Watch Specific Phase
```bash
# Only nsys profiling
tail -f profiling.log | grep -i nsys

# Only ncu profiling
tail -f profiling.log | grep -i ncu

# Only batch sweep
tail -f profiling.log | grep -E "(Batch|bs=)"
```

### Check Current Status
```bash
# See last 50 lines
tail -50 profiling.log

# Check if still running
ps aux | grep "run_with_logging\|speed_test"

# See progress percentage (if running batch sweep)
grep "Testing Batch Size" profiling.log | tail -5
```

## Individual Script Monitoring

### Run Single Script with Logging
```bash
bash run_with_logging.sh 01_profile_baseline_bs1.sh &
tail -f profiling.log
```

### Run Without Logging (Direct Output)
```bash
bash 01_profile_baseline_bs1.sh
```

## Progress Indicators

You'll see output like:

```
[2025-10-15 18:45:23] ========================================
[2025-10-15 18:45:23] PHASE 1: TIMELINE PROFILING (nsys)
[2025-10-15 18:45:23] ========================================
[2025-10-15 18:45:24] 1/4: Baseline bs=1...
[2025-10-15 18:46:42] Profiling complete! Output saved to: nsys_traces/baseline_bs1_is224.nsys-rep
[2025-10-15 18:46:43] 2/4: With TR bs=1...
```

## Stopping Profiling

If you need to stop:

```bash
# Find the process
ps aux | grep run_with_logging

# Kill it (replace PID with actual process ID)
kill <PID>

# Or kill all related processes
pkill -f "run_with_logging\|speed_test"
```

## Checking Results During Run

While profiling is running, you can check partial results:

```bash
# See what's been generated so far
ls -lh nsys_traces/
ls -lh ncu_reports/

# Check batch sweep results
cat batch_size_sweep_results.csv
```

## Estimated Timeline

- **Phase 1 (nsys - 4 configs)**: ~10-15 minutes
  - Each configuration: 2-4 minutes

- **Phase 2 (ncu - 2 configs)**: ~15-20 minutes
  - Detailed kernel analysis is slower

- **Phase 3 (batch sweep)**: ~5-10 minutes
  - 5 batch sizes × 2 configs = 10 runs

**Total**: ~30-45 minutes

## Real-Time Analysis

### While Waiting, Analyze Completed Parts

```bash
# After nsys traces are done, analyze them
bash 08_analyze_results.sh

# Or check kernel counts manually
nsys stats --report cuda_gpu_kern_sum nsys_traces/baseline_bs1_is224.nsys-rep | head -20
```

## Troubleshooting

### Log File Too Big
```bash
# Only show last 100 lines
tail -100 profiling.log

# Clear log and restart
rm profiling.log
bash run_with_logging.sh &
tail -f profiling.log
```

### Process Stuck
```bash
# Check GPU usage
nvidia-smi

# Check if process is actually running
ps aux | grep speed_test

# Force kill if needed
pkill -9 -f speed_test
```

### Out of Memory
```bash
# Check GPU memory
nvidia-smi

# If OOM, reduce batch size in script
# Edit 03_profile_baseline_bs8.sh and change --batch-size 8 to --batch-size 4
```

## Multi-Terminal Setup

**Recommended layout:**

```
Terminal 1 (top-left):     Terminal 2 (top-right):
tail -f profiling.log      nvidia-smi -l 1

Terminal 3 (bottom):
ls -ltrh nsys_traces/ ncu_reports/
# Updates to see new files appear
```

## Alternative: Run in tmux

```bash
# Start tmux session
tmux new -s profiling

# Split window (Ctrl+b, then ")
# Top pane: run profiling
bash run_with_logging.sh

# Bottom pane: monitor (Ctrl+b, then o to switch)
tail -f profiling.log

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t profiling
```
