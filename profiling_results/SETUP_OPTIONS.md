# Setup Options for Profiling

## Current Issue

✗ **CUDA Version Mismatch**
- System has: CUDA 11.4 (driver 470.256.02)
- PyTorch compiled for: CUDA 12.4
- Result: PyTorch cannot access GPU

## Option 1: Fix PyTorch on This Machine (Quick, ~5 minutes)

Install PyTorch compatible with CUDA 11.4:

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision -y

# Install CUDA 11.8 compatible version (works with 11.4 drivers)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run quick test
cd profiling_results
bash quick_test.sh
```

If this works, proceed with profiling on this machine.

## Option 2: Use Edwin's Machine

Copy repository to Edwin's CUDA-enabled machine:

```bash
# On Edwin's machine
git clone git@github.com:noyaboy/TokenReductionPT.git
cd TokenReductionPT/profiling_results

# Run quick test
bash quick_test.sh

# If passes, run profiling
bash run_with_logging.sh

# In another terminal, monitor progress
tail -f profiling.log
```

## Option 3: Copy This Directory to Edwin's Machine

If you want to run from this exact location:

```bash
# From this machine
rsync -avz --progress /home/noah/project/TokenReductionPT/ \
    edwin@machine:/path/to/TokenReductionPT/

# Then SSH to Edwin's machine and run
ssh edwin@machine
cd /path/to/TokenReductionPT/profiling_results
bash quick_test.sh
bash run_with_logging.sh &
tail -f profiling.log
```

## Recommended: Option 1 (Fix PyTorch Here)

This is fastest and you can monitor everything locally.
