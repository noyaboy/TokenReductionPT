#!/bin/bash
# Quick test to verify CUDA and run a simple speed test
# This helps identify issues before running full profiling suite

echo "========================================="
echo "Quick Test: CUDA + Basic Speed Test"
echo "========================================="
echo ""

# Test 1: CUDA availability
echo "--- Test 1: CUDA Check ---"
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    nvidia-smi
else
    echo "✗ nvidia-smi not found - CUDA drivers may not be installed"
    exit 1
fi

echo ""
echo "--- Test 2: PyTorch CUDA ---"
echo ""
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device capability: {torch.cuda.get_device_capability(0)}')
else:
    print('WARNING: CUDA not available to PyTorch!')
    import sys
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ PyTorch cannot access CUDA!"
    echo "This system needs a CUDA-compatible PyTorch installation."
    exit 1
fi

echo ""
echo "--- Test 3: Quick Speed Test (Baseline) ---"
echo ""
echo "Running minimal baseline test (bs=1, warmup only)..."
python speed_test.py \
    --cfg configs/cub_ft_weakaugs.yaml \
    --model deit_base_patch16_224.fb_in1k \
    --input-size 224 \
    --debugging \
    --batch-size 1 \
    --test_multiple 1 \
    --warmup_iters 10 \
    --dummy_loader \
    2>&1 | grep -E "(CUDA|run_name|tp,|Creating|Error|Traceback)" | tail -20

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ ALL TESTS PASSED"
    echo "System is ready for profiling!"
    echo "========================================="
    echo ""
    echo "To run profiling:"
    echo "  bash run_with_logging.sh"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f profiling.log"
    exit 0
else
    echo ""
    echo "========================================="
    echo "✗ TEST FAILED"
    echo "Cannot run speed_test.py"
    echo "========================================="
    exit 1
fi
