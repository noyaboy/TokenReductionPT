#!/bin/bash
# Fix PyTorch for CUDA 11.4 compatibility
# This installs PyTorch 2.0.1 with CUDA 11.8 support (works with 11.4 drivers)

echo "========================================="
echo "Fixing PyTorch for CUDA 11.4"
echo "========================================="
echo ""

# Check current status
echo "Current PyTorch version:"
python -c "import torch; print(torch.__version__)" 2>&1

echo ""
echo "Current CUDA availability:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>&1

echo ""
read -p "Uninstall current PyTorch and install CUDA 11.8 compatible version? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Uninstalling current PyTorch..."
pip uninstall torch torchvision -y

echo ""
echo "Installing PyTorch 2.0.1 with CUDA 11.8 support..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "========================================="
echo "Verification"
echo "========================================="
echo ""

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print('')
    print('✓ SUCCESS! PyTorch can now access GPU')
else:
    print('')
    print('✗ Still not working - may need driver update')
    import sys
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Next steps:"
    echo "  bash quick_test.sh          # Verify with speed test"
    echo "  bash run_with_logging.sh    # Run all profiling"
    echo "  tail -f profiling.log       # Monitor progress"
    echo "========================================="
else
    echo ""
    echo "Fix failed. Consider running on Edwin's machine instead."
fi
