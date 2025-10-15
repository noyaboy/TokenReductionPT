#!/bin/bash
# Wrapper script to run profiling with detailed logging and progress monitoring
# Usage: bash run_with_logging.sh [script_name]
#   If no script specified, runs 00_run_all_profiling.sh
#   Monitor with: tail -f profiling.log

# Configuration
LOG_FILE="profiling.log"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
}

# Start logging
log_section "PROFILING SESSION STARTED"
log "Log file: $LOG_FILE"
log "Monitor with: tail -f $LOG_FILE"

# Check CUDA availability
log ""
log "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi >> "$LOG_FILE" 2>&1
    CUDA_AVAILABLE=true
    log "✓ CUDA is available"
else
    log "✗ WARNING: nvidia-smi not found"
    CUDA_AVAILABLE=false
fi

# Check PyTorch CUDA
log ""
log "Checking PyTorch CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1 | tee -a "$LOG_FILE"

# Determine which script to run
SCRIPT_TO_RUN="${1:-00_run_all_profiling.sh}"

if [ ! -f "$SCRIPT_TO_RUN" ]; then
    log "ERROR: Script not found: $SCRIPT_TO_RUN"
    exit 1
fi

log ""
log "Running: $SCRIPT_TO_RUN"
log ""

# Run the script with output to both console and log file
bash "$SCRIPT_TO_RUN" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# Log completion
log ""
if [ $EXIT_CODE -eq 0 ]; then
    log_section "✓ PROFILING COMPLETED SUCCESSFULLY"
else
    log_section "✗ PROFILING FAILED (Exit code: $EXIT_CODE)"
fi

log "Total log file size: $(du -h $LOG_FILE | cut -f1)"
log "Session ended at: $(date +"%Y-%m-%d %H:%M:%S")"

exit $EXIT_CODE
