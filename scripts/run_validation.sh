#!/bin/bash
# Run validation script for Kalshi demo environment
# Usage: ./run_validation.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/validation_output"
MIN_TRADES=100
MIN_DAYS=7

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --min-trades)
            MIN_TRADES="$2"
            shift 2
            ;;
        --min-days)
            MIN_DAYS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --backtest-results)
            BACKTEST_RESULTS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --min-trades N       Minimum trades to validate (default: 100)"
            echo "  --min-days N         Minimum days to validate (default: 7)"
            echo "  --output-dir PATH    Output directory for reports"
            echo "  --backtest-results PATH  Path to backtest results JSON"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check for environment variables
if [[ -z "${KALSHI_API_KEY_ID}" ]]; then
    echo "Warning: KALSHI_API_KEY_ID not set"
fi

if [[ -z "${KALSHI_PRIVATE_KEY_PATH}" ]]; then
    echo "Warning: KALSHI_PRIVATE_KEY_PATH not set"
fi

echo "====================================="
echo "Kalshi Demo Validation"
echo "====================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Minimum trades: ${MIN_TRADES}"
echo "Minimum days: ${MIN_DAYS}"
echo ""

# Build command
CMD="python3 -m src.validation"
CMD="${CMD} --min-trades ${MIN_TRADES}"
CMD="${CMD} --min-days ${MIN_DAYS}"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"

if [[ -n "${BACKTEST_RESULTS}" ]]; then
    CMD="${CMD} --backtest-results ${BACKTEST_RESULTS}"
fi

if [[ -n "${KALSHI_API_KEY_ID}" ]]; then
    CMD="${CMD} --api-key ${KALSHI_API_KEY_ID}"
fi

if [[ -n "${KALSHI_PRIVATE_KEY_PATH}" ]]; then
    CMD="${CMD} --private-key ${KALSHI_PRIVATE_KEY_PATH}"
fi

echo "Running: ${CMD}"
echo ""

# Run validation
cd "${PROJECT_DIR}"
if ${CMD}; then
    echo ""
    echo "====================================="
    echo "Validation completed successfully!"
    echo "====================================="
    echo ""
    echo "Reports saved to: ${OUTPUT_DIR}"
    
    # Show latest report
    LATEST_REPORT=$(ls -t "${OUTPUT_DIR}"/validation_summary_*.txt 2>/dev/null | head -1)
    if [[ -n "${LATEST_REPORT}" ]]; then
        echo ""
        cat "${LATEST_REPORT}"
    fi
    
    exit 0
else
    echo ""
    echo "====================================="
    echo "Validation failed!"
    echo "====================================="
    exit 1
fi
