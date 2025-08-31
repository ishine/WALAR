#!/bin/bash

# Script to run Gemini GEMBA evaluation
# Usage: ./evaluate_gemini_gemba.sh <api_key> <lang_pair> <data_file> [output_file]

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <api_key> <lang_pair> <data_file> [output_file]"
    echo "Example: $0 'your_api_key_here' 'en-zh' 'data/translations.jsonl'"
    echo ""
    echo "Arguments:"
    echo "  api_key     - Your Google Gemini API key"
    echo "  lang_pair   - Language pair (e.g., en-zh, de-en, fr-es)"
    echo "  data_file   - Path to JSONL file with source, reference, and prediction"
    echo "  output_file - Optional: Path to save results (default: auto-generated)"
    exit 1
fi

API_KEY="$1"
LANG_PAIR="$2"
DATA_FILE="$3"
OUTPUT_FILE="$4"

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file '$DATA_FILE' not found!"
    exit 1
fi

# Set default output file if not provided
if [ -z "$OUTPUT_FILE" ]; then
    BASE_NAME=$(basename "$DATA_FILE" .jsonl)
    OUTPUT_FILE="result/gemini_gemba/${LANG_PAIR}/${BASE_NAME}.json"
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Gemini GEMBA Translation Evaluation"
echo "=========================================="
echo "Language pair: $LANG_PAIR"
echo "Data file: $DATA_FILE"
echo "Output file: $OUTPUT_FILE"
echo "=========================================="

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "qe-rl"; then
        conda activate qe-rl
        echo "Activated qe-rl environment"
    fi
fi

# Install required packages if not already installed
echo "Checking dependencies..."
python -c "import google.generativeai" 2>/dev/null || {
    echo "Installing google-generativeai..."
    pip install google-generativeai
}

python -c "import tqdm" 2>/dev/null || {
    echo "Installing tqdm..."
    pip install tqdm
}

# Run the evaluation
echo "Starting evaluation..."
cd "$(dirname "$0")/.."

python evaluate/gemini_gemba.py \
    --api_key "$API_KEY" \
    --lang_pair "$LANG_PAIR" \
    --data_file "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_name "gemini-1.5-flash"

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "==========================================" 