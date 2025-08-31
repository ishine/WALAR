#!/bin/bash

# Script to run Gemini text generation
# Usage: ./gemini_generate.sh <api_key> [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 <api_key> [options]"
    echo ""
    echo "Required arguments:"
    echo "  api_key     - Your Google Gemini API key"
    echo ""
    echo "Options:"
    echo "  --prompt <text>           - Single prompt to generate from"
    echo "  --input <file>            - Input file containing prompts"
    echo "  --output <file>           - Output file for results"
    echo "  --model <name>            - Gemini model to use (default: gemini-1.5-flash)"
    echo "  --max-tokens <number>     - Maximum tokens to generate (default: 1024)"
    echo "  --temperature <float>     - Generation temperature 0.0-1.0 (default: 0.7)"
    echo "  --format <format>         - Output format: json, jsonl, csv, txt (default: auto)"
    echo "  --delay <seconds>         - Delay between API calls (default: 0.1)"
    echo ""
    echo "Examples:"
    echo "  # Generate from single prompt"
    echo "  $0 'your_api_key' --prompt 'Write a short story about a robot.'"
    echo ""
    echo "  # Generate from file"
    echo "  $0 'your_api_key' --input prompts.jsonl --output results.json"
    echo ""
    echo "  # Generate with custom parameters"
    echo "  $0 'your_api_key' --input prompts.txt --temperature 0.9 --max-tokens 2048"
    echo ""
    echo "Supported input formats:"
    echo "  - JSONL: Each line is a JSON object with 'prompt' field"
    echo "  - CSV: CSV file with 'prompt' column (or first non-empty column)"
    echo "  - TXT: One prompt per line"
    echo "  - JSON: Array of prompts or object with prompt data"
}

# Check if API key is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

API_KEY="$1"
shift

# Parse command line arguments
PROMPT=""
INPUT_FILE=""
OUTPUT_FILE=""
MODEL_NAME="gemini-1.5-flash"
MAX_TOKENS="1024"
TEMPERATURE="0.7"
FORMAT=""
DELAY="0.1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$PROMPT" ] && [ -z "$INPUT_FILE" ]; then
    print_error "Either --prompt or --input must be specified"
    show_usage
    exit 1
fi

if [ -n "$PROMPT" ] && [ -n "$INPUT_FILE" ]; then
    print_error "Cannot specify both --prompt and --input"
    exit 1
fi

if [ -n "$INPUT_FILE" ] && [ ! -f "$INPUT_FILE" ]; then
    print_error "Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Set default output file if not provided
if [ -z "$OUTPUT_FILE" ]; then
    if [ -n "$PROMPT" ]; then
        OUTPUT_FILE="output/gemini_generated_$(date +%Y%m%d_%H%M%S).json"
    else
        BASE_NAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')
        OUTPUT_FILE="output/gemini_generated_${BASE_NAME}.json"
    fi
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

print_info "=========================================="
print_info "Gemini Text Generation"
print_info "=========================================="
print_info "Model: $MODEL_NAME"
if [ -n "$PROMPT" ]; then
    print_info "Mode: Single prompt"
    print_info "Prompt: ${PROMPT:0:50}..."
else
    print_info "Mode: Batch from file"
    print_info "Input file: $INPUT_FILE"
fi
print_info "Output file: $OUTPUT_FILE"
print_info "Max tokens: $MAX_TOKENS"
print_info "Temperature: $TEMPERATURE"
print_info "Delay: ${DELAY}s"
print_info "=========================================="

# Activate conda environment if available
if command -v conda &> /dev/null; then
    print_info "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "qe-rl"; then
        conda activate qe-rl
        print_success "Activated qe-rl environment"
    fi
fi

# Install required packages if not already installed
print_info "Checking dependencies..."
python -c "import google.generativeai" 2>/dev/null || {
    print_warning "Installing google-generativeai..."
    pip install google-generativeai
}

python -c "import tqdm" 2>/dev/null || {
    print_warning "Installing tqdm..."
    pip install tqdm
}

# Build command
CMD="python code/gemini_generate.py --api_key '$API_KEY' --model_name '$MODEL_NAME'"

if [ -n "$PROMPT" ]; then
    CMD="$CMD --prompt '$PROMPT'"
else
    CMD="$CMD --input_file '$INPUT_FILE'"
fi

CMD="$CMD --output_file '$OUTPUT_FILE' --max_tokens $MAX_TOKENS --temperature $TEMPERATURE --delay $DELAY"

if [ -n "$FORMAT" ]; then
    CMD="$CMD --format $FORMAT"
fi

# Run the generation
print_info "Starting generation..."
cd "$(dirname "$0")/.."

print_info "Running command: $CMD"
eval $CMD

if [ $? -eq 0 ]; then
    print_success "=========================================="
    print_success "Generation completed successfully!"
    print_success "Results saved to: $OUTPUT_FILE"
    print_success "=========================================="
else
    print_error "Generation failed!"
    exit 1
fi 