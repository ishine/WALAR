#!/bin/bash

# Script to run directory walker analysis
# Usage: ./walk_directory.sh <directory_path> [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

print_header() {
    echo -e "${CYAN}$1${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 <directory_path> [options]"
    echo ""
    echo "Required arguments:"
    echo "  directory_path  - Path to the directory to analyze"
    echo ""
    echo "Options:"
    echo "  --max-depth <number>      - Maximum depth to traverse"
    echo "  --follow-symlinks         - Follow symbolic links"
    echo "  --calculate-hashes        - Calculate MD5 hashes for files"
    echo "  --count-lines             - Count lines in text files"
    echo "  --output <file>           - Output file to save results"
    echo "  --format <format>         - Output format: json, txt (default: json)"
    echo "  --quiet                   - Suppress progress output"
    echo "  --help, -h                - Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic directory analysis"
    echo "  $0 /path/to/directory"
    echo ""
    echo "  # Analyze with depth limit and save results"
    echo "  $0 /path/to/directory --max-depth 3 --output analysis.json"
    echo ""
    echo "  # Full analysis with hashes and line counts"
    echo "  $0 /path/to/directory --calculate-hashes --count-lines --output full_analysis.json"
    echo ""
    echo "  # Quick analysis (quiet mode)"
    echo "  $0 /path/to/directory --quiet --output quick_analysis.json"
}

# Check if directory path is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

DIRECTORY_PATH="$1"
shift

# Parse command line arguments
MAX_DEPTH=""
FOLLOW_SYMLINKS=""
CALCULATE_HASHES=""
COUNT_LINES=""
OUTPUT_FILE=""
FORMAT="json"
QUIET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-depth)
            MAX_DEPTH="--max-depth $2"
            shift 2
            ;;
        --follow-symlinks)
            FOLLOW_SYMLINKS="--follow-symlinks"
            shift
            ;;
        --calculate-hashes)
            CALCULATE_HASHES="--calculate-hashes"
            shift
            ;;
        --count-lines)
            COUNT_LINES="--count-lines"
            shift
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --quiet)
            QUIET="--quiet"
            shift
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

# Validate directory path
if [ ! -d "$DIRECTORY_PATH" ]; then
    print_error "Directory '$DIRECTORY_PATH' does not exist or is not a directory!"
    exit 1
fi

# Resolve absolute path
DIRECTORY_PATH=$(realpath "$DIRECTORY_PATH")

# Set default output file if not provided
if [ -z "$OUTPUT_FILE" ]; then
    BASE_NAME=$(basename "$DIRECTORY_PATH")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="output/directory_analysis/${BASE_NAME}_${TIMESTAMP}.${FORMAT}"
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

print_header "=========================================="
print_header "Directory Walker Analysis"
print_header "=========================================="
print_info "Target directory: $DIRECTORY_PATH"
print_info "Output file: $OUTPUT_FILE"
print_info "Format: $FORMAT"

if [ -n "$MAX_DEPTH" ]; then
    print_info "Max depth: ${MAX_DEPTH#--max-depth }"
fi

if [ -n "$FOLLOW_SYMLINKS" ]; then
    print_info "Following symlinks: Yes"
fi

if [ -n "$CALCULATE_HASHES" ]; then
    print_info "Calculating hashes: Yes"
fi

if [ -n "$COUNT_LINES" ]; then
    print_info "Counting lines: Yes"
fi

if [ -n "$QUIET" ]; then
    print_info "Quiet mode: Yes"
fi

print_header "=========================================="

# Activate conda environment if available
if command -v conda &> /dev/null; then
    print_info "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "qe-rl"; then
        conda activate qe-rl
        print_success "Activated qe-rl environment"
    fi
fi

# Check if Python script exists
SCRIPT_PATH="code/directory_walker.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    print_error "Directory walker script not found at $SCRIPT_PATH"
    exit 1
fi

# Build command
CMD="python $SCRIPT_PATH \"$DIRECTORY_PATH\""

if [ -n "$MAX_DEPTH" ]; then
    CMD="$CMD $MAX_DEPTH"
fi

if [ -n "$FOLLOW_SYMLINKS" ]; then
    CMD="$CMD $FOLLOW_SYMLINKS"
fi

if [ -n "$CALCULATE_HASHES" ]; then
    CMD="$CMD $CALCULATE_HASHES"
fi

if [ -n "$COUNT_LINES" ]; then
    CMD="$CMD $COUNT_LINES"
fi

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output \"$OUTPUT_FILE\" --format $FORMAT"
fi

if [ -n "$QUIET" ]; then
    CMD="$CMD $QUIET"
fi

# Run the directory walker
print_info "Starting directory analysis..."
cd "$(dirname "$0")/.."

print_info "Running command: $CMD"
eval $CMD

if [ $? -eq 0 ]; then
    print_success "=========================================="
    print_success "Directory analysis completed successfully!"
    print_success "Results saved to: $OUTPUT_FILE"
    print_success "=========================================="
else
    print_error "Directory analysis failed!"
    exit 1
fi 