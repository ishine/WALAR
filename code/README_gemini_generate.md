# Gemini Text Generation

This module provides text generation capabilities using Google's Gemini API, supporting both single prompt generation and batch processing from files.

## Features

- **Single prompt generation**: Generate responses for individual prompts
- **Batch processing**: Process multiple prompts from various file formats
- **Multiple input formats**: Support for JSONL, CSV, TXT, and JSON files
- **Flexible output formats**: Save results as JSON, JSONL, CSV, or TXT
- **Configurable generation parameters**: Temperature, max tokens, top-p, top-k
- **Rate limiting**: Built-in delays to respect API rate limits
- **Error handling**: Robust error handling with detailed logging
- **Progress tracking**: Visual progress bars for batch operations

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_gemini.txt
```

2. Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

### Command Line Interface

#### Single Prompt Generation
```bash
python code/gemini_generate.py \
    --api_key "your_api_key_here" \
    --prompt "Write a short story about a robot." \
    --output_file "output/story.json"
```

#### Batch Generation from File
```bash
python code/gemini_generate.py \
    --api_key "your_api_key_here" \
    --input_file "data/prompts.jsonl" \
    --output_file "output/responses.json" \
    --temperature 0.8 \
    --max_tokens 2048
```

### Shell Script

#### Single Prompt
```bash
chmod +x scripts/gemini_generate.sh
./scripts/gemini_generate.sh "your_api_key_here" \
    --prompt "Explain quantum computing in simple terms."
```

#### Batch Processing
```bash
./scripts/gemini_generate.sh "your_api_key_here" \
    --input "data/prompts.txt" \
    --output "output/responses.jsonl" \
    --temperature 0.9 \
    --max-tokens 1500
```

### Python API

```python
from code.gemini_generate import GeminiGenerator

# Initialize generator
generator = GeminiGenerator("your_api_key", "gemini-1.5-flash")

# Generate single response
result = generator.generate_single_response(
    "What is machine learning?",
    {
        'max_tokens': 512,
        'temperature': 0.7
    }
)

# Generate batch responses
prompts = [
    {"prompt": "Explain AI", "id": 1},
    {"prompt": "What is deep learning?", "id": 2}
]
results = generator.generate_batch_responses(prompts, {
    'max_tokens': 1024,
    'temperature': 0.5
})
```

## Input File Formats

### JSONL (Recommended)
Each line contains a JSON object:
```jsonl
{"prompt": "What is artificial intelligence?", "id": 1, "category": "tech"}
{"prompt": "Explain machine learning", "id": 2, "category": "tech"}
{"prompt": "How does a neural network work?", "id": 3, "category": "tech"}
```

### CSV
CSV file with prompt column:
```csv
id,prompt,category
1,What is AI?,tech
2,Explain ML,tech
3,Neural networks,tech
```

### TXT
One prompt per line:
```
What is artificial intelligence?
Explain machine learning
How does a neural network work?
```

### JSON
Array of prompts or structured data:
```json
{
  "prompts": [
    {"prompt": "What is AI?", "id": 1},
    {"prompt": "Explain ML", "id": 2}
  ]
}
```

## Output Formats

### JSON (Default)
Structured output with metadata:
```json
{
  "metadata": {
    "model": "gemini-1.5-flash",
    "timestamp": "2024-01-01 12:00:00",
    "total_prompts": 3,
    "successful_generations": 3
  },
  "results": [
    {
      "prompt_id": 1,
      "prompt": "What is artificial intelligence?",
      "response": "Artificial intelligence (AI) is...",
      "success": true,
      "timestamp": "2024-01-01 12:00:00",
      "prompt_tokens": 8,
      "response_tokens": 45,
      "total_tokens": 53
    }
  ]
}
```

### JSONL
One result per line:
```jsonl
{"prompt_id": 1, "prompt": "What is AI?", "response": "AI is...", "success": true}
{"prompt_id": 2, "prompt": "Explain ML", "response": "ML is...", "success": true}
```

### CSV
Tabular format:
```csv
prompt_id,prompt,response,success,timestamp
1,What is AI?,AI is...,true,2024-01-01 12:00:00
2,Explain ML,ML is...,true,2024-01-01 12:00:00
```

### TXT
Human-readable format:
```
PROMPT:
What is artificial intelligence?

RESPONSE:
Artificial intelligence (AI) is a branch of computer science...

--------------------------------------------------------------------------------

PROMPT:
Explain machine learning

RESPONSE:
Machine learning is a subset of artificial intelligence...
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--api_key` | Yes | - | Google Gemini API key |
| `--model_name` | No | `gemini-1.5-flash` | Gemini model to use |
| `--input_file` | No* | - | Path to file containing prompts |
| `--prompt` | No* | - | Single prompt to generate from |
| `--output_file` | No | Auto-generated | Path to save results |
| `--max_tokens` | No | 1024 | Maximum tokens to generate |
| `--temperature` | No | 0.7 | Generation temperature (0.0-1.0) |
| `--top_p` | No | 1.0 | Top-p sampling parameter |
| `--top_k` | No | 40 | Top-k sampling parameter |
| `--delay` | No | 0.1 | Delay between API calls (seconds) |
| `--format` | No | Auto | Output format (json, jsonl, csv, txt) |

*Either `--input_file` or `--prompt` must be specified

## Generation Parameters

### Temperature
- **0.0**: Deterministic, consistent responses
- **0.3-0.7**: Balanced creativity and consistency
- **0.8-1.0**: More creative, varied responses

### Max Tokens
- **512**: Short, concise responses
- **1024**: Medium-length responses
- **2048+**: Long, detailed responses

### Top-p and Top-k
- **Top-p**: Nucleus sampling (0.1-1.0)
- **Top-k**: Limit vocabulary diversity (1-100)

## Examples

### Creative Writing
```bash
./scripts/gemini_generate.sh "$GEMINI_API_KEY" \
    --prompt "Write a science fiction story about time travel." \
    --temperature 0.9 \
    --max-tokens 2048
```

### Technical Explanation
```bash
./scripts/gemini_generate.sh "$GEMINI_API_KEY" \
    --prompt "Explain how transformers work in NLP." \
    --temperature 0.3 \
    --max-tokens 1024
```

### Batch Processing
```bash
# Create prompts file
echo "What is Python?" > prompts.txt
echo "Explain OOP" >> prompts.txt
echo "What are decorators?" >> prompts.txt

# Generate responses
./scripts/gemini_generate.sh "$GEMINI_API_KEY" \
    --input prompts.txt \
    --output "python_explanations.jsonl" \
    --temperature 0.5
```

### Custom Model
```bash
./scripts/gemini_generate.sh "$GEMINI_API_KEY" \
    --model "gemini-1.5-pro" \
    --prompt "Analyze this code for security vulnerabilities." \
    --temperature 0.1
```

## Performance Tips

1. **Batch size**: Process multiple prompts together for efficiency
2. **Rate limiting**: Adjust delays based on your API quota
3. **Model selection**: Use `gemini-1.5-flash` for speed, `gemini-1.5-pro` for quality
4. **Token limits**: Set appropriate max_tokens to avoid unnecessary API costs
5. **Parallel processing**: For large datasets, split into multiple files and run in parallel

## Error Handling

The script handles various error scenarios:

1. **API failures**: Automatic retry with exponential backoff
2. **Rate limiting**: Built-in delays and retry mechanisms
3. **Invalid input**: Graceful handling of malformed files
4. **Network issues**: Connection timeout and retry logic

## Troubleshooting

### Common Issues

1. **API key errors**: Verify your API key and quota
2. **Rate limiting**: Increase delays between calls
3. **File format errors**: Check input file format and encoding
4. **Memory issues**: Process large datasets in smaller batches

### Getting Help

- Check the Google Generative AI documentation: https://ai.google.dev/docs
- Verify your API key and quota in Google AI Studio
- Ensure your input data follows the expected format

## Integration with Existing Workflow

This script integrates well with your existing qe-lr project:

1. **Use generated text as training data** for your models
2. **Generate prompts for evaluation** using the GEMBA evaluator
3. **Create synthetic datasets** for testing and development
4. **Generate explanations** for model outputs

## License

This script is part of the qe-lr project and follows the same licensing terms. 