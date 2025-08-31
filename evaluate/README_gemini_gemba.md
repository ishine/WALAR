# Gemini GEMBA Translation Evaluation

This module provides translation quality evaluation using Google's Gemini API in the GEMBA (Generative Evaluation Metrics Based on Assessment) style.

## What is GEMBA?

GEMBA is a framework for evaluating machine translation quality using generative AI models. Instead of traditional metrics like BLEU or COMET that rely on reference translations, GEMBA asks the AI model to directly assess translation quality across multiple dimensions:

- **Accuracy**: How accurately the translation conveys the source meaning
- **Fluency**: How natural and fluent the target language text is
- **Completeness**: Whether the translation includes all information without additions/omissions
- **Overall Quality**: Comprehensive assessment considering all aspects

## Features

- **Multi-dimensional scoring**: Evaluates translations on 4 key quality aspects
- **Detailed explanations**: Provides reasoning for each score
- **Batch processing**: Efficiently evaluates multiple translations
- **Robust error handling**: Fallback parsing and retry mechanisms
- **Comprehensive metrics**: Calculates mean, std, min, max, and median scores
- **JSON output**: Structured results for further analysis

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_gemini.txt
```

2. Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

### Command Line Interface

```bash
python evaluate/gemini_gemba.py \
    --api_key "your_api_key_here" \
    --lang_pair "en-zh" \
    --data_file "data/translations.jsonl" \
    --output_file "results/gemini_gemba_en_zh.json"
```

### Shell Script

```bash
chmod +x scripts/evaluate_gemini_gemba.sh
./scripts/evaluate_gemini_gemba.sh "your_api_key_here" "en-zh" "data/translations.jsonl"
```

### Python API

```python
from evaluate.gemini_gemba import GeminiGembaEvaluator

# Initialize evaluator
evaluator = GeminiGembaEvaluator("your_api_key", "gemini-1.5-flash")

# Evaluate a single translation
result = evaluator.evaluate_single_translation(
    source="Hello, how are you?",
    reference="你好，你好吗？",
    prediction="你好，你好吗？",
    src_lang="en",
    tgt_lang="zh"
)

# Evaluate a batch
data = [
    {"src": "Hello", "ref": "你好", "pred": "你好"},
    {"src": "Goodbye", "ref": "再见", "pred": "再见"}
]
results = evaluator.evaluate_batch(data, "en", "zh")
```

## Input Data Format

The script expects a JSONL file where each line contains a JSON object with:

```json
{"src": "source text", "ref": "reference translation", "pred": "machine translation"}
```

Alternative field names are also supported:
- `source`/`reference`/`prediction`
- `src`/`ref`/`pred`

## Output Format

Results are saved as a JSON file with the following structure:

```json
{
  "metadata": {
    "model": "gemini-gemba",
    "timestamp": "2024-01-01 12:00:00",
    "total_samples": 100
  },
  "metrics": {
    "accuracy_scores_mean": 85.5,
    "accuracy_scores_std": 12.3,
    "fluency_scores_mean": 82.1,
    "fluency_scores_std": 15.7,
    "completeness_scores_mean": 88.9,
    "completeness_scores_std": 10.2,
    "overall_quality_scores_mean": 85.8,
    "overall_quality_scores_std": 13.1
  },
  "results": [
    {
      "source": "Hello, how are you?",
      "reference": "你好，你好吗？",
      "prediction": "你好，你好吗？",
      "evaluation": {
        "accuracy": {
          "score": 95,
          "explanation": "The translation accurately conveys the greeting and question."
        },
        "fluency": {
          "score": 90,
          "explanation": "Natural Chinese expression that sounds native."
        },
        "completeness": {
          "score": 100,
          "explanation": "All elements of the source text are translated."
        },
        "overall_quality": {
          "score": 95,
          "explanation": "Excellent translation with high accuracy and fluency."
        }
      }
    }
  ]
}
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--api_key` | Yes | - | Google Gemini API key |
| `--model_name` | No | `gemini-1.5-flash` | Gemini model to use |
| `--lang_pair` | Yes | - | Language pair (e.g., en-zh) |
| `--data_file` | Yes | - | Path to JSONL data file |
| `--output_file` | No | Auto-generated | Path to save results |
| `--batch_size` | No | 10 | Batch size for processing |
| `--max_retries` | No | 3 | Max retries for API calls |

## Supported Language Pairs

The script supports any language pair, with automatic language name mapping for common languages:

- English (en) → Chinese (zh)
- German (de) → English (en)
- French (fr) → Spanish (es)
- And many more...

## Rate Limiting

The script includes a 0.1-second delay between API calls to respect rate limits. For production use, you may need to adjust this based on your API quota.

## Error Handling

The script includes robust error handling:

1. **API failures**: Automatic retry with exponential backoff
2. **JSON parsing errors**: Fallback parsing using regex to extract scores
3. **Missing data**: Skips incomplete entries with warnings
4. **Rate limiting**: Built-in delays and retry mechanisms

## Examples

### Evaluate FLORES dataset
```bash
python evaluate/gemini_gemba.py \
    --api_key "$GEMINI_API_KEY" \
    --lang_pair "en-zh" \
    --data_file "data/flores_en_zh.jsonl" \
    --output_file "results/gemini_gemba_flores_en_zh.json"
```

### Evaluate WMT results
```bash
python evaluate/gemini_gemba.py \
    --api_key "$GEMINI_API_KEY" \
    --lang_pair "de-en" \
    --data_file "result/wmt24/metricX-xl/seg/de-en.jsonl" \
    --output_file "result/gemini_gemba/de-en/wmt24_metricX_xl.json"
```

## Performance Tips

1. **Use appropriate batch sizes**: Start with 10 and adjust based on your API quota
2. **Monitor API usage**: Check your Google AI Studio dashboard for usage statistics
3. **Parallel processing**: For large datasets, consider splitting into multiple files and running parallel evaluations
4. **Caching**: Results are saved incrementally, so you can resume interrupted evaluations

## Troubleshooting

### Common Issues

1. **API key errors**: Ensure your API key is valid and has sufficient quota
2. **Rate limiting**: Increase delays between calls if you hit rate limits
3. **JSON parsing errors**: Check that your input data is properly formatted
4. **Memory issues**: For very large datasets, process in smaller batches

### Getting Help

- Check the Google Generative AI documentation: https://ai.google.dev/docs
- Verify your API key and quota in Google AI Studio
- Ensure your input data follows the expected JSONL format

## License

This script is part of the qe-lr project and follows the same licensing terms. 