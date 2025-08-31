import os
import json
import argparse
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import google.generativeai as genai
from tqdm import tqdm
import numpy as np
from pathlib import Path

@dataclass
class GeminiGembaArguments:
    """
    Arguments for Gemini GEMBA evaluation.
    """
    api_key: str = field(
        metadata={"help": "Google Gemini API key"}
    )
    model_name: str = field(
        default="gemini-1.5-flash",
        metadata={"help": "Gemini model to use for evaluation"}
    )
    lang_pair: str = field(
        default="en-zh",
        metadata={"help": "Language pair for evaluation (e.g., en-zh)"}
    )
    data_file: str = field(
        default=None,
        metadata={"help": "Path to JSONL file containing source, reference, and prediction"}
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save evaluation results"}
    )
    batch_size: int = field(
        default=10,
        metadata={"help": "Batch size for API calls"}
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of retries for failed API calls"}
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for Gemini API calls"}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p for Gemini API calls"}
    )

class GeminiGembaEvaluator:
    """
    Evaluator using Gemini API for GEMBA-style translation quality assessment.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """Initialize the Gemini API client."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
    def create_gemba_prompt(self, source: str, reference: str, prediction: str, 
                           src_lang: str, tgt_lang: str) -> str:
        """
        Create a GEMBA-style prompt for translation quality assessment.
        
        GEMBA (Generative Evaluation Metrics Based on Assessment) evaluates translations
        by asking the model to assess quality aspects and provide scores.
        """
        # Language mapping for better prompts
        lang_names = {
            "en": "English", "zh": "Chinese", "de": "German", "fr": "French", 
            "es": "Spanish", "ru": "Russian", "ja": "Japanese", "ko": "Korean",
            "ar": "Arabic", "hi": "Hindi", "pt": "Portuguese", "it": "Italian"
        }
        
        src_lang_name = lang_names.get(src_lang, src_lang)
        tgt_lang_name = lang_names.get(tgt_lang, tgt_lang)
        
        prompt = f"""You are an expert translator and translation quality assessor. Please evaluate the quality of a machine translation from {src_lang_name} to {tgt_lang_name}.

Source text ({src_lang_name}):
{source}

Reference translation ({tgt_lang_name}):
{reference}

Machine translation ({tgt_lang_name}):
{prediction}

Please evaluate the machine translation on the following aspects and provide scores from 0-100 (where 100 is perfect):

1. **Accuracy**: How accurately does the translation convey the meaning of the source text?
2. **Fluency**: How natural and fluent is the target language text?
3. **Completeness**: How complete is the translation (no missing or added information)?
4. **Overall Quality**: Overall assessment considering all aspects above.

For each aspect, provide:
- A score (0-100)
- A brief explanation of your reasoning

Format your response as a JSON object with this exact structure:
{{
    "accuracy": {{
        "score": <score>,
        "explanation": "<explanation>"
    }},
    "fluency": {{
        "score": <score>,
        "explanation": "<explanation>"
    }},
    "completeness": {{
        "score": <score>,
        "explanation": "<explanation>"
    }},
    "overall_quality": {{
        "score": <score>,
        "explanation": "<explanation>"
    }}
}}

Respond only with the JSON object, no additional text."""
        
        return prompt
    
    def evaluate_single_translation(self, source: str, reference: str, prediction: str,
                                  src_lang: str, tgt_lang: str) -> Dict:
        """Evaluate a single translation using Gemini API."""
        prompt = self.create_gemba_prompt(source, reference, prediction, src_lang, tgt_lang)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    top_p=1.0,
                    max_output_tokens=1000,
                )
            )
            
            # Parse the response
            response_text = response.text.strip()
            
            # Try to extract JSON from the response
            try:
                # Find JSON content between curly braces
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    # Validate the structure
                    required_keys = ['accuracy', 'fluency', 'completeness', 'overall_quality']
                    for key in required_keys:
                        if key not in result or 'score' not in result[key]:
                            raise ValueError(f"Missing required key: {key}")
                    
                    return result
                else:
                    raise ValueError("No JSON content found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: try to extract scores using regex or manual parsing
                print(f"Failed to parse JSON response: {e}")
                print(f"Raw response: {response_text}")
                return self._fallback_parsing(response_text)
                
        except Exception as e:
            print(f"API call failed: {e}")
            return {
                "accuracy": {"score": 0, "explanation": f"API error: {str(e)}"},
                "fluency": {"score": 0, "explanation": f"API error: {str(e)}"},
                "completeness": {"score": 0, "explanation": f"API error: {str(e)}"},
                "overall_quality": {"score": 0, "explanation": f"API error: {str(e)}"}
            }
    
    def _fallback_parsing(self, response_text: str) -> Dict:
        """Fallback parsing when JSON parsing fails."""
        # Simple fallback: look for numbers that could be scores
        import re
        
        # Find all numbers in the response
        numbers = re.findall(r'\b(\d{1,3})\b', response_text)
        
        # If we have at least 4 numbers, use them as scores
        if len(numbers) >= 4:
            scores = [int(n) for n in numbers[:4]]
            return {
                "accuracy": {"score": scores[0], "explanation": "Fallback parsing"},
                "fluency": {"score": scores[1], "explanation": "Fallback parsing"},
                "completeness": {"score": scores[2], "explanation": "Fallback parsing"},
                "overall_quality": {"score": scores[3], "explanation": "Fallback parsing"}
            }
        
        # If no numbers found, return default scores
        return {
            "accuracy": {"score": 50, "explanation": "Fallback parsing failed"},
            "fluency": {"score": 50, "explanation": "Fallback parsing failed"},
            "completeness": {"score": 50, "explanation": "Fallback parsing failed"},
            "overall_quality": {"score": 50, "explanation": "Fallback parsing failed"}
        }
    
    def evaluate_batch(self, data: List[Dict], src_lang: str, tgt_lang: str) -> List[Dict]:
        """Evaluate a batch of translations."""
        results = []
        
        for item in tqdm(data, desc="Evaluating translations"):
            source = item.get('src', item.get('source', ''))
            reference = item.get('ref', item.get('reference', ''))
            prediction = item.get('pred', item.get('prediction', ''))
            
            if not all([source, reference, prediction]):
                print(f"Skipping item with missing data: {item}")
                continue
            
            # Add delay to respect API rate limits
            time.sleep(0.1)
            
            evaluation = self.evaluate_single_translation(
                source, reference, prediction, src_lang, tgt_lang
            )
            
            results.append({
                'source': source,
                'reference': reference,
                'prediction': prediction,
                'evaluation': evaluation
            })
        
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics from evaluation results."""
        if not results:
            return {}
        
        metrics = {
            'accuracy_scores': [],
            'fluency_scores': [],
            'completeness_scores': [],
            'overall_quality_scores': []
        }
        
        for result in results:
            eval_data = result['evaluation']
            metrics['accuracy_scores'].append(eval_data['accuracy']['score'])
            metrics['fluency_scores'].append(eval_data['fluency']['score'])
            metrics['completeness_scores'].append(eval_data['completeness']['score'])
            metrics['overall_quality_scores'].append(eval_data['overall_quality']['score'])
        
        # Calculate statistics
        summary = {}
        for metric_name, scores in metrics.items():
            if scores:
                summary[f'{metric_name}_mean'] = np.mean(scores)
                summary[f'{metric_name}_std'] = np.std(scores)
                summary[f'{metric_name}_min'] = np.min(scores)
                summary[f'{metric_name}_max'] = np.max(scores)
                summary[f'{metric_name}_median'] = np.median(scores)
        
        return summary

def load_data(data_file: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_results(results: List[Dict], metrics: Dict, output_file: str):
    """Save evaluation results to file."""
    output_data = {
        'metadata': {
            'model': 'gemini-gemba',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(results)
        },
        'metrics': metrics,
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate translations using Gemini API in GEMBA style")
    parser.add_argument("--api_key", required=True, help="Google Gemini API key")
    parser.add_argument("--model_name", default="gemini-1.5-flash", help="Gemini model to use")
    parser.add_argument("--lang_pair", required=True, help="Language pair (e.g., en-zh)")
    parser.add_argument("--data_file", required=True, help="Path to JSONL data file")
    parser.add_argument("--output_file", help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for API calls")
    
    args = parser.parse_args()
    
    # Parse language pair
    src_lang, tgt_lang = args.lang_pair.split('-')
    
    # Set default output file if not specified
    if not args.output_file:
        base_name = Path(args.data_file).stem
        args.output_file = f"gemini_gemba_{args.lang_pair}_{base_name}.json"
    
    print(f"Initializing Gemini GEMBA evaluator with model: {args.model_name}")
    print(f"Language pair: {src_lang} -> {tgt_lang}")
    print(f"Data file: {args.data_file}")
    print(f"Output file: {args.output_file}")
    
    # Initialize evaluator
    evaluator = GeminiGembaEvaluator(args.api_key, args.model_name)
    
    # Load data
    print("Loading data...")
    data = load_data(args.data_file)
    print(f"Loaded {len(data)} samples")
    
    # Evaluate translations
    print("Starting evaluation...")
    results = evaluator.evaluate_batch(data, src_lang, tgt_lang)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = evaluator.calculate_metrics(results)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total samples evaluated: {len(results)}")
    print(f"Accuracy: {metrics.get('accuracy_scores_mean', 0):.2f} ± {metrics.get('accuracy_scores_std', 0):.2f}")
    print(f"Fluency: {metrics.get('fluency_scores_mean', 0):.2f} ± {metrics.get('fluency_scores_std', 0):.2f}")
    print(f"Completeness: {metrics.get('completeness_scores_mean', 0):.2f} ± {metrics.get('completeness_scores_std', 0):.2f}")
    print(f"Overall Quality: {metrics.get('overall_quality_scores_mean', 0):.2f} ± {metrics.get('overall_quality_scores_std', 0):.2f}")
    
    # Save results
    print(f"\nSaving results to {args.output_file}...")
    save_results(results, metrics, args.output_file)
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main() 