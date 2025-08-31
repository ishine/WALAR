import os
import json
import argparse
import time
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
import google.generativeai as genai
from tqdm import tqdm
from pathlib import Path
import csv

@dataclass
class GeminiGenerateArguments:
    """
    Arguments for Gemini text generation.
    """
    api_key: str = field(
        metadata={"help": "Google Gemini API key"}
    )
    model_name: str = field(
        default="gemini-1.5-flash",
        metadata={"help": "Gemini model to use for generation"}
    )
    input_file: str = field(
        default=None,
        metadata={"help": "Path to file containing prompts (JSONL, CSV, or TXT)"}
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save generated responses"}
    )
    prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Single prompt to generate from (alternative to input_file)"}
    )
    max_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum number of tokens to generate"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Temperature for generation (0.0 = deterministic, 1.0 = creative)"}
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling parameter"}
    )
    top_k: int = field(
        default=40,
        metadata={"help": "Top-k sampling parameter"}
    )
    batch_size: int = field(
        default=10,
        metadata={"help": "Batch size for processing"}
    )
    delay: float = field(
        default=0.1,
        metadata={"help": "Delay between API calls in seconds"}
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of retries for failed API calls"}
    )

class GeminiGenerator:
    """
    Text generator using Gemini API.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """Initialize the Gemini API client."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
    def generate_single_response(self, prompt: str, generation_config: Dict) -> Dict:
        """Generate a single response using Gemini API."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=generation_config.get('max_tokens', 1024),
                    temperature=generation_config.get('temperature', 0.7),
                    top_p=generation_config.get('top_p', 1.0),
                    top_k=generation_config.get('top_k', 40),
                )
            )
            
            return {
                'success': True,
                'response': response.text.strip(),
                'prompt_tokens': getattr(response, 'prompt_token_count', None),
                'response_tokens': getattr(response, 'response_token_count', None),
                'total_tokens': getattr(response, 'total_token_count', None),
                'finish_reason': getattr(response, 'finish_reason', None)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'prompt_tokens': None,
                'response_tokens': None,
                'total_tokens': None,
                'finish_reason': None
            }
    
    def generate_batch_responses(self, prompts: List[Dict], generation_config: Dict) -> List[Dict]:
        """Generate responses for a batch of prompts."""
        results = []
        
        for i, prompt_data in enumerate(tqdm(prompts, desc="Generating responses")):
            prompt_text = prompt_data.get('prompt', prompt_data.get('text', ''))
            prompt_id = prompt_data.get('id', i)
            
            if not prompt_text:
                print(f"Warning: Empty prompt at index {i}")
                continue
            
            # Generate response
            result = self.generate_single_response(prompt_text, generation_config)
            
            # Add metadata
            result.update({
                'prompt_id': prompt_id,
                'prompt': prompt_text,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            results.append(result)
            
            # Add delay to respect rate limits
            if i < len(prompts) - 1:  # Don't delay after the last item
                time.sleep(generation_config.get('delay', 0.1))
        
        return results
    
    def load_prompts(self, input_file: str) -> List[Dict]:
        """Load prompts from various file formats."""
        file_ext = Path(input_file).suffix.lower()
        
        if file_ext == '.jsonl':
            return self._load_jsonl(input_file)
        elif file_ext == '.csv':
            return self._load_csv(input_file)
        elif file_ext == '.txt':
            return self._load_txt(input_file)
        elif file_ext == '.json':
            return self._load_json(input_file)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """Load prompts from JSONL file."""
        prompts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            prompts.append(data)
                        else:
                            prompts.append({'prompt': str(data), 'id': line_num})
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at line {line_num + 1}: {e}")
                        prompts.append({'prompt': line.strip(), 'id': line_num})
        return prompts
    
    def _load_csv(self, file_path: str) -> List[Dict]:
        """Load prompts from CSV file."""
        prompts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader):
                # Try to find prompt column
                prompt_col = None
                for col in ['prompt', 'text', 'input', 'question', 'query']:
                    if col in row and row[col].strip():
                        prompt_col = col
                        break
                
                if prompt_col:
                    prompts.append({
                        'id': row_num,
                        'prompt': row[prompt_col].strip(),
                        **{k: v for k, v in row.items() if k != prompt_col}
                    })
                else:
                    # Use first non-empty column as prompt
                    for col, value in row.items():
                        if value and value.strip():
                            prompts.append({
                                'id': row_num,
                                'prompt': value.strip(),
                                **{k: v for k, v in row.items() if k != col}
                            })
                            break
        return prompts
    
    def _load_txt(self, file_path: str) -> List[Dict]:
        """Load prompts from text file (one prompt per line)."""
        prompts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    prompts.append({
                        'id': line_num,
                        'prompt': line.strip()
                    })
        return prompts
    
    def _load_json(self, file_path: str) -> List[Dict]:
        """Load prompts from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # If it's a dict with prompts as values or nested structure
            if 'prompts' in data:
                return data['prompts']
            elif 'data' in data:
                return data['data']
            else:
                # Convert dict to list of prompts
                return [{'prompt': v, 'id': k} for k, v in data.items()]
        else:
            raise ValueError(f"Unexpected JSON structure in {file_path}")
    
    def save_results(self, results: List[Dict], output_file: str, format: str = 'auto'):
        """Save results to file in various formats."""
        if format == 'auto':
            format = Path(output_file).suffix.lower()
        
        if format == '.jsonl':
            self._save_jsonl(results, output_file)
        elif format == '.json':
            self._save_json(results, output_file)
        elif format == '.csv':
            self._save_csv(results, output_file)
        elif format == '.txt':
            self._save_txt(results, output_file)
        else:
            # Default to JSON
            self._save_json(results, output_file)
    
    def _save_jsonl(self, results: List[Dict], output_file: str):
        """Save results as JSONL."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def _save_json(self, results: List[Dict], output_file: str):
        """Save results as JSON."""
        output_data = {
            'metadata': {
                'model': self.model_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_prompts': len(results),
                'successful_generations': sum(1 for r in results if r['success'])
            },
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    def _save_csv(self, results: List[Dict], output_file: str):
        """Save results as CSV."""
        if not results:
            return
        
        # Get all possible fields
        all_fields = set()
        for result in results:
            all_fields.update(result.keys())
        
        # Order fields logically
        field_order = ['prompt_id', 'prompt', 'response', 'success', 'timestamp']
        remaining_fields = [f for f in sorted(all_fields) if f not in field_order]
        field_order.extend(remaining_fields)
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_order)
            writer.writeheader()
            
            for result in results:
                # Ensure all fields are present
                row = {field: result.get(field, '') for field in field_order}
                writer.writerow(row)
    
    def _save_txt(self, results: List[Dict], output_file: str):
        """Save results as text (prompt + response pairs)."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"PROMPT:\n{result['prompt']}\n\n")
                if result['success']:
                    f.write(f"RESPONSE:\n{result['response']}\n")
                else:
                    f.write(f"ERROR:\n{result['error']}\n")
                f.write("-" * 80 + "\n\n")

def main():
    parser = argparse.ArgumentParser(description="Generate text using Gemini API")
    parser.add_argument("--api_key", required=True, help="Google Gemini API key")
    parser.add_argument("--model_name", default="gemini-1.5-flash", help="Gemini model to use")
    parser.add_argument("--input_file", help="Path to file containing prompts")
    parser.add_argument("--prompt", help="Single prompt to generate from")
    parser.add_argument("--output_file", help="Path to save generated responses")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls")
    parser.add_argument("--format", choices=['json', 'jsonl', 'csv', 'txt'], 
                       help="Output format (auto-detected from output_file if not specified)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input_file and not args.prompt:
        parser.error("Either --input_file or --prompt must be specified")
    
    if args.input_file and args.prompt:
        parser.error("Cannot specify both --input_file and --prompt")
    
    # Set default output file if not specified
    if not args.output_file:
        if args.prompt:
            args.output_file = f"gemini_generated_{int(time.time())}.json"
        else:
            base_name = Path(args.input_file).stem
            args.output_file = f"gemini_generated_{base_name}.json"
    
    print(f"Initializing Gemini generator with model: {args.model_name}")
    
    # Initialize generator
    generator = GeminiGenerator(args.api_key, args.model_name)
    
    # Prepare generation config
    generation_config = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'delay': args.delay
    }
    
    if args.prompt:
        # Single prompt generation
        print(f"Generating response for single prompt...")
        result = generator.generate_single_response(args.prompt, generation_config)
        result.update({
            'prompt_id': 0,
            'prompt': args.prompt,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        results = [result]
        
    else:
        # Batch generation from file
        print(f"Loading prompts from {args.input_file}...")
        prompts = generator.load_prompts(args.input_file)
        print(f"Loaded {len(prompts)} prompts")
        
        print("Starting generation...")
        results = generator.generate_batch_responses(prompts, generation_config)
    
    # Calculate statistics
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n" + "="*50)
    print("GENERATION SUMMARY")
    print("="*50)
    print(f"Total prompts: {len(results)}")
    print(f"Successful generations: {successful}")
    print(f"Failed generations: {failed}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    
    if successful > 0:
        response_lengths = [len(r['response']) for r in results if r['success']]
        print(f"Average response length: {sum(response_lengths)/len(response_lengths):.1f} characters")
    
    # Save results
    print(f"\nSaving results to {args.output_file}...")
    output_format = args.format or Path(args.output_file).suffix.lstrip('.')
    generator.save_results(results, args.output_file, output_format)
    print("Generation completed successfully!")

if __name__ == "__main__":
    main() 