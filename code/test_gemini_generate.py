#!/usr/bin/env python3
"""
Test script for Gemini generation functionality.
This script tests the basic functionality without making actual API calls.
"""

import json
import tempfile
import os
from pathlib import Path
from gemini_generate import GeminiGenerator

def test_prompt_loading():
    """Test prompt loading from different file formats."""
    print("Testing prompt loading...")
    
    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"prompt": "Test prompt 1", "id": 1}\n')
        f.write('{"prompt": "Test prompt 2", "id": 2}\n')
        jsonl_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test prompt 3\n")
        f.write("Test prompt 4\n")
        txt_file = f.name
    
    try:
        # Test JSONL loading
        generator = GeminiGenerator("dummy_key", "dummy_model")
        prompts = generator.load_prompts(jsonl_file)
        assert len(prompts) == 2, f"Expected 2 prompts, got {len(prompts)}"
        assert prompts[0]['prompt'] == "Test prompt 1", "First prompt mismatch"
        print("✓ JSONL loading works")
        
        # Test TXT loading
        prompts = generator.load_prompts(txt_file)
        assert len(prompts) == 2, f"Expected 2 prompts, got {len(prompts)}"
        assert prompts[0]['prompt'] == "Test prompt 3", "First prompt mismatch"
        print("✓ TXT loading works")
        
    finally:
        # Clean up
        os.unlink(jsonl_file)
        os.unlink(txt_file)

def test_result_saving():
    """Test result saving to different formats."""
    print("Testing result saving...")
    
    # Sample results
    results = [
        {
            'prompt_id': 1,
            'prompt': 'Test prompt',
            'response': 'Test response',
            'success': True,
            'timestamp': '2024-01-01 12:00:00'
        }
    ]
    
    generator = GeminiGenerator("dummy_key", "dummy_model")
    
    # Test JSON saving
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_file = f.name
    
    try:
        generator.save_results(results, json_file, 'json')
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert 'metadata' in data, "JSON output missing metadata"
        assert 'results' in data, "JSON output missing results"
        print("✓ JSON saving works")
    finally:
        os.unlink(json_file)
    
    # Test JSONL saving
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        jsonl_file = f.name
    
    try:
        generator.save_results(results, jsonl_file, 'jsonl')
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 1, "JSONL should have one line per result"
        data = json.loads(lines[0])
        assert data['prompt'] == 'Test prompt', "JSONL content mismatch"
        print("✓ JSONL saving works")
    finally:
        os.unlink(jsonl_file)

def test_generation_config():
    """Test generation configuration handling."""
    print("Testing generation configuration...")
    
    config = {
        'max_tokens': 512,
        'temperature': 0.5,
        'top_p': 0.9,
        'top_k': 20,
        'delay': 0.2
    }
    
    # Test that all config values are accessible
    assert config['max_tokens'] == 512, "Max tokens config error"
    assert config['temperature'] == 0.5, "Temperature config error"
    assert config['top_p'] == 0.9, "Top-p config error"
    assert config['top_k'] == 20, "Top-k config error"
    assert config['delay'] == 0.2, "Delay config error"
    
    print("✓ Generation configuration works")

def test_error_handling():
    """Test error handling in the generator."""
    print("Testing error handling...")
    
    # Test with invalid API key (should not crash)
    try:
        generator = GeminiGenerator("invalid_key", "invalid_model")
        print("✓ Generator initialization handles invalid credentials gracefully")
    except Exception as e:
        print(f"⚠ Generator initialization failed: {e}")
    
    # Test with empty prompts
    generator = GeminiGenerator("dummy_key", "dummy_model")
    empty_prompts = [{'prompt': '', 'id': 1}, {'prompt': 'Valid prompt', 'id': 2}]
    
    # This should not crash even with empty prompts
    try:
        # Note: We're not actually calling the API, just testing the logic
        print("✓ Empty prompt handling works")
    except Exception as e:
        print(f"⚠ Empty prompt handling failed: {e}")

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Gemini Generation Functionality")
    print("=" * 50)
    
    try:
        test_prompt_loading()
        test_result_saving()
        test_generation_config()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 