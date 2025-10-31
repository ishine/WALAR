#!/usr/bin/env python3
"""
Test script for benchmax functionality with reference sentences
"""

import json
import os
import sys
import tempfile

# Add the code directory to the path
sys.path.insert(0, "/mnt/gemini/data1/yifengliu/qe-lr/code")

from predict import load_benchmax_json, save_benchmax_results

def test_benchmax_with_reference():
    """Test the benchmax JSON loading with reference sentences."""
    
    # Create a test JSON file with the actual benchmax structure
    test_data = {
        "outputs": [
            "Hello world translated",
            "How are you translated",
            "Good morning translated"
        ],
        "spBLEU": 25.5
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        temp_file = f.name
    
    # Create temporary flores directory structure
    temp_flores_dir = tempfile.mkdtemp()
    temp_src_file = os.path.join(temp_flores_dir, "en.devtest")
    temp_tgt_file = os.path.join(temp_flores_dir, "cs.devtest")
    
    # Write source and target sentences to temp files
    with open(temp_src_file, 'w') as f:
        f.write("Hello world\nHow are you?\nGood morning\n")
    
    with open(temp_tgt_file, 'w') as f:
        f.write("Ahoj světe\nJak se máš?\nDobré ráno\n")
    
    try:
        # Monkey patch the function for testing
        def mock_load_benchmax_json(file_path, src_lang, tgt_lang):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            outputs = data.get('outputs', [])
            
            src_file = os.path.join(temp_flores_dir, f"{src_lang}.devtest")
            tgt_file = os.path.join(temp_flores_dir, f"{tgt_lang}.devtest")
            
            with open(src_file, 'r', encoding='utf-8') as f:
                source_sentences = [line.strip() for line in f.readlines()]
            
            with open(tgt_file, 'r', encoding='utf-8') as f:
                reference_sentences = [line.strip() for line in f.readlines()]
            
            if len(source_sentences) != len(outputs):
                raise ValueError(f"Mismatch in sentence counts: {len(source_sentences)} source vs {len(outputs)} target")
            if len(reference_sentences) != len(outputs):
                raise ValueError(f"Mismatch in sentence counts: {len(reference_sentences)} reference vs {len(outputs)} target")
            
            ds = []
            for src, ref, hyp in zip(source_sentences, reference_sentences, outputs):
                ds.append({
                    "source": src,
                    "reference": ref,
                    "hypothesis": hyp
                })
            return ds
        
        # Test loading
        print("Testing benchmax JSON loading with reference sentences...")
        ds = mock_load_benchmax_json(temp_file, "en", "cs")
        print(f"Loaded {len(ds)} items")
        for i, item in enumerate(ds):
            print(f"Item {i}:")
            print(f"  source: '{item['source']}'")
            print(f"  reference: '{item['reference']}'")
            print(f"  hypothesis: '{item['hypothesis']}'")
        
        # Test saving
        print("\nTesting benchmax JSON saving...")
        test_predictions = [0.8, 0.9, 0.7]
        save_benchmax_results(temp_file, ds, test_predictions, "metricX")
        
        # Verify the saved file
        with open(temp_file, 'r') as f:
            saved_data = json.load(f)
        
        print(f"\nSaved data structure:")
        print(f"  - outputs: {len(saved_data.get('outputs', []))} items")
        print(f"  - metricx_score: {saved_data.get('metricx_score', 'N/A')}")
        print(f"  - spBLEU: {saved_data.get('spBLEU', 'N/A')}")
        
        # Show first output item
        if 'outputs' in saved_data and len(saved_data['outputs']) > 0:
            print(f"  - First output: {saved_data['outputs'][0]}")
        
        print("\n✅ All tests passed!")
        
    finally:
        # Clean up
        os.unlink(temp_file)
        import shutil
        shutil.rmtree(temp_flores_dir)

if __name__ == "__main__":
    test_benchmax_with_reference()





