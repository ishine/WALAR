#!/usr/bin/env python3
"""
Example usage of the DirectoryWalker class.
This script demonstrates various ways to use the directory walker.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from directory_walker import DirectoryWalker

def example_basic_analysis():
    """Example of basic directory analysis."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Directory Analysis")
    print("=" * 60)
    
    # Analyze the current directory (code/)
    current_dir = Path(__file__).parent
    
    walker = DirectoryWalker(
        root_path=str(current_dir),
        max_depth=2,  # Limit depth for demo
        follow_symlinks=False,
        calculate_hashes=False,
        count_lines=False
    )
    
    print(f"Analyzing directory: {current_dir}")
    stats = walker.analyze()
    
    # Print summary
    walker.print_summary()
    
    # Save results
    output_file = "example_basic_analysis.json"
    walker.save_results(output_file, "json")
    print(f"\nResults saved to: {output_file}")

def example_detailed_analysis():
    """Example of detailed analysis with hashes and line counts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Detailed Analysis with Hashes and Line Counts")
    print("=" * 60)
    
    # Analyze the parent directory (qe-lr project root)
    project_root = Path(__file__).parent.parent
    
    walker = DirectoryWalker(
        root_path=str(project_root),
        max_depth=1,  # Limit depth for demo
        follow_symlinks=False,
        calculate_hashes=True,  # Calculate MD5 hashes
        count_lines=True        # Count lines in text files
    )
    
    print(f"Analyzing project root: {project_root}")
    stats = walker.analyze()
    
    # Print summary
    walker.print_summary()
    
    # Save results
    output_file = "example_detailed_analysis.json"
    walker.save_results(output_file, "json")
    print(f"\nResults saved to: {output_file}")

def example_custom_analysis():
    """Example of custom analysis with specific focus."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Analysis - Python Files Only")
    print("=" * 60)
    
    # Analyze only Python files in the code directory
    code_dir = Path(__file__).parent
    
    walker = DirectoryWalker(
        root_path=str(code_dir),
        max_depth=None,  # No depth limit
        follow_symlinks=False,
        calculate_hashes=False,
        count_lines=True  # Count lines in Python files
    )
    
    print(f"Analyzing Python files in: {code_dir}")
    stats = walker.analyze()
    
    # Filter for Python files only
    python_files = [f for f in walker.all_files if f.extension == '.py']
    
    print(f"\nFound {len(python_files)} Python files:")
    total_lines = 0
    for file_info in python_files:
        if file_info.line_count is not None and file_info.line_count >= 0:
            print(f"  {file_info.name}: {file_info.line_count} lines")
            total_lines += file_info.line_count
        else:
            print(f"  {file_info.name}: Could not count lines")
    
    print(f"\nTotal lines of Python code: {total_lines:,}")
    
    # Save filtered results
    output_file = "example_python_analysis.json"
    
    # Create custom output with only Python files
    output_data = {
        'summary': {
            'root_path': str(code_dir),
            'total_python_files': len(python_files),
            'total_lines': total_lines
        },
        'python_files': [
            {
                'name': f.name,
                'path': f.path,
                'size': f.size,
                'line_count': f.line_count,
                'modified_time': f.modified_time
            }
            for f in python_files
        ]
    }
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nPython analysis saved to: {output_file}")

def example_comparison_analysis():
    """Example of comparing two directory states."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Comparison Analysis")
    print("=" * 60)
    
    # Analyze current state
    current_dir = Path(__file__).parent
    
    print("Analyzing current state...")
    walker1 = DirectoryWalker(
        root_path=str(current_dir),
        max_depth=1,
        calculate_hashes=True
    )
    stats1 = walker1.analyze()
    
    # Simulate some changes (create a temporary file)
    temp_file = current_dir / "temp_example.txt"
    try:
        with open(temp_file, 'w') as f:
            f.write("This is a temporary file for demonstration\n")
            f.write("It will be deleted after the example\n")
        
        print("Created temporary file for comparison...")
        
        # Analyze new state
        print("Analyzing new state...")
        walker2 = DirectoryWalker(
            root_path=str(current_dir),
            max_depth=1,
            calculate_hashes=True
        )
        stats2 = walker2.analyze()
        
        # Compare states
        print(f"\nComparison Results:")
        print(f"Files before: {stats1.total_files}")
        print(f"Files after:  {stats2.total_files}")
        print(f"Size before:  {walker1._format_size(stats1.total_size)}")
        print(f"Size after:   {walker2._format_size(stats2.total_size)}")
        
        # Find new files
        files1 = {f.path for f in walker1.all_files}
        files2 = {f.path for f in walker2.all_files}
        new_files = files2 - files1
        deleted_files = files1 - files2
        
        if new_files:
            print(f"\nNew files: {len(new_files)}")
            for file_path in new_files:
                print(f"  + {Path(file_path).name}")
        
        if deleted_files:
            print(f"\nDeleted files: {len(deleted_files)}")
            for file_path in deleted_files:
                print(f"  - {Path(file_path).name}")
        
        # Save comparison results
        comparison_data = {
            'before': {
                'total_files': stats1.total_files,
                'total_size': stats1.total_size,
                'file_hashes': {f.path: f.md5_hash for f in walker1.all_files if f.md5_hash}
            },
            'after': {
                'total_files': stats2.total_files,
                'total_size': stats2.total_size,
                'file_hashes': {f.path: f.md5_hash for f in walker2.all_files if f.md5_hash}
            },
            'changes': {
                'new_files': list(new_files),
                'deleted_files': list(deleted_files)
            }
        }
        
        output_file = "example_comparison_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nComparison results saved to: {output_file}")
        
    finally:
        # Clean up temporary file
        if temp_file.exists():
            temp_file.unlink()
            print("Cleaned up temporary file")

def main():
    """Run all examples."""
    print("Directory Walker Examples")
    print("This script demonstrates various usage patterns of the DirectoryWalker class.")
    print()
    
    try:
        # Run examples
        example_basic_analysis()
        example_detailed_analysis()
        example_custom_analysis()
        example_comparison_analysis()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - example_basic_analysis.json")
        print("  - example_detailed_analysis.json")
        print("  - example_python_analysis.json")
        print("  - example_comparison_analysis.json")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 