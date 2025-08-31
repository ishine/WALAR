# Directory Walker

A comprehensive tool to explore and analyze directory structures, providing detailed insights into file organization, sizes, types, and metadata.

## Features

- **Complete directory traversal**: Walk through entire directory trees
- **File analysis**: Size, type, permissions, modification time, extensions
- **Symlink handling**: Detect and optionally follow symbolic links
- **Hash calculation**: Optional MD5 hash computation for files
- **Line counting**: Optional line count for text files
- **Depth control**: Limit traversal depth for large directories
- **Multiple output formats**: JSON and text reports
- **Performance statistics**: Timing and progress information
- **Error handling**: Graceful handling of permission issues and broken links

## Usage

### Command Line Interface

#### Basic Directory Analysis
```bash
python code/directory_walker.py /path/to/directory
```

#### Limited Depth Analysis
```bash
python code/directory_walker.py /path/to/directory --max-depth 3
```

#### Full Analysis with Hashes and Line Counts
```bash
python code/directory_walker.py /path/to/directory \
    --calculate-hashes \
    --count-lines \
    --output analysis.json
```

#### Follow Symlinks
```bash
python code/directory_walker.py /path/to/directory \
    --follow-symlinks \
    --output symlink_analysis.json
```

### Shell Script

#### Basic Usage
```bash
chmod +x scripts/walk_directory.sh
./scripts/walk_directory.sh /path/to/directory
```

#### Advanced Analysis
```bash
./scripts/walk_directory.sh /path/to/directory \
    --max-depth 5 \
    --calculate-hashes \
    --count-lines \
    --output "full_analysis.json"
```

#### Quick Analysis
```bash
./scripts/walk_directory.sh /path/to/directory \
    --quiet \
    --output "quick_analysis.json"
```

### Python API

```python
from code.directory_walker import DirectoryWalker

# Initialize walker
walker = DirectoryWalker(
    root_path="/path/to/directory",
    max_depth=3,
    follow_symlinks=False,
    calculate_hashes=True,
    count_lines=True
)

# Analyze directory
stats = walker.analyze()

# Print summary
walker.print_summary()

# Save results
walker.save_results("analysis.json", "json")
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `path` | Root directory path to analyze | Required |
| `--max-depth` | Maximum depth to traverse | Unlimited |
| `--follow-symlinks` | Follow symbolic links | False |
| `--calculate-hashes` | Calculate MD5 hashes for files | False |
| `--count-lines` | Count lines in text files | False |
| `--output` | Output file to save results | None |
| `--format` | Output format (json, txt) | json |
| `--quiet` | Suppress progress output | False |

## Output Formats

### JSON Output
Comprehensive structured data including:
- Summary statistics
- File type and extension counts
- Largest, oldest, and newest files
- Complete directory structure
- Detailed file information

```json
{
  "summary": {
    "root_path": "/path/to/directory",
    "total_files": 1250,
    "total_directories": 45,
    "total_size": 1073741824,
    "walk_time": 12.34
  },
  "file_types": {
    "file": 1200,
    "directory": 45,
    "symlink": 5
  },
  "extensions": {
    ".py": 150,
    ".txt": 89,
    ".json": 67
  },
  "largest_files": [
    ["/path/to/large/file.dat", 1048576],
    ["/path/to/another/file.bin", 524288]
  ],
  "files": [
    {
      "name": "example.py",
      "path": "/path/to/example.py",
      "size": 1024,
      "file_type": "file",
      "extension": ".py",
      "permissions": "-rw-r--r--",
      "modified_time": 1640995200.0,
      "is_symlink": false,
      "md5_hash": "d41d8cd98f00b204e9800998ecf8427e",
      "line_count": 45
    }
  ]
}
```

### Text Output
Human-readable report format:
```
DIRECTORY ANALYSIS REPORT
==================================================

Root path: /path/to/directory
Total files: 1,250
Total directories: 45
Total size: 1.0 GB
Walk time: 12.34 seconds

FILE TYPES:
  file: 1,200
  directory: 45
  symlink: 5

TOP 10 EXTENSIONS:
  .py: 150
  .txt: 89
  .json: 67
  .md: 45
  .sh: 34

LARGEST FILES:
  1.0 MB: large_file.dat
  512.0 KB: another_file.bin

DIRECTORY STRUCTURE:
  root/ (1,200 files, 45 dirs, 1.0 GB)
    src/ (450 files, 15 dirs, 450.0 MB)
      utils/ (89 files, 2 dirs, 89.0 MB)
    docs/ (67 files, 5 dirs, 67.0 MB)
```

## Use Cases

### 1. **Project Analysis**
Analyze code repositories to understand structure and identify large files:
```bash
./scripts/walk_directory.sh /path/to/project \
    --max-depth 4 \
    --calculate-hashes \
    --output "project_analysis.json"
```

### 2. **Storage Analysis**
Identify space usage and largest files:
```bash
./scripts/walk_directory.sh /path/to/storage \
    --output "storage_analysis.json"
```

### 3. **Backup Verification**
Check file integrity and detect changes:
```bash
./scripts/walk_directory.sh /path/to/backup \
    --calculate-hashes \
    --output "backup_verification.json"
```

### 4. **Code Metrics**
Analyze codebase structure and line counts:
```bash
./scripts/walk_directory.sh /path/to/codebase \
    --count-lines \
    --max-depth 3 \
    --output "code_metrics.json"
```

### 5. **System Cleanup**
Find old and large files for cleanup:
```bash
./scripts/walk_directory.sh /path/to/system \
    --output "cleanup_analysis.json"
```

## Performance Considerations

### **Large Directories**
- Use `--max-depth` to limit traversal depth
- Use `--quiet` for faster processing
- Consider excluding hash calculation for very large directories

### **Network Drives**
- Be cautious with symlink following on network drives
- Hash calculation may be slow on network storage
- Consider using `--quiet` to reduce network overhead

### **Memory Usage**
- The script loads all file information into memory
- For extremely large directories, consider processing in chunks
- Use depth limits to control memory usage

## Integration Examples

### **With Your qe-lr Project**
```bash
# Analyze your project structure
./scripts/walk_directory.sh . \
    --max-depth 3 \
    --output "qe_lr_analysis.json"

# Analyze specific subdirectories
./scripts/walk_directory.sh code/ \
    --count-lines \
    --output "code_analysis.json"

./scripts/walk_directory.sh data/ \
    --output "data_analysis.json"
```

### **Automated Analysis**
```bash
#!/bin/bash
# Daily analysis script
DATE=$(date +%Y%m%d)
./scripts/walk_directory.sh /path/to/monitor \
    --output "daily_analysis_${DATE}.json" \
    --quiet
```

### **Comparison Analysis**
```bash
# Analyze before and after changes
./scripts/walk_directory.sh /path/to/project \
    --calculate-hashes \
    --output "before_changes.json"

# Make changes...

./scripts/walk_directory.sh /path/to/project \
    --calculate-hashes \
    --output "after_changes.json"

# Compare the JSON files to see what changed
```

## Error Handling

The script handles various error scenarios gracefully:

1. **Permission Denied**: Skips inaccessible directories with warnings
2. **Broken Symlinks**: Detects and reports broken symbolic links
3. **File Access Errors**: Continues processing other files when one fails
4. **Unicode Issues**: Handles encoding problems in filenames
5. **Network Issues**: Gracefully handles network drive disconnections

## Troubleshooting

### **Common Issues**

1. **Permission Denied**: Run with appropriate permissions or use `--quiet`
2. **Out of Memory**: Use `--max-depth` to limit traversal
3. **Slow Performance**: Disable hash calculation and line counting for large directories
4. **Unicode Errors**: Ensure your terminal supports UTF-8

### **Performance Tips**

1. **Use depth limits** for large directory trees
2. **Disable hash calculation** unless needed
3. **Use quiet mode** for faster processing
4. **Process subdirectories separately** for very large structures

## License

This script is part of the qe-lr project and follows the same licensing terms. 