#!/usr/bin/env python3
"""
Directory Walker - A comprehensive tool to explore and analyze directory structures.
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib
import mimetypes
import stat

@dataclass
class FileInfo:
    """Information about a single file."""
    name: str
    path: str
    size: int
    modified_time: float
    file_type: str
    extension: str
    permissions: str
    is_symlink: bool
    symlink_target: Optional[str] = None
    md5_hash: Optional[str] = None
    line_count: Optional[int] = None

@dataclass
class DirectoryInfo:
    """Information about a directory."""
    name: str
    path: str
    file_count: int
    dir_count: int
    total_size: int
    modified_time: float
    permissions: str
    depth: int
    subdirectories: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)

@dataclass
class DirectoryStats:
    """Overall statistics for the directory walk."""
    root_path: str
    total_files: int
    total_directories: int
    total_size: int
    start_time: float
    end_time: float
    file_types: Dict[str, int] = field(default_factory=dict)
    extensions: Dict[str, int] = field(default_factory=dict)
    largest_files: List[Tuple[str, int]] = field(default_factory=list)
    oldest_files: List[Tuple[str, float]] = field(default_factory=list)
    newest_files: List[Tuple[str, float]] = field(default_factory=list)

class DirectoryWalker:
    """
    A comprehensive directory walker that analyzes directory structures.
    """
    
    def __init__(self, root_path: str, max_depth: Optional[int] = None, 
                 follow_symlinks: bool = False, calculate_hashes: bool = False,
                 count_lines: bool = False):
        """
        Initialize the directory walker.
        
        Args:
            root_path: Root directory to start walking from
            max_depth: Maximum depth to traverse (None for unlimited)
            follow_symlinks: Whether to follow symbolic links
            calculate_hashes: Whether to calculate MD5 hashes for files
            count_lines: Whether to count lines in text files
        """
        self.root_path = Path(root_path).resolve()
        self.max_depth = max_depth
        self.follow_symlinks = follow_symlinks
        self.calculate_hashes = calculate_hashes
        self.count_lines = count_lines
        
        # Statistics
        self.stats = DirectoryStats(
            root_path=str(self.root_path),
            total_files=0,
            total_directories=0,
            total_size=0,
            start_time=time.time(),
            end_time=0
        )
        
        # File type and extension counters
        self.file_types = Counter()
        self.extensions = Counter()
        
        # File lists for analysis
        self.all_files: List[FileInfo] = []
        self.all_directories: List[DirectoryInfo] = []
        
        # Initialize mimetypes
        mimetypes.init()
    
    def get_file_info(self, file_path: Path, depth: int) -> Optional[FileInfo]:
        """Extract information about a single file."""
        try:
            stat_info = file_path.stat(follow_symlinks=self.follow_symlinks)
            
            # Check if it's a symlink
            is_symlink = file_path.is_symlink()
            symlink_target = None
            if is_symlink:
                try:
                    symlink_target = str(file_path.readlink())
                except OSError:
                    symlink_target = "broken_link"
            
            # Get file type
            if file_path.is_file():
                file_type = "file"
            elif file_path.is_dir():
                file_type = "directory"
            elif file_path.is_symlink():
                file_type = "symlink"
            else:
                file_type = "other"
            
            # Get extension
            extension = file_path.suffix.lower()
            
            # Get permissions
            permissions = stat.filemode(stat_info.st_mode)
            
            # Calculate MD5 hash if requested
            md5_hash = None
            if self.calculate_hashes and file_path.is_file() and not is_symlink:
                try:
                    md5_hash = self._calculate_md5(file_path)
                except Exception:
                    md5_hash = "error"
            
            # Count lines if requested
            line_count = None
            if self.count_lines and file_path.is_file() and not is_symlink:
                try:
                    line_count = self._count_lines(file_path)
                except Exception:
                    line_count = -1
            
            return FileInfo(
                name=file_path.name,
                path=str(file_path),
                size=stat_info.st_size,
                modified_time=stat_info.st_mtime,
                file_type=file_type,
                extension=extension,
                permissions=permissions,
                is_symlink=is_symlink,
                symlink_target=symlink_target,
                md5_hash=md5_hash,
                line_count=line_count
            )
            
        except (OSError, PermissionError) as e:
            print(f"Error accessing {file_path}: {e}")
            return None
    
    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except UnicodeDecodeError:
            # Try binary mode for non-text files
            try:
                with open(file_path, 'rb') as f:
                    return sum(1 for _ in f)
            except Exception:
                return -1
    
    def walk_directory(self, current_path: Path, depth: int = 0) -> DirectoryInfo:
        """Recursively walk through a directory."""
        if self.max_depth is not None and depth > self.max_depth:
            return DirectoryInfo(
                name=current_path.name,
                path=str(current_path),
                file_count=0,
                dir_count=0,
                total_size=0,
                modified_time=0,
                permissions="",
                depth=depth
            )
        
        try:
            # Get directory info
            dir_stat = current_path.stat(follow_symlinks=self.follow_symlinks)
            permissions = stat.filemode(dir_stat.st_mode)
            
            # Initialize directory info
            dir_info = DirectoryInfo(
                name=current_path.name,
                path=str(current_path),
                file_count=0,
                dir_count=0,
                total_size=0,
                modified_time=dir_stat.st_mtime,
                permissions=permissions,
                depth=depth
            )
            
            # List directory contents
            try:
                items = list(current_path.iterdir())
            except PermissionError:
                print(f"Permission denied accessing {current_path}")
                return dir_info
            
            # Process items
            for item in items:
                try:
                    if item.is_file() or (item.is_symlink() and not self.follow_symlinks):
                        # Process file
                        file_info = self.get_file_info(item, depth)
                        if file_info:
                            self.all_files.append(file_info)
                            dir_info.files.append(file_info.name)
                            dir_info.file_count += 1
                            dir_info.total_size += file_info.size
                            
                            # Update statistics
                            self.stats.total_files += 1
                            self.stats.total_size += file_info.size
                            
                            # Count file types and extensions
                            self.file_types[file_info.file_type] += 1
                            if file_info.extension:
                                self.extensions[file_info.extension] += 1
                    
                    elif item.is_dir():
                        # Process subdirectory
                        subdir_info = self.walk_directory(item, depth + 1)
                        dir_info.subdirectories.append(item.name)
                        dir_info.dir_count += 1
                        dir_info.total_size += subdir_info.total_size
                        
                        # Update statistics
                        self.stats.total_directories += 1
                        
                except (OSError, PermissionError) as e:
                    print(f"Error processing {item}: {e}")
                    continue
            
            self.all_directories.append(dir_info)
            return dir_info
            
        except Exception as e:
            print(f"Error walking directory {current_path}: {e}")
            return DirectoryInfo(
                name=current_path.name,
                path=str(current_path),
                file_count=0,
                dir_count=0,
                total_size=0,
                modified_time=0,
                permissions="",
                depth=depth
            )
    
    def analyze(self) -> DirectoryStats:
        """Analyze the directory structure."""
        print(f"Starting directory walk from: {self.root_path}")
        print(f"Max depth: {self.max_depth or 'unlimited'}")
        print(f"Follow symlinks: {self.follow_symlinks}")
        print(f"Calculate hashes: {self.calculate_hashes}")
        print(f"Count lines: {self.count_lines}")
        print("-" * 60)
        
        # Walk the directory
        self.walk_directory(self.root_path)
        
        # Finalize statistics
        self.stats.end_time = time.time()
        self.stats.file_types = dict(self.file_types)
        self.stats.extensions = dict(self.extensions)
        
        # Find largest files
        self.stats.largest_files = sorted(
            [(f.path, f.size) for f in self.all_files if f.file_type == "file"],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Find oldest and newest files
        file_times = [(f.path, f.modified_time) for f in self.all_files if f.file_type == "file"]
        if file_times:
            self.stats.oldest_files = sorted(file_times, key=lambda x: x[1])[:10]
            self.stats.newest_files = sorted(file_times, key=lambda x: x[1], reverse=True)[:10]
        
        return self.stats
    
    def print_summary(self):
        """Print a summary of the directory analysis."""
        print("\n" + "=" * 60)
        print("DIRECTORY ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Root path: {self.stats.root_path}")
        print(f"Total files: {self.stats.total_files:,}")
        print(f"Total directories: {self.stats.total_directories:,}")
        print(f"Total size: {self._format_size(self.stats.total_size)}")
        print(f"Walk time: {self.stats.end_time - self.stats.start_time:.2f} seconds")
        
        print(f"\nFile types:")
        for file_type, count in self.stats.file_types.items():
            print(f"  {file_type}: {count:,}")
        
        print(f"\nTop 10 file extensions:")
        for ext, count in self.stats.extensions.most_common(10):
            print(f"  {ext or '(no extension)'}: {count:,}")
        
        if self.stats.largest_files:
            print(f"\nTop 10 largest files:")
            for path, size in self.stats.largest_files:
                print(f"  {self._format_size(size)}: {Path(path).name}")
        
        if self.stats.oldest_files:
            print(f"\nTop 10 oldest files:")
            for path, mtime in self.stats.oldest_files:
                print(f"  {time.ctime(mtime)}: {Path(path).name}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def save_results(self, output_file: str, format: str = 'json'):
        """Save analysis results to file."""
        if format == 'json':
            self._save_json(output_file)
        elif format == 'txt':
            self._save_txt(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_json(self, output_file: str):
        """Save results as JSON."""
        output_data = {
            'summary': {
                'root_path': self.stats.root_path,
                'total_files': self.stats.total_files,
                'total_directories': self.stats.total_directories,
                'total_size': self.stats.total_size,
                'walk_time': self.stats.end_time - self.stats.start_time
            },
            'file_types': self.stats.file_types,
            'extensions': self.stats.extensions,
            'largest_files': self.stats.largest_files,
            'oldest_files': self.stats.oldest_files,
            'newest_files': self.stats.newest_files,
            'directories': [
                {
                    'name': d.name,
                    'path': d.path,
                    'file_count': d.file_count,
                    'dir_count': d.dir_count,
                    'total_size': d.total_size,
                    'depth': d.depth
                }
                for d in self.all_directories
            ],
            'files': [
                {
                    'name': f.name,
                    'path': f.path,
                    'size': f.size,
                    'file_type': f.file_type,
                    'extension': f.extension,
                    'permissions': f.permissions,
                    'modified_time': f.modified_time,
                    'is_symlink': f.is_symlink,
                    'symlink_target': f.symlink_target,
                    'md5_hash': f.md5_hash,
                    'line_count': f.line_count
                }
                for f in self.all_files
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    def _save_txt(self, output_file: str):
        """Save results as text."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DIRECTORY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Root path: {self.stats.root_path}\n")
            f.write(f"Total files: {self.stats.total_files:,}\n")
            f.write(f"Total directories: {self.stats.total_directories:,}\n")
            f.write(f"Total size: {self._format_size(self.stats.total_size)}\n")
            f.write(f"Walk time: {self.stats.end_time - self.stats.start_time:.2f} seconds\n\n")
            
            f.write("FILE TYPES:\n")
            for file_type, count in self.stats.file_types.items():
                f.write(f"  {file_type}: {count:,}\n")
            
            f.write(f"\nTOP 10 EXTENSIONS:\n")
            for ext, count in self.stats.extensions.most_common(10):
                f.write(f"  {ext or '(no extension)'}: {count:,}\n")
            
            f.write(f"\nLARGEST FILES:\n")
            for path, size in self.stats.largest_files:
                f.write(f"  {self._format_size(size)}: {Path(path).name}\n")
            
            f.write(f"\nDIRECTORY STRUCTURE:\n")
            for d in sorted(self.all_directories, key=lambda x: x.depth):
                indent = "  " * d.depth
                f.write(f"{indent}{d.name}/ ({d.file_count} files, {d.dir_count} dirs, {self._format_size(d.total_size)})\n")

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Directory Walker - Analyze directory structures")
    parser.add_argument("path", help="Root directory path to analyze")
    parser.add_argument("--max-depth", type=int, help="Maximum depth to traverse")
    parser.add_argument("--follow-symlinks", action="store_true", help="Follow symbolic links")
    parser.add_argument("--calculate-hashes", action="store_true", help="Calculate MD5 hashes for files")
    parser.add_argument("--count-lines", action="store_true", help="Count lines in text files")
    parser.add_argument("--output", help="Output file to save results")
    parser.add_argument("--format", choices=['json', 'txt'], default='json', help="Output format")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Check if path exists
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist")
        return 1
    
    if not os.path.isdir(args.path):
        print(f"Error: Path '{args.path}' is not a directory")
        return 1
    
    # Create directory walker
    walker = DirectoryWalker(
        root_path=args.path,
        max_depth=args.max_depth,
        follow_symlinks=args.follow_symlinks,
        calculate_hashes=args.calculate_hashes,
        count_lines=args.count_lines
    )
    
    # Analyze directory
    try:
        stats = walker.analyze()
        
        # Print summary
        if not args.quiet:
            walker.print_summary()
        
        # Save results if requested
        if args.output:
            walker.save_results(args.output, args.format)
            print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nDirectory walk interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during directory walk: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 