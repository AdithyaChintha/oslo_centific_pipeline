#!/usr/bin/env python3
"""
Chunk ID Extraction Utility

This module provides functions to extract chunk IDs and other metadata
from file paths in the multi-chunk processing pipeline.
"""

import re
import os
from pathlib import Path
from typing import Optional, Dict, Tuple

def extract_chunk_id_from_path(file_path: str) -> Optional[str]:
    """
    Extract chunk ID from file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Chunk ID string (e.g., "01-video", "02-audio") or None if not found
    """
    filename = os.path.basename(file_path)
    
    # Pattern for multi-chunk files: 11_515fb09c-f12f-48ee-91e7-b21c9f4ac5ab_video1_working-on-laptop_20250907T00:00:00.000Z122100_01_video.insv
    pattern = r'(\d+)_[^_]+_[^_]+_[^_]+_\d{8}T\d{2}:\d{2}:\d{2}\.\d{3}Z\d+_(\d+)_(video|audio)'
    match = re.search(pattern, filename)
    
    if match:
        sequence_number = match.group(2)
        chunk_type = match.group(3)
        return f"{sequence_number}-{chunk_type}"
    
    # Fallback pattern for simpler filenames
    pattern_simple = r'(\d+)_(video|audio)'
    match_simple = re.search(pattern_simple, filename)
    
    if match_simple:
        sequence_number = match_simple.group(1)
        chunk_type = match_simple.group(2)
        return f"{sequence_number}-{chunk_type}"
    
    return None

def extract_sequence_number_from_path(file_path: str) -> Optional[str]:
    """
    Extract sequence number from file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Sequence number string (e.g., "01", "02") or None if not found
    """
    filename = os.path.basename(file_path)
    
    # Pattern for multi-chunk files
    pattern = r'(\d+)_[^_]+_[^_]+_[^_]+_\d{8}T\d{2}:\d{2}:\d{2}\.\d{3}Z\d+_(\d+)_(video|audio)'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(2)
    
    # Fallback pattern
    pattern_simple = r'(\d+)_(video|audio)'
    match_simple = re.search(pattern_simple, filename)
    
    if match_simple:
        return match_simple.group(1)
    
    return None

def extract_chunk_type_from_path(file_path: str) -> Optional[str]:
    """
    Extract chunk type from file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Chunk type string ("video" or "audio") or None if not found
    """
    filename = os.path.basename(file_path)
    
    # Pattern for multi-chunk files
    pattern = r'(\d+)_[^_]+_[^_]+_[^_]+_\d{8}T\d{2}:\d{2}:\d{2}\.\d{3}Z\d+_(\d+)_(video|audio)'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(3)
    
    # Fallback pattern
    pattern_simple = r'(\d+)_(video|audio)'
    match_simple = re.search(pattern_simple, filename)
    
    if match_simple:
        return match_simple.group(2)
    
    return None

def extract_session_metadata_from_path(file_path: str) -> Dict[str, str]:
    """
    Extract comprehensive metadata from file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with extracted metadata
    """
    filename = os.path.basename(file_path)
    
    # Pattern for multi-chunk files
    pattern = r'(\d+)_([^_]+)_([^_]+)_([^_]+)_(\d{8}T\d{2}:\d{2}:\d{2}\.\d{3}Z\d+)_(\d+)_(video|audio)'
    match = re.search(pattern, filename)
    
    if match:
        return {
            "home_id": match.group(1),
            "participant_id": match.group(2),
            "activity": match.group(3),
            "timestamp": match.group(5),
            "sequence_number": match.group(6),
            "chunk_type": match.group(7),
            "chunk_id": f"{match.group(6)}-{match.group(7)}"
        }
    
    # Fallback for simpler patterns
    chunk_id = extract_chunk_id_from_path(file_path)
    sequence_number = extract_sequence_number_from_path(file_path)
    chunk_type = extract_chunk_type_from_path(file_path)
    
    return {
        "chunk_id": chunk_id,
        "sequence_number": sequence_number,
        "chunk_type": chunk_type
    }

def get_chunk_id_for_shard(shard_path: str) -> Optional[str]:
    """
    Extract chunk ID from shard file path.
    
    Args:
        shard_path: Path to the shard file
        
    Returns:
        Chunk ID string or None if not found
    """
    filename = os.path.basename(shard_path)
    
    # Pattern for shard files: 11_515fb09c-f12f-48ee-91e7-b21c9f4ac5ab_video1_working-on-laptop_20250907T00:00:00.000Z122100_01_video_ERP_4096x2048_part0.mp4
    pattern = r'(\d+)_[^_]+_[^_]+_[^_]+_\d{8}T\d{2}:\d{2}:\d{2}\.\d{3}Z\d+_(\d+)_(video|audio)'
    match = re.search(pattern, filename)
    
    if match:
        sequence_number = match.group(2)
        chunk_type = match.group(3)
        return f"{sequence_number}-{chunk_type}"
    
    return None

def get_view_name_from_shard_path(shard_path: str) -> Optional[str]:
    """
    Extract view name from shard file path.
    
    Args:
        shard_path: Path to the shard file
        
    Returns:
        View name (front, back, left, right, erp) or None if not found
    """
    filename = os.path.basename(shard_path)
    
    # Check for view names in the path
    view_names = ["front", "back", "left", "right", "erp"]
    
    for view_name in view_names:
        if view_name in filename.lower():
            return view_name
    
    # Check parent directory
    parent_dir = os.path.basename(os.path.dirname(shard_path))
    for view_name in view_names:
        if view_name in parent_dir.lower():
            return view_name
    
    return None
