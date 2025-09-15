#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from typing import List, Optional
import ray

# ---------- helpers ----------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _basename_noext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def _run(cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def _ffprobe_duration(path: str) -> float:
    """Get duration of audio/video file using ffprobe"""
    out = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], timeout=10)
    if out.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {out.stderr}")
    return float(out.stdout.strip())

# ---------- public API ----------
@ray.remote
def split_audio_into_shards(
    audio_path: str,
    output_dir: str = "/tmp/audio_shards",
    duration_sec: int = 60,
    audio_format: str = "wav",  # Output format: wav, mp3, aac, etc.
    audio_codec: str = "pcm_s16le",  # For WAV: pcm_s16le, pcm_s24le, pcm_f32le
    sample_rate: Optional[int] = None,  # Resample if specified
    channels: Optional[int] = None,  # Mono=1, Stereo=2, etc.
    start_idx: int = 0,  # Starting index for part numbering
) -> List[str]:
    """
    Split audio file into shards with the same timestamps as video shards.
    This ensures audio and video shards are perfectly synchronized.
    
    Args:
        audio_path: Path to input audio file (.wav, .mp3, .aac, etc.)
        output_dir: Directory to save audio shards
        duration_sec: Duration of each shard in seconds (should match video)
        audio_format: Output audio format (wav, mp3, aac, flac)
        audio_codec: Audio codec to use
        sample_rate: Target sample rate (None to keep original)
        channels: Target number of channels (None to keep original)
    
    Returns:
        List of paths to audio shard files
    """
    p = str(Path(audio_path))
    if not os.path.exists(p):
        raise FileNotFoundError(f"Audio file not found: {p}")
    _ensure_dir(output_dir)

    total_duration = _ffprobe_duration(p)

    # Submit per-shard Ray tasks
    futures = []
    t = 0.0
    idx = start_idx  # Use the provided starting index

    while t < total_duration - 1e-6:
        outp = os.path.join(output_dir, f"{_basename_noext(p)}_part{idx}.{audio_format}")
        dur = min(duration_sec, max(0.0, total_duration - t))

        futures.append(_audio_shard_task.remote(
            audio_path=p,
            start_time=t,
            duration=dur,
            output_path=outp,
            audio_codec=audio_codec,
            sample_rate=sample_rate,
            channels=channels,
        ))
        t += duration_sec
        idx += 1

    results = ray.get(futures)
    return [r for r in results if r is not None]

# ---------- per-shard task ----------
@ray.remote
def _audio_shard_task(
    audio_path: str,
    start_time: float,
    duration: float,
    output_path: str,
    audio_codec: str,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
) -> Optional[str]:
    """
    Extract a single audio shard from the input audio file.
    
    Args:
        audio_path: Path to input audio file
        start_time: Start time in seconds
        duration: Duration in seconds
        output_path: Output file path
        audio_codec: Audio codec to use
        sample_rate: Target sample rate (None to keep original)
        channels: Target number of channels (None to keep original)
    
    Returns:
        Output path if successful, None if failed
    """
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_time}",  # Start time
        "-i", audio_path,        # Input file
        "-t", f"{duration}",     # Duration
        "-c:a", audio_codec,     # Audio codec
    ]
    
    # Add sample rate if specified
    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])
    
    # Add channel configuration if specified
    if channels:
        cmd.extend(["-ac", str(channels)])
    
    # Add output path
    cmd.append(output_path)

    try:
        print(f"🎵 audio shard {Path(output_path).name} | start={start_time:.1f}s | dur={duration:.1f}s")
        res = _run(cmd, timeout=600)  # 10 minute timeout for audio processing
        
        if res.returncode == 0:
            # Verify output file was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            else:
                print(f"Audio shard created but file is empty or missing: {output_path}")
                return None
        else:
            print(f"FFmpeg failed for audio shard: {res.stderr[:400]}")
            return None
            
    except Exception as e:
        print(f"Audio shard error: {e}")
        return None

# ---------- utility functions ----------
def get_audio_info(audio_path: str) -> dict:
    """Get detailed information about an audio file"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", audio_path
    ]
    
    try:
        result = _run(cmd, timeout=30)
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
        else:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")
    except Exception as e:
        raise RuntimeError(f"Failed to get audio info: {e}")

def match_video_shards_timestamps(video_shards: List[str], audio_path: str, 
                                output_dir: str, duration_sec: int = 60) -> List[str]:
    """
    Create audio shards that exactly match the timestamps of existing video shards.
    
    Args:
        video_shards: List of video shard file paths
        audio_path: Path to source audio file
        output_dir: Directory to save audio shards
        duration_sec: Expected duration per shard
    
    Returns:
        List of audio shard paths matching video shard timestamps
    """
    if not ray.is_initialized():
        ray.init()
    
    # Sort video shards by filename to ensure correct order
    sorted_video_shards = sorted(video_shards, key=lambda x: os.path.basename(x))
    
    # Use the same splitting logic but with exact timing
    future = split_audio_into_shards.remote(
        audio_path=audio_path,
        output_dir=output_dir,
        duration_sec=duration_sec,
        audio_format="wav",
        audio_codec="pcm_s16le"
    )
    
    return ray.get(future)

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    import json
    
    ap = argparse.ArgumentParser(description="Split audio file into shards matching video timestamps")
    ap.add_argument("--audio", type=str, help="Input audio file path")
    ap.add_argument("--output_dir", type=str, help="Output directory for audio shards")
    ap.add_argument("--duration", type=int, default=60, help="Duration per shard in seconds")
    ap.add_argument("--format", type=str, default="wav", help="Output audio format")
    ap.add_argument("--codec", type=str, default="pcm_s16le", help="Audio codec")
    ap.add_argument("--sample-rate", type=int, help="Target sample rate")
    ap.add_argument("--channels", type=int, help="Target number of channels")
    ap.add_argument("--info", action="store_true", help="Show audio file information")
    
    args = ap.parse_args()

    if args.info:
        # Show audio file information
        try:
            info = get_audio_info(args.audio)
            print(json.dumps(info, indent=2))
        except Exception as e:
            print(f"Error getting audio info: {e}")
        exit(0)

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    # Split audio into shards
    try:
        future = split_audio_into_shards.remote(
            audio_path=args.audio,
            output_dir=args.output_dir,
            duration_sec=args.duration,
            audio_format=args.format,
            audio_codec=args.codec,
            sample_rate=args.sample_rate,
            channels=args.channels,
        )
        
        shard_paths = ray.get(future)
        
        print(f"✅ Created {len(shard_paths)} audio shards:")
        for path in shard_paths:
            print(f"  {path}")
            
    except Exception as e:
        print(f"❌ Error creating audio shards: {e}")
        raise
