# if we havve to use stream as azure blob use this commented code
# import ray
# import os
# import subprocess
# import cv2
# from pathlib import Path

# @ray.remote
# def split_video_into_shards(video_path, output_dir="/tmp/shards", duration_sec=60):
#     """Split video into shards preserving both video AND audio"""

#     video_path = str(Path(video_path))  # ensure plain string
#     if not os.path.exists(video_path):
#         raise FileNotFoundError(f"Video not found: {video_path}")

#     # Get video info
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()

#     if fps <= 0:
#         raise ValueError(f"Invalid FPS ({fps}) for video: {video_path}")

#     total_duration = total_frames / fps

#     basename = os.path.basename(video_path).split('.')[0]
#     os.makedirs(output_dir, exist_ok=True)

#     shard_paths = []
#     current_time = 0
#     shard_idx = 0

#     # Split using FFmpeg (preserve audio + video)
#     while current_time < total_duration:
#         shard_path = os.path.join(output_dir, f"{basename}_part{shard_idx}.mp4")

#         cmd = [
#             "ffmpeg", "-y",
#             "-i", video_path,
#             "-ss", str(current_time),
#             "-t", str(duration_sec),
#             "-c:v", "libx264",
#             "-c:a", "copy",
#             shard_path,
#         ]

#         result = subprocess.run(cmd, capture_output=True, text=True)
#         if result.returncode == 0:
#             shard_paths.append(shard_path)
#         else:
#             print(f"⚠️ Failed to create shard {shard_idx}: {result.stderr}")

#         current_time += duration_sec
#         shard_idx += 1

#     return shard_paths

import ray
import os
import subprocess
import cv2

@ray.remote
def split_video_into_shards(video_path, output_dir="/tmp/shards", duration_sec=60, downscale_to_480p=True):
    """Split video into shards preserving both video AND audio"""
    
    # Get video info first
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    cap.release()
    
    basename = os.path.basename(video_path).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)
    
    shard_paths = []
    current_time = 0
    shard_idx = 0
    
    # Split using FFmpeg to preserve audio + video
    while current_time < total_duration:
        shard_path = os.path.join(output_dir, f"{basename}_part{shard_idx}.mp4")
        
        # FFmpeg command to extract segment with forced video format + preserved audio
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(current_time),          # Start time
            '-t', str(duration_sec),           # Duration
            '-c:v', 'libx264',                 # Force H.264 video (consistent format)
        ]

        if downscale_to_480p:
            cmd.extend(['-vf', 'scale=-2:480'])
        
        cmd.extend([
            '-c:a', 'copy',                    # Copy audio as-is (preserve quality)
            shard_path
        ])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                shard_paths.append(shard_path)
            else:
                print(f"Warning: Failed to create shard {shard_idx}: {result.stderr}")
        except Exception as e:
            print(f"Error creating shard {shard_idx}: {e}")
        
        current_time += duration_sec
        shard_idx += 1
    
    return shard_paths

#//

# import ray
# import os
# import cv2

# @ray.remote
# def split_video_into_shards(video_path, output_dir="/tmp/shards", duration_sec=60):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frames_per_shard = duration_sec * fps

#     basename = os.path.basename(video_path).split('.')[0]
#     os.makedirs(output_dir, exist_ok=True)

#     current_shard = 0
#     out = None
#     frame_idx = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_idx % frames_per_shard == 0:
#             if out:
#                 out.release()
#             shard_path = os.path.join(output_dir, f"{basename}_part{current_shard}.mp4")
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(shard_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
#             current_shard += 1

#         out.write(frame)
#         frame_idx += 1

#     cap.release()
#     if out:
#         out.release()

#     return [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.mp4')]