import ray
import os
import cv2

@ray.remote
def split_video_into_shards(video_path, output_dir="/tmp/shards", duration_sec=60):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_per_shard = duration_sec * fps

    basename = os.path.basename(video_path).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)

    current_shard = 0
    out = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frames_per_shard == 0:
            if out:
                out.release()
            shard_path = os.path.join(output_dir, f"{basename}_part{current_shard}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(shard_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
            current_shard += 1

        out.write(frame)
        frame_idx += 1

    cap.release()
    if out:
        out.release()

    return [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.mp4')]