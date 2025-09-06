import cv2
import numpy as np
import ray

@ray.remote
def detect_blur_and_black_segments(video_path: str, blur_thresh=100.0, black_thresh=10.0, segment_seconds=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps

    blur_segments = []
    black_segments = []

    segment_frames = int(segment_seconds * fps)
    current_frame = 0

    while current_frame < frame_count:
        blur_scores = []
        black_scores = []

        for i in range(segment_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + i)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Blur detection using Laplacian variance
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(lap_var)

            # Black screen detection using mean intensity
            mean_intensity = np.mean(gray)
            black_scores.append(mean_intensity)

        if len(blur_scores) > 0:
            avg_blur = np.mean(blur_scores)
            if avg_blur < blur_thresh:
                start_time = current_frame / fps
                end_time = (current_frame + segment_frames) / fps
                blur_segments.append({"start_time": round(start_time, 2), "end_time": round(end_time, 2)})

        if len(black_scores) > 0:
            avg_black = np.mean(black_scores)
            if avg_black < black_thresh:
                start_time = current_frame / fps
                end_time = (current_frame + segment_frames) / fps
                black_segments.append({"start_time": round(start_time, 2), "end_time": round(end_time, 2)})

        current_frame += segment_frames

    cap.release()
    return {
        "video_path": video_path,
        "blur_segments": blur_segments,
        "black_segments": black_segments
    }