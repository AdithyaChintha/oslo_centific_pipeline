import cv2
import numpy as np
from typing import Callable, Dict, Iterator, List, Optional, Tuple
from typing import Any, Dict, List, Optional


def frame_generator(video_path: str, frame_stride: int = 1,
                    preprocess: Optional[Callable[[any], any]] = None
                    ) -> Iterator[Tuple[any, int, float]]:
    """
    Yield frames from a video with a stride. Returns (frame, frame_idx, fps).
    - preprocess: optional callable to transform the frame (e.g., undistort).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_stride == 0:
            out_frame = preprocess(frame) if preprocess is not None else frame
            yield out_frame, idx, fps
        idx += 1

    cap.release()