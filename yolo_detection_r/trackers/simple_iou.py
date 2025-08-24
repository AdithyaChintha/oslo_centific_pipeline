import numpy as np
from typing import List, Tuple, Optional

class IoUTracker:
    """
    Very small IoU-based tracker for fallback.
    - update(detections, frame_idx) where detections is list of [x1,y1,x2,y2,score,cls]
    - returns list of track_ids aligned with detections (int or -1)
    """
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30, min_hits: int = 1):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = {}  # id -> {'bbox': [x1,y1,x2,y2], 'cls': int, 'last_frame': int, 'age': int, 'hits': int}
        self._next_id = 1

    @staticmethod
    def _iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        interW = max(0.0, xB - xA); interH = max(0.0, yB - yA)
        inter = interW * interH
        areaA = max(0.0, (a[2]-a[0])*(a[3]-a[1]))
        areaB = max(0.0, (b[2]-b[0])*(b[3]-b[1]))
        union = areaA + areaB - inter
        return 0.0 if union <= 0 else inter/union

    def update(self, detections: List[List[float]], frame_idx: int) -> List[int]:
        """
        detections: list of [x1,y1,x2,y2,score,cls]
        returns: list of track_id aligned with detections (int)
        """
        track_ids = [-1] * len(detections)

        # build candidate mapping of existing tracks
        used_tracks = set()

        # greedy match: for each detection, find best track by IoU (and same class)
        for d_i, det in enumerate(detections):
            bbox = tuple(det[0:4])
            cls = int(det[5]) if len(det) > 5 else None
            best_tid = None
            best_iou = 0.0
            for tid, t in self.tracks.items():
                if tid in used_tracks:
                    continue
                if (cls is not None) and (t.get('cls') is not None) and (t.get('cls') != cls):
                    continue
                iou = self._iou(bbox, tuple(t['bbox']))
                if iou > best_iou and iou >= self.iou_thresh:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                # assign
                track_ids[d_i] = best_tid
                used_tracks.add(best_tid)
                self.tracks[best_tid]['bbox'] = list(bbox)
                self.tracks[best_tid]['last_frame'] = frame_idx
                self.tracks[best_tid]['age'] = 0
                self.tracks[best_tid]['hits'] += 1
            else:
                # new track
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = {
                    'bbox': list(bbox),
                    'cls': cls,
                    'last_frame': frame_idx,
                    'age': 0,
                    'hits': 1
                }
                track_ids[d_i] = tid
                used_tracks.add(tid)

        # age and prune tracks
        remove = []
        for tid, t in self.tracks.items():
            if t['last_frame'] != frame_idx:
                t['age'] += 1
            if t['age'] > self.max_age:
                remove.append(tid)
        for tid in remove:
            del self.tracks[tid]

        return track_ids