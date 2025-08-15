import sys
import json
import cv2
from datetime import timedelta
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

# make ../ importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from deepface import DeepFace
from utils import config as C


class FaceAgeDetector:
    """
    A class for detecting age and gender from video frames using DeepFace.
    """
    
    def __init__(self, config=None):
        """
        Initialize the FaceAgeDetector.
        
        Args:
            config: Configuration object with settings (defaults to utils.config)
        """
        self.config = config or C
        self.processed_frames = 0
        self.frames_json = []
        
    def ensure_dir(self, path: Union[str, Path]) -> Path:
        """Create directory if it doesn't exist."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    def normalize_gender(self, g):
        """
        Normalize gender output from DeepFace.
        
        DeepFace can return:
          - a string label ("Man"/"Woman"), or
          - a dict with probabilities {"Woman": np.float32(...), "Man": np.float32(...)}
        
        Returns:
            tuple: (label, scores_dict_as_float) where label is "Man"/"Woman"
        """
        if isinstance(g, str):
            return g, None
        if isinstance(g, dict):
            # convert np.float32 to float and pick max label
            scores = {k: float(v) for k, v in g.items()}
            label = max(scores, key=scores.get)
            return label, scores
        return None, None
    
    def classify_age(self, age):
        """
        Classify age into minor, adult, or senior categories.
        
        Args:
            age: Age value (int or float)
            
        Returns:
            str: Age classification ("minor", "adult", or "senior")
        """
        if age is None:
            return "unknown"
        
        age = float(age)
        if age < self.config.MINOR_AGE_THRESHOLD:
            return "minor"
        elif age >= self.config.SENIOR_AGE_THRESHOLD:
            return "senior"
        else:
            return "adult"
    
    def analyze_frame(self, frame):
        """
        Analyze a single frame for face detection, age, and gender.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            List[Dict]: List of face analysis results with age, age_classification, gender_label, and gender_scores
        """
        result = DeepFace.analyze(
            frame,
            actions=self.config.ACTIONS,
            enforce_detection=self.config.ENFORCE_DETECTION
        )

        # DeepFace sometimes returns dict or list
        if isinstance(result, dict):
            result = [result]

        faces_out = []
        for r in result:
            age = r.get("age", None)
            gender_raw = r.get("gender", None)
            gender_label, gender_scores = self.normalize_gender(gender_raw)
            
            # Add age classification
            age_classification = self.classify_age(age)
            
            faces_out.append({
                "age": age if (age is None or isinstance(age, (int, float))) else float(age),
                "age_classification": age_classification,
                "gender_label": gender_label,
                "gender_scores": gender_scores
            })
        return faces_out
    
    def process_video(self, video_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
        """
        Process a video file and extract face analysis from frames.
        
        Args:
            video_path: Path to the input video file
            output_dir: Optional output directory (overrides config default)
            
        Returns:
            dict: Summary of processing results
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video path not found: {video_path}")

        video_stem = video_path.stem
        # Use provided output_dir if specified, otherwise fall back to config default
        if output_dir is not None:
            base_out = self.ensure_dir(Path(output_dir))
        else:
            base_out = self.ensure_dir(Path(self.config.OUTPUT_DIR) / video_stem)
        
        # Only create frames directory if frames are being saved
        frames_out = None
        if self.config.SAVE_FRAMES:
            frames_out = self.ensure_dir(base_out / "frames")
        
        json_path = base_out / "predictions.json"
        jsonl_path = base_out / "predictions.jsonl"

        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Reset state for new video
        self.processed_frames = 0
        self.frames_json = []

        try:
            self._process_frames(cap, fps, frames_out, jsonl_path)
        finally:
            cap.release()

        # Save final results
        self._save_results(video_path, json_path, base_out, output_dir)
        
        return {
            "video_path": str(video_path),
            "output_dir": str(base_out),
            "processed_frames": self.processed_frames,
            "total_frames": len(self.frames_json)
        }
    
    def _process_frames(self, cap, fps, frames_out, jsonl_path):
        """Internal method to process video frames."""
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % self.config.FRAME_INTERVAL == 0:
                self._process_single_frame(frame, frame_num, fps, frames_out, jsonl_path)
                self.processed_frames += 1
                
                if self.processed_frames % 50 == 0:
                    print(f"Processed {self.processed_frames} sampled frames (last={frame_num})")

            frame_num += 1
    
    def _process_single_frame(self, frame, frame_num, fps, frames_out, jsonl_path):
        """Process a single frame and save results."""
        timestamp = str(timedelta(seconds=int(frame_num / fps)))
        
        # Only create frame path if frames are being saved
        frame_path = None
        if frames_out is not None:
            frame_filename = f"frame_{frame_num:06d}.jpg"
            frame_path = frames_out / frame_filename

        try:
            faces = self.analyze_frame(frame)
            num_faces = len(faces)

            should_save = self.config.SAVE_FRAMES and (num_faces > 0 if self.config.SAVE_ONLY_DETECTIONS else True)
            if should_save and frame_path is not None:
                cv2.imwrite(str(frame_path), frame)

            frame_record = {
                "frame_num": frame_num,
                "timestamp": timestamp,
                "frame_path": str(frame_path) if should_save and frame_path is not None else "",
                "num_faces": num_faces,
                "faces": faces,
                "error": ""
            }

        except Exception as e:
            # optional save on error
            if self.config.SAVE_FRAMES and not self.config.SAVE_ONLY_DETECTIONS and frame_path is not None:
                cv2.imwrite(str(frame_path), frame)

            frame_record = {
                "frame_num": frame_num,
                "timestamp": timestamp,
                "frame_path": str(frame_path) if self.config.SAVE_FRAMES and not self.config.SAVE_ONLY_DETECTIONS and frame_path is not None else "",
                "num_faces": 0,
                "faces": [],
                "error": str(e)
            }

        if self.config.WRITE_JSON:
            self.frames_json.append(frame_record)

        
    
    def _save_results(self, video_path, json_path, base_out, output_dir=None):
        """Save final results to JSON file."""
        if self.config.WRITE_JSON:
            # Only save predictions.json if using default config output directory
            # If custom output_dir is provided, let the calling code handle output
            if output_dir is None:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "video_path": str(video_path),
                        "frame_interval": self.config.FRAME_INTERVAL,
                        "results": self.frames_json
                    }, f, ensure_ascii=False, indent=2)
    
    def get_processing_summary(self):
        """Get summary of current processing session."""
        # Count age classifications
        age_counts = {"minor": 0, "adult": 0, "senior": 0, "unknown": 0}
        for frame in self.frames_json:
            for face in frame.get("faces", []):
                age_class = face.get("age_classification", "unknown")
                age_counts[age_class] += 1
        
        return {
            "processed_frames": self.processed_frames,
            "total_faces_detected": sum(len(frame.get("faces", [])) for frame in self.frames_json),
            "frames_with_errors": sum(1 for frame in self.frames_json if frame.get("error")),
            "age_classifications": age_counts
        }


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python src/deepfacedetect.py /path/to/video.mp4")
        sys.exit(1)

    try:
        detector = FaceAgeDetector()
        results = detector.process_video(sys.argv[1])
        
        print(f"\n✅ Done. Outputs saved to: {results['output_dir']}")
        print(f"- Processed {results['processed_frames']} frames")
        
        summary = detector.get_processing_summary()
        print(f"- Total faces detected: {summary['total_faces_detected']}")
        if summary['frames_with_errors'] > 0:
            print(f"- Frames with errors: {summary['frames_with_errors']}")
        
        # Display age classification statistics
        age_stats = summary.get('age_classifications', {})
        if age_stats:
            print("\nAge Classification Summary:")
            for age_class, count in age_stats.items():
                if count > 0:
                    print(f"  - {age_class.capitalize()}: {count}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
