import warnings
warnings.filterwarnings("ignore")  # Catch-all for all warnings

import ray
import os
import sys
import tempfile
import json
import torch
import librosa
import soundfile as sf
# Removed subprocess and ffmpeg imports as we no longer extract audio from video
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from presidio_analyzer import AnalyzerEngine

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
from utils.logger import get_logger

# HF_TOKEN=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
os.environ["HF_TOKEN"] = "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"
# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
SAMPLE_RATE = 16000
MIN_SEGMENT_DURATION = 0.1
SPEAKER_MERGE_THRESHOLD = 0.5

logger = get_logger("audio_diarization_pii")

def get_device():
    """Get optimal device for processing"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
        return "cuda"
    else:
        logger.info("Using CPU")
        return "cpu"

def load_models(device):
    """Load all required models based on device"""
    try:
        if not HF_TOKEN:
            logger.warning("HF_TOKEN not found. Set it with: export HF_TOKEN=your_token")
            logger.info("Trying to load models without token (may fail for gated models)")
        
        # Model selection based on device
        if device == "cuda":
            diarization_models = [
                "pyannote/speaker-diarization-3.1",
                "pyannote/speaker-diarization-3.0"
            ]
            whisper_model = "large-v3"
            compute_type = "float16"
        else:
            diarization_models = [
                "pyannote/speaker-diarization-3.0", 
                "pyannote/speaker-diarization-2.1"  # Fallback public model
            ]
            whisper_model = "base"
            compute_type = "int8"
        
        # Try loading diarization models with fallback
        diarization_pipeline = None
        for model_name in diarization_models:
            try:
                logger.info(f"Trying to load diarization model: {model_name}")
                diarization_pipeline = Pipeline.from_pretrained(
                    model_name, 
                    use_auth_token=HF_TOKEN if HF_TOKEN else None
                )
                if device == "cuda":
                    diarization_pipeline.to(torch.device("cuda"))
                logger.info(f"Successfully loaded: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if diarization_pipeline is None:
            raise Exception("Could not load any diarization model")
        
        # Load Whisper
        logger.info(f"Loading Whisper model: {whisper_model}")
        whisper = WhisperModel(whisper_model, device=device, compute_type=compute_type)
        
        # Load PII analyzer
        logger.info("Loading PII analyzer")
        pii_analyzer = AnalyzerEngine()
        
        logger.info("All models loaded successfully")
        return diarization_pipeline, whisper, pii_analyzer
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

def validate_audio_file(audio_path: str) -> bool:
    """Validate that the audio file exists and is readable"""
    try:
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False
        
        if not audio_path.lower().endswith('.wav'):
            logger.warning(f"File is not a WAV file: {audio_path}")
            # Still try to process it
        
        # Test if we can load it
        audio_array, sample_rate = librosa.load(audio_path, sr=None, duration=1.0)
        if len(audio_array) == 0:
            logger.error(f"Audio file appears to be empty: {audio_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Audio file validation failed: {e}")
        return False

def prepare_audio_file(audio_path: str) -> str:
    """Prepare audio file for processing (convert to correct format if needed)"""
    try:
        # Load audio and convert to the required format
        audio_array, original_sr = librosa.load(audio_path, sr=None)
        
        # If already in correct format, return original path
        if original_sr == SAMPLE_RATE:
            return audio_path
        
        # Convert to required sample rate and save as temp file
        audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=SAMPLE_RATE)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_resampled, SAMPLE_RATE)
            logger.info(f"Converted audio from {original_sr}Hz to {SAMPLE_RATE}Hz")
            return temp_file.name
            
    except Exception as e:
        logger.error(f"Audio preparation failed: {e}")
        return None


def merge_consecutive_segments(raw_segments, speaker_order):
    """Merge consecutive segments from same speaker"""
    merged = []
    current_speaker = None
    current_start = None
    current_end = None

    for start, end, speaker in raw_segments:
        mapped_speaker = speaker_order[speaker]
        
        if current_speaker is None:
            current_speaker = mapped_speaker
            current_start = start
            current_end = end
        elif mapped_speaker == current_speaker and (start - current_end) < SPEAKER_MERGE_THRESHOLD:
            current_end = end
        else:
            if (current_end - current_start) >= MIN_SEGMENT_DURATION:
                merged.append((current_start, current_end, current_speaker))
            current_speaker = mapped_speaker
            current_start = start
            current_end = end

    if current_speaker and (current_end - current_start) >= MIN_SEGMENT_DURATION:
        merged.append((current_start, current_end, current_speaker))
        
    return merged

def format_pii_results(pii_results, transcript, segment_start_time=0.0):
    """Format PII detection results with global timestamps"""
    return [{
        "entity_type": result.entity_type,
        "text": transcript[result.start:result.end],
        "start_char": result.start,
        "end_char": result.end,
        "confidence": round(result.score, 3),
        "segment_start_time": segment_start_time  # Track which audio segment this came from
    } for result in pii_results]

def compile_separate_outputs(speakers_data):
    """
    Compile the three separate outputs from speakers_data
    
    Returns:
        tuple: (full_transcript, diarization_segments, all_pii_detections)
    """
    # 1. TRANSCRIPT - Combine all segments in chronological order
    sorted_segments = sorted(speakers_data, key=lambda x: x["start_time"])
    full_transcript = " ".join([seg["transcript"] for seg in sorted_segments if seg["transcript"].strip()])
    
    # 2. DIARIZATION - Speaker timeline information
    diarization_segments = []
    for seg in sorted_segments:
        diarization_segments.append({
            "speaker": seg["speaker_id"],
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "duration": seg["duration"]
        })
    
    # 3. PII DETECTIONS - Flatten all PII detections with context
    all_pii_detections = []
    for seg in speakers_data:
        for pii in seg["pii_detections"]:
            pii_with_context = pii.copy()
            pii_with_context.update({
                "speaker": seg["speaker_id"],
                "audio_start_time": seg["start_time"],
                "audio_end_time": seg["end_time"]
            })
            all_pii_detections.append(pii_with_context)
    
    return full_transcript, diarization_segments, all_pii_detections

@ray.remote
class AudioDiarizationActor:
    def __init__(self):
        """Initialize models once per actor"""
        logger.info("Initializing AudioDiarizationActor")
        try:
            self.device = get_device()
            self.diarization_pipeline, self.whisper_model, self.pii_analyzer = load_models(self.device)
            self.initialized = True
            logger.info("AudioDiarizationActor ready")
        except Exception as e:
            logger.error(f"Failed to initialize AudioDiarizationActor: {e}")
            self.initialized = False
            raise
    
    def process_audio_files(self, audio_paths):
        """Process multiple audio files for diarization and PII detection"""
        if not self.initialized:
            logger.error("Actor not properly initialized")
            return []
            
        results = []
        
        for audio_path in audio_paths:
            try:
                logger.info(f"Processing: {os.path.basename(audio_path)}")
                start_time = datetime.now()
                
                # Validate audio file
                if not validate_audio_file(audio_path):
                    logger.error(f"Skipping invalid audio file: {audio_path}")
                    continue
                
                # Prepare audio file (convert if needed)
                processed_audio_path = prepare_audio_file(audio_path)
                if not processed_audio_path:
                    logger.error(f"Failed to prepare audio file: {audio_path}")
                    continue
                
                # Track if we created a temporary file
                temp_file_created = (processed_audio_path != audio_path)
                
                # Load audio once for memory efficiency
                audio_array, sample_rate = librosa.load(processed_audio_path, sr=SAMPLE_RATE)
                duration = len(audio_array) / sample_rate
                
                # Perform diarization
                diarization_start = datetime.now()
                diarization = self.diarization_pipeline(processed_audio_path)
                diarization_time = (datetime.now() - diarization_start).total_seconds()
                
                # Parse and organize segments
                raw_segments = [(seg.start, seg.end, speaker) 
                              for seg, _, speaker in diarization.itertracks(yield_label=True)]
                raw_segments.sort(key=lambda x: x[0])
                
                # Create speaker mapping
                speaker_order = {}
                for i, (_, _, speaker) in enumerate(raw_segments):
                    if speaker not in speaker_order:
                        speaker_order[speaker] = f"Speaker {len(speaker_order) + 1}"
                
                # Merge consecutive segments
                merged_segments = merge_consecutive_segments(raw_segments, speaker_order)
                
                # Process each segment
                speakers_data = []
                transcription_start = datetime.now()
                
                for start, end, speaker in merged_segments:
                    # Extract audio segment from memory
                    start_sample = int(start * sample_rate)
                    end_sample = int(end * sample_rate)
                    segment_audio = audio_array[start_sample:end_sample]
                    
                    # Transcribe segment
                    transcript = self.transcribe_segment(segment_audio, sample_rate)
                    
                    # Detect PII
                    pii_results = self.pii_analyzer.analyze(text=transcript, language='en')
                    pii_detections = format_pii_results(pii_results, transcript, start)
                    
                    speakers_data.append({
                        "speaker_id": speaker,
                        "start_time": start,
                        "end_time": end,
                        "duration": end - start,
                        "transcript": transcript.strip(),
                        "pii_detections": pii_detections,
                        "has_pii": len(pii_detections) > 0
                    })
                
                transcription_time = (datetime.now() - transcription_start).total_seconds()
                total_time = (datetime.now() - start_time).total_seconds()
                
                # COMPILE THE THREE SEPARATE OUTPUTS
                full_transcript, diarization_segments, all_pii_detections = compile_separate_outputs(speakers_data)
                
                # NEW STRUCTURE - Three separate outputs
                audio_result = {
                    "audio_path": audio_path,
                    "audio_name": os.path.basename(audio_path),
                    "audio_duration": duration,
                    
                    # === THE THREE MAIN OUTPUTS ===
                    "transcript": full_transcript,
                    "diarization": diarization_segments,
                    "pii_detections": all_pii_detections,
                    
                    # === SUMMARY STATISTICS ===
                    "summary": {
                        "total_speakers": len(speaker_order),
                        "total_segments": len(speakers_data),
                        "segments_with_pii": len([seg for seg in speakers_data if seg["has_pii"]]),
                        "total_pii_detections": len(all_pii_detections)
                    },
                    
                    # === PROCESSING METADATA ===
                    "processing_stats": {
                        "total_processing_time": total_time,
                        "diarization_time": diarization_time,
                        "transcription_time": transcription_time,
                        "device_used": self.device
                    }
                }
                
                results.append(audio_result)
                logger.info(f"Completed {os.path.basename(audio_path)} in {total_time:.2f}s")
                
                # Cleanup temporary file if we created one
                if temp_file_created and os.path.exists(processed_audio_path):
                    os.unlink(processed_audio_path)
                    
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                continue
        
        logger.info(f"Processed {len(results)} audio files successfully")
        return results
    
    def transcribe_segment(self, audio_segment, sample_rate):
        """Transcribe audio segment using Whisper"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_segment, sample_rate)
                segments, _ = self.whisper_model.transcribe(temp_file.name)
                transcript = " ".join([segment.text for segment in segments])
                os.unlink(temp_file.name)
                return transcript
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

# Integration Testing changes begin

# The pipeline code expects the results to be saved in specified output directory, so commenting the old section and using updated method
# @ray.remote
# def process_audio_diarization(shard_paths):
#     """Factory function for Ray pipeline integration"""
#     actor = AudioDiarizationActor.remote()
#     return ray.get(actor.process_shards.remote(shard_paths))


@ray.remote
def process_audio_diarization(audio_paths, output_dir=None):
    """Factory function for Ray pipeline integration - now works with audio files directly"""
    actor = AudioDiarizationActor.remote()
    results = ray.get(actor.process_audio_files.remote(audio_paths))
    
    # Save results to output directory if provided
    if output_dir and results:
        os.makedirs(output_dir, exist_ok=True)
        
        for result in results:
            audio_name = result["audio_name"]
            base_name = os.path.splitext(audio_name)[0]
            
            # Save transcript
            transcript_file = os.path.join(output_dir, f"{base_name}_transcript.txt")
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(result["transcript"])
            
            # Save diarization segments
            diarization_file = os.path.join(output_dir, f"{base_name}_diarization.json")
            with open(diarization_file, 'w', encoding='utf-8') as f:
                json.dump(result["diarization"], f, indent=2)
            
            # Save PII detections
            pii_file = os.path.join(output_dir, f"{base_name}_pii_detections.json")
            with open(pii_file, 'w', encoding='utf-8') as f:
                json.dump(result["pii_detections"], f, indent=2)
            
            # Save complete results
            complete_file = os.path.join(output_dir, f"{base_name}_complete_results.json")
            with open(complete_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
    
    return results
# Integration Testing changes end

if __name__ == "__main__":
    # Check if HF_TOKEN is set
    if not HF_TOKEN:
        print("⚠️  HF_TOKEN not found. Please set it:")
        print("export HF_TOKEN=your_huggingface_token")
        print("\nWill try to proceed with fallback models...")
    
    # Test with audio files
    # You can specify multiple WAV files to test
    test_audio_files = ['/home/nvcoe_admin/arian/insta360-video-activity-segmentation/test_files/cleaning-surfaces-20250819_1325-audio.WAV']
    
    if not test_audio_files:
        print("📝 To test the script, add WAV file paths to the test_audio_files list")
        print("Example:")
        print('test_audio_files = ["/path/to/your/audio.wav"]')
        sys.exit(0)
    
    logger.info(f"Testing with {len(test_audio_files)} audio files")
    
    try:
        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init()
            
        results = ray.get(process_audio_diarization.remote(test_audio_files))
        
        if results:
            print(f"✅ Success! Processed {len(results)} audio files")
            
            for i, result in enumerate(results):
                print(f"\n{'='*60}")
                print(f"RESULTS FOR: {result['audio_name']}")
                print(f"{'='*60}")
                
                print("\n=== TRANSCRIPT ===")
                print(result["transcript"])
                
                print("\n=== DIARIZATION ===")
                for seg in result["diarization"]:
                    print(f"{seg['speaker']}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s")
                
                print("\n=== PII DETECTIONS ===")
                if result["pii_detections"]:
                    for pii in result["pii_detections"]:
                        print(f"{pii['entity_type']}: '{pii['text']}' (confidence: {pii['confidence']})")
                else:
                    print("No PII detected")
                
                print(f"\n=== SUMMARY ===")
                summary = result["summary"]
                print(f"Speakers: {summary['total_speakers']}")
                print(f"Segments: {summary['total_segments']}")
                print(f"PII Detections: {summary['total_pii_detections']}")
                print(f"Processing Time: {result['processing_stats']['total_processing_time']:.2f}s")
                
                # Optionally print full result (commented out as it's very verbose)
                # print("\n=== FULL RESULT ===")
                # print(json.dumps(result, indent=2))
        else:
            print("❌ No results returned")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()