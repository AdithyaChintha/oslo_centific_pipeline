import ray
import os
import json
import sys
import time
from pathlib import Path

# Setup paths - go up one level to project root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent  # Go up from test/ to project root
sys.path.append(str(PROJECT_ROOT))

# Import the Ray job
from ray_jobs.audio_diarization_pii import process_audio_diarization

def test_audio_diarization():
    """Test the audio diarization Ray job with your 2 video files"""
    
    print("🚀 Starting Audio Diarization Ray Job Test")
    print("=" * 50)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
        print("✅ Ray initialized")
    else:
        print("✅ Ray already initialized")
    
    # Your 2 test files
    test_files = [
        # "/home/nvcoe_admin/code/oslo/oslo-motion-energy-analysis/input/VID_20250720_152154_00_011.mp4",
        "/home/nvcoe_admin/code/oslo/pii_detection_in_audio/introduce_yourself.mp4"
    ]
    
    # Filter only existing files
    existing_files = []
    for file in test_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            print(f"⚠️  File not found: {file}")
    
    if not existing_files:
        print("❌ No test video files found")
        return False
    
    print(f"📁 Testing {len(existing_files)} files:")
    for i, file in enumerate(existing_files, 1):
        print(f"   {i}. {os.path.basename(file)}")
    print()
    
    # Test results
    results = []
    total_start_time = time.time()
    
    for i, test_file in enumerate(existing_files, 1):
        print(f"🎯 Test {i}/{len(existing_files)}: {os.path.basename(test_file)}")
        print("-" * 40)
        
        start_time = time.time()
        try:
            # Call the Ray job (exactly how it will be used in production)
            result = ray.get(process_audio_diarization.remote([test_file]))
            processing_time = time.time() - start_time
            
            if result and len(result) > 0:
                shard_result = result[0]
                
                print(f"✅ SUCCESS in {processing_time:.2f}s")
                print(f"📊 Results:")
                print(f"   - Duration: {shard_result.get('audio_duration', 0):.1f}s")
                print(f"   - Speakers: {shard_result.get('summary', {}).get('total_speakers', 0)}")
                print(f"   - Segments: {shard_result.get('summary', {}).get('total_segments', 0)}")
                print(f"   - PII found: {shard_result.get('summary', {}).get('total_pii_detections', 0)}")
                print(f"   - Device: {shard_result.get('processing_stats', {}).get('device_used', 'unknown')}")
                
                # Show transcript and PII if any
                transcript = shard_result.get('transcript', '')
                pii_detections = shard_result.get('pii_detections', [])
                diarization = shard_result.get('diarization', [])
                
                if transcript.strip():
                    print(f"📝 Transcript: \"{transcript[:100]}{'...' if len(transcript) > 100 else ''}\"")
                    
                    # Show speaker segments
                    if diarization:
                        print(f"🎙️ Speaker segments:")
                        for seg in diarization[:3]:  # Show first 3
                            print(f"   - {seg['speaker']}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s")
                        if len(diarization) > 3:
                            print(f"   ... and {len(diarization) - 3} more segments")
                    
                    # Show PII detections
                    if pii_detections:
                        print(f"🔍 PII found:")
                        for pii in pii_detections[:3]:  # Show first 3
                            print(f"   - {pii['entity_type']}: '{pii['text']}' (confidence: {pii['confidence']:.2f}) - {pii['speaker']}")
                        if len(pii_detections) > 3:
                            print(f"   ... and {len(pii_detections) - 3} more PII detections")
                    else:
                        print("🔍 No PII detected")
                else:
                    print("📝 No speech detected")
                
                results.append({
                    'file': os.path.basename(test_file),
                    'success': True,
                    'processing_time': processing_time,
                    'speakers': shard_result.get('summary', {}).get('total_speakers', 0),
                    'pii_count': shard_result.get('summary', {}).get('total_pii_detections', 0)
                })
                
            else:
                print(f"❌ FAILED - No results returned")
                results.append({
                    'file': os.path.basename(test_file),
                    'success': False,
                    'error': 'No results returned'
                })
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ FAILED after {processing_time:.2f}s")
            print(f"   Error: {str(e)}")
            results.append({
                'file': os.path.basename(test_file),
                'success': False,
                'error': str(e)
            })
        
        print()
    
    # Final Summary
    total_time = time.time() - total_start_time
    successful = len([r for r in results if r['success']])
    
    print("🏁 FINAL RESULTS")
    print("=" * 50)
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{i}. {result['file']}: {status}")
        if result['success']:
            print(f"   Speakers: {result['speakers']}, PII: {result['pii_count']}, Time: {result.get('processing_time', 0):.1f}s")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
    
    print(f"\nSummary: {successful}/{len(results)} tests passed in {total_time:.1f}s")
    
    return successful == len(results)

if __name__ == "__main__":
    print("🎬 Simple Audio Diarization Test (2 Files)")
    print("=" * 50)
    
    # Check HF token
    if not os.getenv("HF_TOKEN"):
        print("❌ HF_TOKEN not set!")
        print("Run: export HF_TOKEN=your_token")
        sys.exit(1)
    else:
        print(f"✅ HF_TOKEN found: {os.getenv('HF_TOKEN')[:7]}...")
    
    try:
        success = test_audio_diarization()
        
        if success:
            print("\n🎉 ALL TESTS PASSED! Ray job is ready.")
        else:
            print("\n⚠️  Some tests failed. Check errors above.")
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    except Exception as e:
        print(f"\n💥 Test error: {e}")
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("🔧 Ray shutdown")