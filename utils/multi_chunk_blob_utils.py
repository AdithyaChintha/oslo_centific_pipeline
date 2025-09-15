import re
import os
from typing import Optional, Dict, List, Tuple
from utils.logger import get_logger
import json
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import time
from datetime import datetime
import ray
logger = get_logger("ChunkFilenameParser")

def parse_chunk_filename_complete(filename: str) -> Optional[Dict]:
    """
    Parse filenames like:
      - 11_515f...c5ab_audio1_working-on-laptop_20250907T00:00:00.000Z122100_01_audio.WAV
      - 11_515f...c5ab_video_working-on-laptop_20250907T00:00:00.000Z122100_05_video.insv
      - (NOT parsed here) ..._metadata.json  (handled separately)
    Returns a dict or None if unrecognized.
    """
    if not filename:
        return None

    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    ext_lower = ext.lower()

    # Only parse audio/video files here; metadata is discovered elsewhere
    if ext_lower not in {'.wav', '.insv', '.mp4'}:
        return None

    # Tail-first: optional "_<chunk>_" then "<type>" at end of basename (before extension)
    tail_re = re.compile(
        r'^(?P<head>.+?)'
        r'(?:_(?P<chunk>\d{2}))?'             # optional chunk number (01..99)
        r'_(?P<chunk_type>audio|video)$',     # required tail token
        re.IGNORECASE
    )
    m_tail = tail_re.match(name)
    if not m_tail:
        return None

    head = m_tail.group('head')
    chunk_str = m_tail.group('chunk')
    chunk_type = m_tail.group('chunk_type').lower()

    # Head parse:
    # <session_index>_<uuid> [ _<stream> ] [ _<label> ] [ _<timestamp> ]
    head_re = re.compile(
        r'^'
        r'(?P<session_index>\d+)_'
        r'(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})'
        r'(?:_(?P<stream>(?:audio|video)\d*))?'  # optional "audio1"/"video2"/"video"
        r'(?:_(?P<label>[A-Za-z0-9_-]+))?'       # optional label (kebab/underscore ok)
        r'(?:_(?P<timestamp>\d{8}T\d{2}:\d{2}:\d{2}\.\d{3}Z\d+))?'  # optional timestamp
        r'$',
        re.IGNORECASE
    )
    m_head = head_re.match(head)
    if not m_head:
        return None

    session_index = m_head.group('session_index')
    uuid = m_head.group('uuid')
    stream = m_head.group('stream')
    label = m_head.group('label') or "unknown"
    timestamp = m_head.group('timestamp')

    # Validate extension vs type
    if chunk_type == 'audio' and ext_lower not in {'.wav'}:
        return None
    if chunk_type == 'video' and ext_lower not in {'.insv', '.mp4'}:
        return None

    chunk_number = int(chunk_str) if chunk_str is not None else None

    # Session ID used everywhere (must match what metadata extractor returns)
    session_id = f"{session_index}-{uuid}-{label.lower()}"

    return {
        "file_name": filename,
        "session_id": session_id,
        "session_index": session_index,
        "uuid": uuid,
        "stream": stream,                   # e.g., "audio1"
        "label": label.lower(),
        "timestamp": timestamp,
        "chunk_number": chunk_number,       # int or None
        "chunk_type": chunk_type,           # "audio" | "video"
        "is_last_chunk": False,             # will be filled later after completion detection
        "format": "underscore-new",
        "extension": ext_lower
    }


def group_chunks_by_session(parsed_chunks: list) -> Dict[str, Dict]:
    """
    Group parsed chunks by session ID and compute counts & gaps.
    'is_last_chunk' is finalized later (after metadata presence is known).
    """
    sessions: Dict[str, Dict] = {}

    for ch in parsed_chunks:
        if not ch:
            continue

        session_id = ch["session_id"]
        chunk_type = ch["chunk_type"]
        chunk_number = ch.get("chunk_number")

        if session_id not in sessions:
            sessions[session_id] = {
                "session_id": session_id,
                "video_chunks": [],
                "audio_chunks": [],
                "video_sequences": set(),
                "audio_sequences": set(),
                "completion_detected": False,   # set later when metadata is merged
                "completion_method": None,
                "session_metadata": None,
                "expected_files": None,
                # diagnostic fields
                "video_sequence_gaps": [],
                "has_sequence_gaps": False,
                "has_audio_video_mismatch": False,
            }

        session = sessions[session_id]

        # Store the chunk
        entry = {
            "chunk_number": chunk_number,
            "file_name": ch["file_name"],
            "chunk_type": chunk_type,
            "is_last_chunk": False,  # will be set after metadata merge
            "chunk_data": ch,
        }

        if chunk_type == "video":
            session["video_chunks"].append(entry)
            if chunk_number is not None:
                session["video_sequences"].add(int(chunk_number))
        elif chunk_type == "audio":
            session["audio_chunks"].append(entry)
            if chunk_number is not None:
                session["audio_sequences"].add(int(chunk_number))

    # Compute counts & gaps
    for session_id, session in sessions.items():
        session["video_count"] = len(session["video_chunks"])
        session["audio_count"] = len(session["audio_chunks"])

        # Check gaps (video only; you can do the same for audio if helpful)
        if session["video_sequences"]:
            vmin = min(session["video_sequences"])
            vmax = max(session["video_sequences"])
            expected = set(range(vmin, vmax + 1))
            gaps = sorted(expected - session["video_sequences"])
            session["video_sequence_gaps"] = gaps
            session["has_sequence_gaps"] = len(gaps) > 0

        # Audio/video length mismatch (by distinct indices)
        session["has_audio_video_mismatch"] = (
            len(session["video_sequences"]) != len(session["audio_sequences"])
        )

        # Sort sequences for JSON friendliness
        session["video_sequences"] = sorted(session["video_sequences"])
        session["audio_sequences"] = sorted(session["audio_sequences"])

    return sessions


def parse_completion_metadata_json(completion_content: str) -> Optional[Dict]:
    """
    Parse the completion metadata JSON with actual format.
    
    Expected format: 11_515fb09c-f12f-48ee-91e7-b21c9f4ac5ab_activity_working-on-laptop_20250907T00:00:00.000Z122100_metadata.json
    """
    try:
        metadata = json.loads(completion_content)
        
        # Validate required fields
        required_fields = ["id", "home_id", "participant_id", "activity", "start_datetime", "end_datetime"]
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if missing_fields:
            logger.error(f"❌ Missing required fields in metadata JSON: {missing_fields}")
            return None
        
        # Extract session information from the metadata
        session_info = {
            "session_id": f"{metadata['home_id']}-{metadata['participant_id']}-{metadata.get('specific_activity', 'unknown').lower().replace(' ', '-')}",
            "completion_detected": True,
            "completion_method": "metadata_json",
            "session_metadata": {
                "id": metadata["id"],
                "home_id": metadata["home_id"],
                "participant_id": metadata["participant_id"],
                "activity": metadata["activity"],
                "specific_activity": metadata.get("specific_activity", ""),
                "domain": metadata.get("domain", ""),
                "start_datetime": metadata["start_datetime"],
                "end_datetime": metadata["end_datetime"],
                "duration_minutes": metadata.get("duration_minutes"),
                "room": metadata.get("room", ""),
                "day_night": metadata.get("day_night", ""),
                "first_name": metadata.get("first_name", ""),
                "last_name": metadata.get("last_name", ""),
                "email": metadata.get("email", ""),
                "exported_at": metadata.get("exported_at", "")
            },
            "expected_files": {
                "video_name": metadata.get("video_name", ""),
                "audio_name": metadata.get("audio_name", "")
            }
        }
        
        logger.info(f"✅ Parsed metadata for session: {session_info['session_id']}")
        return session_info
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in completion metadata: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to parse completion metadata: {e}")
        return None

def extract_session_id_from_metadata_filename(filename: str) -> Optional[str]:
    """
    Extract session ID from metadata filename.

    Accepts both:
      11_<uuid>_activity_<label>_<timestamp>_metadata.json
      11_<uuid>_<label>_<timestamp>_metadata.json

    Returns: "11-<uuid>-<label>"
    """
    if not filename.lower().endswith("_metadata.json"):
        return None

    base = os.path.basename(filename)[:-len("_metadata.json")]  # strip suffix

    # Common head:
    head = (
        r'^(?P<home_id>\d+)_'
        r'(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})_'
    )

    # Pattern A: with literal "activity"
    pat_a = re.compile(
        head +
        r'activity_'
        r'(?P<label>[A-Za-z0-9_-]+)_'
        r'(?P<ts>\d{8}T\d{2}:\d{2}:\d{2}\.\d{3}Z\d+)$',
        re.IGNORECASE
    )
    # Pattern B: without literal "activity"
    pat_b = re.compile(
        head +
        r'(?P<label>[A-Za-z0-9_-]+)_'
        r'(?P<ts>\d{8}T\d{2}:\d{2}:\d{2}\.\d{3}Z\d+)$',
        re.IGNORECASE
    )

    m = pat_a.match(base) or pat_b.match(base)
    if not m:
        logger.warning(f"⚠️ Unrecognized metadata filename format: {filename}")
        return None

    home_id = m.group("home_id")
    uuid = m.group("uuid")
    label = m.group("label").lower()
    return f"{home_id}-{uuid}-{label}"


def discover_chunks_in_prefix(blob_service_client: BlobServiceClient, 
                            container_name: str, 
                            input_prefix: str) -> List[Dict]:
    """
    Discover all chunk files in a blob prefix.
    This is the most basic discovery - just finding and parsing filenames.
    """
    logger.info(f"🔍 Discovering chunks in {container_name}/{input_prefix}")
    
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs(name_starts_with=input_prefix)
        
        discovered_chunks = []
        
        for blob in blob_list:
            blob_name = blob.name
            file_name = blob_name.split('/')[-1]  # Get just the filename
            
            # Parse chunk filename
            parsed_chunk = parse_chunk_filename_complete(file_name)
            
            if parsed_chunk:
                chunk_info = {
                    "blob_name": blob_name,
                    "file_name": file_name,
                    "size": blob.size,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                    **parsed_chunk
                }
                discovered_chunks.append(chunk_info)
                logger.debug(f"  📄 Found chunk: {file_name} -> {parsed_chunk['session_id']}")
            else:
                logger.debug(f"  ⏭️ Skipping non-chunk file: {file_name}")
        
        logger.info(f"✅ Discovered {len(discovered_chunks)} chunk files")
        return discovered_chunks
        
    except Exception as e:
        logger.error(f"❌ Failed to discover chunks: {e}")
        return []

def discover_completion_metadata(blob_service_client: BlobServiceClient,
                               container_name: str,
                               input_prefix: str) -> Dict[str, Dict]:
    """
    Discover completion metadata JSON files.
    """
    logger.info(f"🔍 Looking for completion metadata files in {container_name}/{input_prefix}")
    
    completion_metadata = {}
    
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs(name_starts_with=input_prefix)
        
        for blob in blob_list:
            if blob.name.endswith("_metadata.json"):
                file_name = blob.name.split('/')[-1]
                logger.info(f"📄 Found metadata file: {file_name}")
                
                # Extract session ID from filename
                session_id = extract_session_id_from_metadata_filename(file_name)
                
                if session_id:
                    # Download and parse metadata
                    blob_client = blob_service_client.get_blob_client(
                        container=container_name, blob=blob.name
                    )
                    content = blob_client.download_blob().readall().decode('utf-8')
                    parsed_metadata = parse_completion_metadata_json(content)
                    
                    if parsed_metadata:
                        completion_metadata[session_id] = {
                            "blob_name": blob.name,
                            "file_name": file_name,
                            **parsed_metadata
                        }
                        logger.info(f"✅ Parsed metadata for session: {session_id}")
    
    except Exception as e:
        logger.error(f"❌ Failed to discover completion metadata: {e}")
    
    logger.info(f"✅ Found metadata for {len(completion_metadata)} sessions")
    return completion_metadata

def is_walkthrough_session(session_data: Dict) -> bool:
    """
    Simple walkthrough detection - check filenames and metadata for 'walkthrough'
    """
    # Check chunk filenames
    all_chunks = session_data.get("video_chunks", []) + session_data.get("audio_chunks", [])
    for chunk_info in all_chunks:
        filename = chunk_info.get("file_name", "")
        if "walkthrough" in filename.lower():
            return True
    
    # Check session metadata
    session_metadata = session_data.get("session_metadata", {})
    if session_metadata:
        activity = session_metadata.get("activity", "").lower()
        specific_activity = session_metadata.get("specific_activity", "").lower()
        if "walkthrough" in activity or "walkthrough" in specific_activity:
            return True
    
    return False

def discover_sessions_basic(blob_service_client: BlobServiceClient,
                          container_name: str, 
                          input_prefix: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Basic session discovery combining chunks and metadata.
    Returns: (ready_sessions, not_ready_sessions)
    """
    logger.info(f"🔍 Basic session discovery in {container_name}/{input_prefix}")
    
    # Step 1: Discover chunks
    discovered_chunks = discover_chunks_in_prefix(blob_service_client, container_name, input_prefix)
    
    # Step 2: Group chunks by session
    sessions = group_chunks_by_session(discovered_chunks)
    
    # Step 3: Discover completion metadata
    completion_metadata = discover_completion_metadata(blob_service_client, container_name, input_prefix)
    
    # Step 4: Merge completion metadata with sessions
    for session_id, metadata in completion_metadata.items():
        if session_id in sessions:
            sessions[session_id]["completion_detected"] = True
            sessions[session_id]["completion_method"] = "metadata_json"
            sessions[session_id]["session_metadata"] = metadata["session_metadata"]
            sessions[session_id]["expected_files"] = metadata["expected_files"]
        else:
            # Create session from metadata only
            sessions[session_id] = {
                "session_id": session_id,
                "video_chunks": [],
                "audio_chunks": [],
                "video_count": 0,
                "audio_count": 0,
                "completion_detected": True,
                "completion_method": "metadata_json",
                "session_metadata": metadata["session_metadata"],
                "expected_files": metadata["expected_files"],
                "has_sequence_gaps": False,
                "has_audio_video_mismatch": False
            }
    
    # Step 5: Separate ready vs not-ready sessions
    ready_sessions = {}
    not_ready_sessions = {}
    # --- AFTER Step 4 (metadata merged); BEFORE Step 5 (ready/not-ready split) ---

    # Finalize last-chunk flags ONLY for sessions that have completion metadata
    for sid, sess in sessions.items():
        if not sess.get("completion_detected"):
            continue

        # Video: mark only the max chunk_number as last
        v_nums = [c.get("chunk_number") for c in sess.get("video_chunks", []) if isinstance(c.get("chunk_number"), int)]
        if v_nums:
            v_max = max(v_nums)
            for c in sess["video_chunks"]:
                c["is_last_chunk"] = (c.get("chunk_number") == v_max)

        # Audio: mark only the max chunk_number as last
        a_nums = [c.get("chunk_number") for c in sess.get("audio_chunks", []) if isinstance(c.get("chunk_number"), int)]
        if a_nums:
            a_max = max(a_nums)
            for c in sess["audio_chunks"]:
                c["is_last_chunk"] = (c.get("chunk_number") == a_max)

    for session_id, session in sessions.items():
        if session["completion_detected"]:
            if is_walkthrough_session(session):
                session["domain"] = "Walkthrough"
                logger.info(f"Walkthrough session detected: {session_id}")
            ready_sessions[session_id] = session
            logger.info(f"✅ Ready session: {session_id} ({session['completion_method']})")
        else:
            not_ready_sessions[session_id] = session
            logger.info(f"⏳ Not ready: {session_id} (no completion indicator)")


    logger.info(f"📊 Discovery complete: {len(ready_sessions)} ready, {len(not_ready_sessions)} not ready")
    
    return ready_sessions, not_ready_sessions

class BasicDownloadManager:
    """
    Basic download manager for multi-chunk files.
    Downloads chunks sequentially with progress tracking.
    """
    
    def __init__(self, blob_service_client: BlobServiceClient, container_name: str, local_download_dir: str):
        self.blob_service_client = blob_service_client
        self.container_name = container_name
        self.local_download_dir = local_download_dir
        self.download_stats = {
            "total_downloads": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_bytes": 0,
            "start_time": None,
            "end_time": None
        }
        
        # Ensure download directory exists
        os.makedirs(local_download_dir, exist_ok=True)
        logger.info(f"📁 Download directory: {local_download_dir}")
    
    def download_single_chunk(self, chunk_info: Dict, max_retries: int = 3) -> Dict:
        """
        Download a single chunk file with retry logic.
        
        Args:
            chunk_info: Chunk information from discovery
            max_retries: Maximum retry attempts
            
        Returns:
            Dict with download result
        """
        blob_name = chunk_info["blob_name"]
        file_name = chunk_info["file_name"]
        num = chunk_info.get("chunk_number")
        num_str = f"{num:02d}" if isinstance(num, int) else "NA"
        chunk_id = f"{num_str}-{chunk_info['chunk_type']}"
        
        # Create local file path
        local_file_path = os.path.join(self.local_download_dir, file_name)
        
        logger.info(f"📥 Downloading {file_name} (chunk {chunk_id})")
        
        # Check if file already exists and is complete
        if os.path.exists(local_file_path):
            local_size = os.path.getsize(local_file_path)
            remote_size = chunk_info.get("size", 0)
            
            if local_size == remote_size and local_size > 0:
                logger.info(f"⏭️ File already exists and complete: {file_name}")
                return {
                    "success": True,
                    "chunk_id": chunk_id,
                    "local_path": local_file_path,
                    "blob_name": blob_name,
                    "file_name": file_name,
                    "size_bytes": local_size,
                    "download_time": 0,
                    "already_existed": True,
                    "chunk_type": chunk_info["chunk_type"],
                    "chunk_number": chunk_info.get("chunk_number"),
                }
        
        # Download with retries
        last_error = None
        download_start = time.time()
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"  📥 Attempt {attempt + 1}/{max_retries}")
                
                # Get blob client
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name, 
                    blob=blob_name
                )
                
                # Download to local file
                with open(local_file_path, "wb") as download_file:
                    download_stream = blob_client.download_blob()
                    download_file.write(download_stream.readall())
                
                # Verify download
                if os.path.exists(local_file_path):
                    local_size = os.path.getsize(local_file_path)
                    remote_size = chunk_info.get("size", 0)
                    
                    if local_size > 0 and (remote_size == 0 or local_size == remote_size):
                        download_time = time.time() - download_start
                        logger.info(f"✅ Downloaded {file_name} ({local_size:,} bytes in {download_time:.1f}s)")
                        
                        return {
                            "success": True,
                            "chunk_id": chunk_id,
                            "local_path": local_file_path,
                            "blob_name": blob_name,
                            "file_name": file_name,
                            "size_bytes": local_size,
                            "download_time": download_time,
                            "already_existed": False,
                            "chunk_type": chunk_info["chunk_type"],
                            "chunk_number": chunk_info.get("chunk_number"),
                            "attempts": attempt + 1
                        }
                    else:
                        last_error = f"Size mismatch: local={local_size}, remote={remote_size}"
                        logger.warning(f"⚠️ {last_error}")
                
            except ResourceNotFoundError:
                last_error = f"Blob not found: {blob_name}"
                logger.error(f"❌ {last_error}")
                break  # Don't retry for missing files
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"⚠️ Download attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # All attempts failed
        logger.error(f"❌ Failed to download {file_name} after {max_retries} attempts: {last_error}")
        
        # Clean up partial file
        if os.path.exists(local_file_path):
            try:
                os.remove(local_file_path)
            except:
                pass
        
        return {
            "success": False,
            "chunk_id": chunk_id,
            "local_path": None,
            "blob_name": blob_name,
            "file_name": file_name,
            "error": last_error,
            "chunk_type": chunk_info["chunk_type"],
            "chunk_number": chunk_info.get("chunk_number"),
            "attempts": max_retries
        }
    
    def download_session_chunks(self, session_data: Dict) -> Dict:
        """
        Download all chunks for a session sequentially.
        
        Args:
            session_data: Session information from discovery
            
        Returns:
            Dict with download results for all chunks
        """
        session_id = session_data["session_id"]
        video_chunks = session_data.get("video_chunks", [])
        audio_chunks = session_data.get("audio_chunks", [])
        
        logger.info(f"📦 Downloading session: {session_id}")
        logger.info(f"   📹 Video chunks: {len(video_chunks)}")
        logger.info(f"   🎵 Audio chunks: {len(audio_chunks)}")
        
        # Reset stats
        self.download_stats = {
            "total_downloads": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_bytes": 0,
            "start_time": time.time(),
            "end_time": None
        }
        
        download_results = {
            "session_id": session_id,
            "video_downloads": {},
            "audio_downloads": {},
            "success": False,
            "download_stats": None
        }
        
        # Download video chunks
        for chunk_info in video_chunks:
            self.download_stats["total_downloads"] += 1
            
            # Extract chunk metadata for download
            chunk_download_info = {
                **chunk_info["chunk_data"],  # includes file_name, chunk_type, chunk_number, etc.
                "blob_name": chunk_info["chunk_data"]["blob_name"],
                "size": chunk_info["chunk_data"].get("size", 0),
            }
            
            result = self.download_single_chunk(chunk_download_info)
            chunk_id = result["chunk_id"]
            download_results["video_downloads"][chunk_id] = result
            
            if result["success"]:
                self.download_stats["successful_downloads"] += 1
                self.download_stats["total_bytes"] += result["size_bytes"]
            else:
                self.download_stats["failed_downloads"] += 1
        
        # Download audio chunks
        for chunk_info in audio_chunks:
            self.download_stats["total_downloads"] += 1
            
            # Extract chunk metadata for download
            chunk_download_info = {
                **chunk_info["chunk_data"],  # includes file_name, chunk_type, chunk_number, etc.
                "blob_name": chunk_info["chunk_data"]["blob_name"],
                "size": chunk_info["chunk_data"].get("size", 0),
            }
            
            result = self.download_single_chunk(chunk_download_info)
            chunk_id = result["chunk_id"]
            download_results["audio_downloads"][chunk_id] = result
            
            if result["success"]:
                self.download_stats["successful_downloads"] += 1
                self.download_stats["total_bytes"] += result["size_bytes"]
            else:
                self.download_stats["failed_downloads"] += 1
        
        # Finalize stats
        self.download_stats["end_time"] = time.time()
        download_time = self.download_stats["end_time"] - self.download_stats["start_time"]
        
        # Determine overall success
        total_chunks = len(video_chunks) + len(audio_chunks)
        success_rate = self.download_stats["successful_downloads"] / max(total_chunks, 1)
        download_results["success"] = success_rate >= 0.8  # 80% success rate threshold
        
        # Add stats to result
        download_results["download_stats"] = {
            **self.download_stats,
            "download_time": download_time,
            "success_rate": success_rate,
            "mb_per_second": (self.download_stats["total_bytes"] / (1024 * 1024)) / max(download_time, 1)
        }
        
        logger.info(f"📊 Session download complete: {session_id}")
        logger.info(f"   ✅ Success: {self.download_stats['successful_downloads']}/{total_chunks}")
        logger.info(f"   📈 Rate: {success_rate:.1%}")
        logger.info(f"   💾 Data: {self.download_stats['total_bytes'] / (1024 * 1024):.1f} MB")
        logger.info(f"   ⏱️ Time: {download_time:.1f}s")
        
        return download_results
    
    def download_session_chunks_parallel(self, session_data: Dict) -> Dict:
        """
        Download all chunks for a session in parallel using Ray.
        
        Args:
            session_data: Session information from discovery
            
        Returns:
            Dict with download results for all chunks
        """
        session_id = session_data["session_id"]
        video_chunks = session_data.get("video_chunks", [])
        audio_chunks = session_data.get("audio_chunks", [])
        
        logger.info(f"📦 Downloading session in PARALLEL: {session_id}")
        logger.info(f"   📹 Video chunks: {len(video_chunks)}")
        logger.info(f"   🎵 Audio chunks: {len(audio_chunks)}")
        
        # Ensure Ray is initialized
        if not ray.is_initialized():
            logger.warning("⚠️ Ray not initialized, initializing now...")
            ray.init()
        
        # Reset stats
        self.download_stats = {
            "total_downloads": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_bytes": 0,
            "start_time": time.time(),
            "end_time": None
        }
        
        download_results = {
            "session_id": session_id,
            "video_downloads": {},
            "audio_downloads": {},
            "success": False,
            "download_stats": None
        }
        
        # Prepare all chunk download tasks
        all_chunks = []
        chunk_types = []
        
        # Add video chunks
        for chunk_info in video_chunks:
            chunk_download_info = {
                **chunk_info["chunk_data"],
                "blob_name": chunk_info["chunk_data"]["blob_name"],
                "size": chunk_info["chunk_data"].get("size", 0),
            }
            all_chunks.append(chunk_download_info)
            chunk_types.append("video")
        
        # Add audio chunks
        for chunk_info in audio_chunks:
            chunk_download_info = {
                **chunk_info["chunk_data"],
                "blob_name": chunk_info["chunk_data"]["blob_name"],
                "size": chunk_info["chunk_data"].get("size", 0),
            }
            all_chunks.append(chunk_download_info)
            chunk_types.append("audio")
        
        logger.info(f"🚀 Launching {len(all_chunks)} parallel chunk download tasks")
        
        # Launch all download tasks in parallel
        download_tasks = []
        for chunk_info in all_chunks:
            task = download_single_chunk_ray.remote(
                self.blob_service_client, self.container_name, chunk_info, self.local_download_dir
            )
            download_tasks.append(task)
        
        # Wait for all downloads to complete
        logger.info(f"⏳ Waiting for {len(download_tasks)} parallel chunk downloads to complete...")
        chunk_results = ray.get(download_tasks)
        
        # Organize results by type
        video_downloads = {}
        audio_downloads = {}
        
        for i, result in enumerate(chunk_results):
            chunk_type = chunk_types[i]
            chunk_id = result["chunk_id"]
            
            if chunk_type == "video":
                video_downloads[chunk_id] = result
            else:
                audio_downloads[chunk_id] = result
            
            # Update stats
            self.download_stats["total_downloads"] += 1
            if result["success"]:
                self.download_stats["successful_downloads"] += 1
                self.download_stats["total_bytes"] += result.get("size_bytes", 0)
            else:
                self.download_stats["failed_downloads"] += 1
        
        # Finalize stats
        self.download_stats["end_time"] = time.time()
        download_time = self.download_stats["end_time"] - self.download_stats["start_time"]
        
        # Determine overall success
        total_chunks = len(all_chunks)
        success_rate = self.download_stats["successful_downloads"] / max(total_chunks, 1)
        download_results["success"] = success_rate >= 0.8  # 80% success rate threshold
        
        # Add stats to result
        download_results["download_stats"] = {
            **self.download_stats,
            "download_time": download_time,
            "success_rate": success_rate,
            "mb_per_second": (self.download_stats["total_bytes"] / (1024 * 1024)) / max(download_time, 1),
            "parallel_processing": True
        }
        
        # Update results
        download_results["video_downloads"] = video_downloads
        download_results["audio_downloads"] = audio_downloads
        
        logger.info(f"📊 PARALLEL Session download complete: {session_id}")
        logger.info(f"   ✅ Success: {self.download_stats['successful_downloads']}/{total_chunks}")
        logger.info(f"   📈 Rate: {success_rate:.1%}")
        logger.info(f"   💾 Data: {self.download_stats['total_bytes'] / (1024 * 1024):.1f} MB")
        logger.info(f"   ⏱️ Time: {download_time:.1f}s")
        logger.info(f"   🚀 Parallel processing: ENABLED")
        
        return download_results
    
    def cleanup_downloads(self, keep_successful: bool = True):
        """
        Clean up downloaded files.
        
        Args:
            keep_successful: If True, only delete failed downloads
        """
        if not os.path.exists(self.local_download_dir):
            return
        
        logger.info(f"🧹 Cleaning up downloads in {self.local_download_dir}")
        
        try:
            if keep_successful:
                logger.info("   Keeping successful downloads, manual cleanup required")
            else:
                import shutil
                shutil.rmtree(self.local_download_dir, ignore_errors=True)
                logger.info("   All downloads cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

class DownloadProgressTracker:
    """
    Track download progress and save state to disk.
    Allows resuming interrupted downloads.
    """
    
    def __init__(self, session_id: str, progress_dir: str):
        self.session_id = session_id
        self.progress_dir = progress_dir
        self.progress_file = os.path.join(progress_dir, f"{session_id}_download_progress.json")
        
        os.makedirs(progress_dir, exist_ok=True)
        
        # Initialize or load existing progress
        self.progress_data = self._load_progress()
    
    def _load_progress(self) -> Dict:
        """Load existing progress or create new"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"📊 Loaded existing download progress for {self.session_id}")
                return data
            except Exception as e:
                logger.warning(f"⚠️ Failed to load progress, starting fresh: {e}")
        
        # Create new progress structure
        return {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "download_status": "not_started",  # not_started, in_progress, completed, failed
            "chunks": {},
            "stats": {
                "total_chunks": 0,
                "completed_chunks": 0,
                "failed_chunks": 0,
                "total_bytes": 0,
                "downloaded_bytes": 0
            }
        }
    
    def _save_progress(self):
        """Save progress to disk"""
        try:
            self.progress_data["last_updated"] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save progress: {e}")
    
    def initialize_session(self, session_data: Dict):
        """Initialize progress tracking for a session"""
        video_chunks = session_data.get("video_chunks", [])
        audio_chunks = session_data.get("audio_chunks", [])
        
        # Initialize chunk tracking
        for chunk_info in video_chunks + audio_chunks:
            num = chunk_info.get("chunk_number")
            num_str = f"{num:02d}" if isinstance(num, int) else "NA"
            chunk_id = f"{num_str}-{chunk_info['chunk_data']['chunk_type']}"
            
            if chunk_id not in self.progress_data["chunks"]:
                self.progress_data["chunks"][chunk_id] = {
                    "chunk_id": chunk_id,
                    "file_name": chunk_info["file_name"],
                    "blob_name": chunk_info.get("blob_name", chunk_info["file_name"]),
                    "chunk_type": chunk_info["chunk_data"]["chunk_type"],
                    "chunk_number": chunk_info.get("chunk_number"),
                    "size_bytes": chunk_info.get("size", 0),
                    "download_status": "pending",  # pending, downloading, completed, failed
                    "local_path": None,
                    "download_start": None,
                    "download_end": None,
                    "error": None,
                    "attempts": 0
                }
        
        # Update stats
        self.progress_data["stats"]["total_chunks"] = len(self.progress_data["chunks"])
        self.progress_data["stats"]["total_bytes"] = sum(
            chunk["size_bytes"] for chunk in self.progress_data["chunks"].values()
        )
        
        self.progress_data["download_status"] = "initialized"
        self._save_progress()
        
        logger.info(f"📊 Initialized progress tracking for {len(self.progress_data['chunks'])} chunks")
    
    def update_chunk_status(self, chunk_id: str, status: str, **kwargs):
        """Update status of a specific chunk"""
        if chunk_id not in self.progress_data["chunks"]:
            logger.warning(f"⚠️ Unknown chunk ID: {chunk_id}")
            return
        
        chunk_data = self.progress_data["chunks"][chunk_id]
        chunk_data["download_status"] = status
        
        # Update specific fields
        for key, value in kwargs.items():
            if key in chunk_data:
                chunk_data[key] = value
        
        # Update timestamps
        if status == "downloading":
            chunk_data["download_start"] = datetime.now().isoformat()
            chunk_data["attempts"] = chunk_data.get("attempts", 0) + 1
        elif status in ["completed", "failed"]:
            chunk_data["download_end"] = datetime.now().isoformat()
        
        # Update overall stats
        self._update_stats()
        self._save_progress()
        
        logger.debug(f"📊 Updated {chunk_id}: {status}")
    
    def _update_stats(self):
        """Update overall statistics"""
        stats = self.progress_data["stats"]
        chunks = self.progress_data["chunks"].values()
        
        stats["completed_chunks"] = len([c for c in chunks if c["download_status"] == "completed"])
        stats["failed_chunks"] = len([c for c in chunks if c["download_status"] == "failed"])
        stats["downloaded_bytes"] = sum(
            c["size_bytes"] for c in chunks if c["download_status"] == "completed"
        )
        
        # Update overall status
        if stats["completed_chunks"] == stats["total_chunks"]:
            self.progress_data["download_status"] = "completed"
        elif stats["failed_chunks"] > 0:
            if stats["completed_chunks"] + stats["failed_chunks"] == stats["total_chunks"]:
                self.progress_data["download_status"] = "completed_with_errors"
            else:
                self.progress_data["download_status"] = "in_progress"
        elif stats["completed_chunks"] > 0:
            self.progress_data["download_status"] = "in_progress"
    
    def get_progress_summary(self) -> Dict:
        """Get current progress summary"""
        stats = self.progress_data["stats"]
        
        progress_percentage = 0
        if stats["total_chunks"] > 0:
            progress_percentage = (stats["completed_chunks"] / stats["total_chunks"]) * 100
        
        bytes_percentage = 0
        if stats["total_bytes"] > 0:
            bytes_percentage = (stats["downloaded_bytes"] / stats["total_bytes"]) * 100
        
        return {
            "session_id": self.session_id,
            "download_status": self.progress_data["download_status"],
            "progress_percentage": progress_percentage,
            "bytes_percentage": bytes_percentage,
            "chunks_completed": stats["completed_chunks"],
            "chunks_total": stats["total_chunks"],
            "chunks_failed": stats["failed_chunks"],
            "bytes_downloaded": stats["downloaded_bytes"],
            "bytes_total": stats["total_bytes"],
            "last_updated": self.progress_data["last_updated"]
        }
    
    def get_failed_chunks(self) -> List[str]:
        """Get list of failed chunk IDs for retry"""
        return [
            chunk_id for chunk_id, chunk_data in self.progress_data["chunks"].items()
            if chunk_data["download_status"] == "failed"
        ]
    
    def can_resume(self) -> bool:
        """Check if download can be resumed"""
        return (
            self.progress_data["download_status"] in ["in_progress", "completed_with_errors"] and
            len(self.get_failed_chunks()) > 0
        )

def download_ready_sessions(blob_service_client: BlobServiceClient,
                          container_name: str,
                          input_prefix: str,
                          local_download_dir: str,
                          max_sessions: int = None,
                          use_parallel_chunks: bool = True) -> Dict:
    """
    Complete workflow: Discover sessions and download chunks.
    This combines Phase 1 (discovery) and Phase 2 (download).
    """
    logger.info(f"🚀 Starting session discovery and download")
    logger.info(f"   📁 Container: {container_name}")
    logger.info(f"   📁 Prefix: {input_prefix}")
    logger.info(f"   📁 Download to: {local_download_dir}")
    
    workflow_start = time.time()
    
    # Step 1: Discover sessions (Phase 1)
    logger.info("🔍 Step 1: Discovering sessions...")
    ready_sessions, not_ready_sessions = discover_sessions_basic(
        blob_service_client, container_name, input_prefix
    )
    
    if not ready_sessions:
        logger.info(f"📭 No ready sessions found. {len(not_ready_sessions)} incomplete sessions.")
        return {
            "success": True,
            "sessions_discovered": len(ready_sessions) + len(not_ready_sessions),
            "sessions_ready": 0,
            "sessions_downloaded": 0,
            "message": "No complete sessions ready for download"
        }
    
    # Limit sessions if specified
    if max_sessions and len(ready_sessions) > max_sessions:
        session_items = list(ready_sessions.items())[:max_sessions]
        ready_sessions = dict(session_items)
        logger.info(f"📊 Limited to first {max_sessions} sessions")
    
    # Step 2: Download sessions (Phase 2)
    logger.info(f"📥 Step 2: Downloading {len(ready_sessions)} sessions...")
    
    download_manager = BasicDownloadManager(
        blob_service_client, container_name, local_download_dir
    )
    
    download_results = {}
    successful_downloads = 0
    failed_downloads = 0
    
    for i, (session_id, session_data) in enumerate(ready_sessions.items(), 1):
        logger.info(f"📦 Downloading session {i}/{len(ready_sessions)}: {session_id}")
        
        # Create progress tracker
        progress_tracker = DownloadProgressTracker(session_id, f"{local_download_dir}/../progress")
        progress_tracker.initialize_session(session_data)
        
        try:
            # Download session (sequential or parallel chunks)
            if use_parallel_chunks:
                download_result = download_manager.download_session_chunks_parallel(session_data)
            else:
                download_result = download_manager.download_session_chunks(session_data)
            
            # Update progress tracker
            for chunk_id, chunk_result in download_result["video_downloads"].items():
                status = "completed" if chunk_result["success"] else "failed"
                progress_tracker.update_chunk_status(
                    chunk_id, status,
                    local_path=chunk_result.get("local_path"),
                    error=chunk_result.get("error")
                )
            
            for chunk_id, chunk_result in download_result["audio_downloads"].items():
                status = "completed" if chunk_result["success"] else "failed"
                progress_tracker.update_chunk_status(
                    chunk_id, status,
                    local_path=chunk_result.get("local_path"),
                    error=chunk_result.get("error")
                )
            
            # Store result with session metadata
            download_results[session_id] = {
                "success": download_result["success"],
                "download_result": download_result,
                "progress_summary": progress_tracker.get_progress_summary(),
                "session_metadata": session_data.get("session_metadata")  # ← ADD SESSION METADATA
            }
            
            if download_result["success"]:
                successful_downloads += 1
                logger.info(f"✅ Session downloaded successfully: {session_id}")
            else:
                failed_downloads += 1
                logger.error(f"❌ Session download failed: {session_id}")
                
        except Exception as e:
            failed_downloads += 1
            logger.error(f"❌ Session download error {session_id}: {e}")
            download_results[session_id] = {
                "success": False,
                "error": str(e),
                "progress_summary": progress_tracker.get_progress_summary(),
                "session_metadata": session_data.get("session_metadata")  # ← ADD SESSION METADATA
            }
    
    # Final summary
    total_time = time.time() - workflow_start
    success_rate = successful_downloads / len(ready_sessions)
    
    result = {
        "success": successful_downloads > 0,
        "sessions_discovered": len(ready_sessions) + len(not_ready_sessions),
        "sessions_ready": len(ready_sessions),
        "sessions_downloaded": successful_downloads,
        "sessions_failed": failed_downloads,
        "success_rate": success_rate,
        "total_processing_time": total_time,
        "download_results": download_results,
        "incomplete_sessions": len(not_ready_sessions)
    }
    
    logger.info(f"🎉 Download workflow complete:")
    logger.info(f"   ✅ Success: {successful_downloads}/{len(ready_sessions)} sessions")
    logger.info(f"   📈 Rate: {success_rate:.1%}")
    logger.info(f"   ⏱️ Time: {total_time:.1f}s")
    
    return result


# =============================================================================
# RAY PARALLEL DOWNLOAD FUNCTIONS
# =============================================================================

@ray.remote
def download_single_chunk_ray(blob_service_client: BlobServiceClient, 
                            container_name: str, 
                            chunk_info: Dict, 
                            local_download_dir: str,
                            max_retries: int = 3) -> Dict:
    """
    Ray remote task to download a single chunk file with retry logic.
    
    Args:
        blob_service_client: Azure blob service client
        container_name: Azure container name
        chunk_info: Chunk information from discovery
        local_download_dir: Local directory to download files
        max_retries: Maximum retry attempts
        
    Returns:
        Dict with download result
    """
    try:
        import ray
        ray_worker_id = ray.get_runtime_context().get_worker_id()
        
        blob_name = chunk_info["blob_name"]
        file_name = chunk_info["file_name"]
        num = chunk_info.get("chunk_number")
        num_str = f"{num:02d}" if isinstance(num, int) else "NA"
        chunk_id = f"{num_str}-{chunk_info['chunk_type']}"
        
        # Create local file path
        local_file_path = os.path.join(local_download_dir, file_name)
        
        logger.info(f"📥 RAY WORKER {ray_worker_id} - Downloading {file_name} (chunk {chunk_id})")
        
        # Check if file already exists and is complete
        if os.path.exists(local_file_path):
            local_size = os.path.getsize(local_file_path)
            remote_size = chunk_info.get("size", 0)
            
            if local_size == remote_size and local_size > 0:
                logger.info(f"⏭️ RAY WORKER {ray_worker_id} - File already exists and complete: {file_name}")
                return {
                    "success": True,
                    "chunk_id": chunk_id,
                    "local_path": local_file_path,
                    "size_bytes": local_size,
                    "message": "File already exists and complete",
                    "ray_worker_id": ray_worker_id
                }
        
        # Download with retry logic
        for attempt in range(max_retries):
            try:
                # Get blob client
                blob_client = blob_service_client.get_blob_client(
                    container=container_name, 
                    blob=blob_name
                )
                
                # Download blob
                with open(local_file_path, "wb") as download_file:
                    download_stream = blob_client.download_blob()
                    download_file.write(download_stream.readall())
                
                # Verify download
                local_size = os.path.getsize(local_file_path)
                remote_size = chunk_info.get("size", 0)
                
                if remote_size > 0 and local_size != remote_size:
                    raise Exception(f"Size mismatch: local={local_size}, remote={remote_size}")
                
                logger.info(f"✅ RAY WORKER {ray_worker_id} - Successfully downloaded {file_name} ({local_size} bytes)")
                
                return {
                    "success": True,
                    "chunk_id": chunk_id,
                    "local_path": local_file_path,
                    "size_bytes": local_size,
                    "attempts": attempt + 1,
                    "ray_worker_id": ray_worker_id
                }
                
            except Exception as e:
                logger.warning(f"⚠️ RAY WORKER {ray_worker_id} - Download attempt {attempt + 1} failed for {file_name}: {e}")
                
                # Clean up partial file
                if os.path.exists(local_file_path):
                    try:
                        os.remove(local_file_path)
                    except:
                        pass
                
                if attempt == max_retries - 1:
                    # Final attempt failed
                    logger.error(f"❌ RAY WORKER {ray_worker_id} - All download attempts failed for {file_name}")
                    return {
                        "success": False,
                        "chunk_id": chunk_id,
                        "error": str(e),
                        "attempts": max_retries,
                        "ray_worker_id": ray_worker_id
                    }
                
                # Wait before retry
                time.sleep(2 ** attempt)  # Exponential backoff
        
    except Exception as e:
        logger.error(f"❌ RAY WORKER - Unexpected error downloading {chunk_info.get('file_name', 'unknown')}: {e}")
        return {
            "success": False,
            "chunk_id": chunk_info.get("chunk_id", "unknown"),
            "error": str(e),
            "ray_worker_id": ray.get_runtime_context().get_worker_id() if ray.is_initialized() else "unknown"
        }



